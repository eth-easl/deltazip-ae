import os
import signal
import asyncio
import inspect
import importlib
from contextlib import asynccontextmanager
from http import HTTPStatus
import time
import fastapi
import uvicorn
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ReloadRequest,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.swap.request import find_swap_model

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
reload_lock = asyncio.Lock()
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


@app.get("/sysinfo")
async def get_sysinfo():
    engine_info = engine_args.to_json()
    cli_args = {
        "swap_modules": [swap_module.to_json() for swap_module in args.swap_modules],
        "lora_modules": [lora_module.to_json() for lora_module in args.lora_modules],
        "delta_modules": [
            delta_module.to_json() for delta_module in args.delta_modules
        ],
    }
    engine_info.update(cli_args)
    engine_info.update({"pid": os.getpid()})
    return JSONResponse(content=engine_info)

@app.get("/kill")
async def kill():
    os.kill(os.getpid(), signal.SIGTERM)
    

@app.post("/v1/reload")
async def reload_model_weights(request: ReloadRequest):
    model_id, found_model = find_swap_model(
        served_model, request.target, args.swap_modules
    )
    if found_model:
        while engine.engine.has_running_requests():
            time.sleep(0.1)
        await reload_lock.acquire()
        await engine.reload_model(model_id, found_model)
        reload_lock.release()
        return JSONResponse(content={"message": "Model reloaded"})
    else:
        return JSONResponse(
            content={
                "error": f"Model not found, requested: {request.model}, available: {[x.name for x in args.swap_modules] + [served_model]}"
            },
            status_code=404,
        )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    arrival_time = time.time()
    # wait until the engine finishes reloading
    generator = await openai_serving_completion.create_completion(
        request,
        raw_request,
        arrival_time=arrival_time,
    )
    if isinstance(generator, ErrorResponse):
        response = JSONResponse(
            content=generator.model_dump(), status_code=generator.code
        )
    if request.stream:
        response = StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        response = JSONResponse(content=generator.model_dump())
    return response


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if not request.url.path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. " f"Must be a function or a class."
            )

    logger.info(f"vLLM API server version {vllm.__version__}")
    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    openai_serving_chat = OpenAIServingChat(
        engine,
        served_model,
        args.response_role,
        args.lora_modules,
        args.delta_modules,
        args.swap_modules,
        args.chat_template,
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine,
        served_model,
        args.lora_modules,
        args.delta_modules,
        args.swap_modules,
    )
    app.root_path = args.root_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
