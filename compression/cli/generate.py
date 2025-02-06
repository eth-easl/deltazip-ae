import torch
import os
import json
from deltazip import AutoDeltaZipModelForCausalLM
import transformers
from cli.utils import generate_readme, upload_and_delete, update_chat_template

ignore_keywords = [
    'norm',
    'embed',
    'lm_head'
]

def merge(base, delta):
    base = base.bfloat16()
    delta = delta.bfloat16()
    for name, param in base.model.named_parameters():
        if any([kw in name for kw in ignore_keywords]):
            pass
        else:
            delta.model.state_dict()[name].copy_(
                param + delta.model.state_dict()[name]
            )
    delta = delta.to(torch.device("cuda"))
    return delta

def generate(target_model:str, prompt:str):
    with open(os.path.join(target_model, "delta_config.json"), "r") as fp:
        config = json.load(fp)
    model = AutoDeltaZipModelForCausalLM.from_compressed(target_model, device="cpu", strict=True, unpack=True)
    
    print(f"Loading base model {config['base_model']}...")
    base_model = AutoDeltaZipModelForCausalLM.from_pretrained(config['base_model'], None)
    
    model = merge(base_model, model)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['base_model'])
    pipe = transformers.TextGenerationPipeline(
        model=model, tokenizer=tokenizer, device="cuda"
    )
    if tokenizer.chat_template is None and update_chat_template(target_model):
        print("[warn] default chat template is not set in the tokenizer.")
        tokenizer.chat_template = update_chat_template(target_model)

    prompt = tokenizer.apply_chat_template([{
        "role": "user",
        "content": prompt
    }], add_generation_prompt=True, tokenize=False)
    outputs = pipe(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.6,
        top_k=50,
        top_p=0.9,
        return_full_text=False,
    )
    print(f"Generated: {outputs[0]['generated_text']}")
    return outputs

def main(args):
    print(args)
    generate(args.target_model, args.prompt)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="facebook/opt-125m")
    parser.add_argument("--prompt", type=str, default="Who is Alan Turing?")
    main(parser.parse_args())