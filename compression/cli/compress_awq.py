import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def compress(args):
    model_path = args.model
    compressed_name = "awq." + args.model.replace("/", ".") + ".4b128g"
    quant_path = os.path.join(args.outdir, compressed_name)
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(model_path)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    compress(parser.parse_args())