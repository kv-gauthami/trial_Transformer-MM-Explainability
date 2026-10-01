"""
Layer-wise A and R maps for LLaVA-1.5 (row t-1 convention).

Example:
  python llava_folder/run_layerwise_maps.py \
      --image llava_folder/man_dog_1.png \
      --prompt "USER: <image>\nWhat is in the image? ASSISTANT: A man and a dog." \
      --words man dog --out_dir llava_folder/man_and_dog
"""
import argparse
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llava_explainability.pipeline import prepare_model_for_relevance, run_layerwise_maps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--image", default=os.path.join(HERE, "man_dog_1.png"))
    p.add_argument("--prompt", default="USER: <image>\nWhat is in the image? ASSISTANT: A man and a dog.")
    p.add_argument("--words", nargs="+", default=["man", "dog"])
    p.add_argument("--occurrence", default="last", help="first | last | <int> among matches in the response")
    p.add_argument("--prompt_idx", type=int, default=0)
    p.add_argument("--out_dir", default=os.path.join(HERE, "man_and_dog"))
    p.add_argument("--save_name", default="relevancy_maps")
    p.add_argument("--dtype", default="float32", choices=list(DTYPES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.prompt = args.prompt.replace("\\n", "\n")   # allow literal \n from the shell
    torch.manual_seed(0)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=DTYPES[args.dtype], device_map=args.device,
        attn_implementation="eager")
    prepare_model_for_relevance(model)

    image = Image.open(args.image).convert("RGB")
    inputs = processor(text=args.prompt, images=image, return_tensors="pt").to(args.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(DTYPES[args.dtype])

    run_layerwise_maps(
        model, processor.tokenizer, inputs, args.words, args.out_dir,
        prompt_idx=args.prompt_idx, occurrence=args.occurrence, save_name=args.save_name,
        make_plots=not args.no_plots,
        extra_meta={"model_id": args.model_id, "image": os.path.abspath(args.image),
                    "prompt": args.prompt, "dtype": args.dtype},
    )


if __name__ == "__main__":
    main()
