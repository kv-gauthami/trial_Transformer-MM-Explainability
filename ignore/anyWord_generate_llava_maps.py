import os
import sys
import pickle
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llava_explainability.hook_handler import AttentionHookManager
from llava_explainability.explainer import LlavaExplainer

# -------------------------------- CONFIG --------------------------------
IMAGE_PATH   = "/rdata/mithun/gauthamikv/trial_Transformer-MM-Explainability/llava_folder/man_dog_1.png"
OUT_DIR      = "/rdata/mithun/gauthamikv/trial_Transformer-MM-Explainability/llava_folder"
TARGET_WORDS = ["dog", "man"]
PROMPT       = "USER: <image>\nWhat is in the image? ASSISTANT: A man and a dog."

# -------------------------------- MODEL ---------------------------------
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    device_map="cuda"
)

hook_mgr = AttentionHookManager(model)
explainer = LlavaExplainer(model, hook_mgr)

# ---------------------------- PREPROCESSING -----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")

input_ids = inputs["input_ids"][0].tolist()

# Locate image token span (LLaVA-1.5 inserts 576 tokens for <image>)
IMAGE_TOKEN_ID = model.config.image_token_index
img_start_idx = input_ids.index(IMAGE_TOKEN_ID)
img_num_tokens = 576

# Helper: locate token index for each target word in the prompt
def find_token_index(word: str, input_ids: list, tokenizer) -> int:
    """Finds the last matching sub-token index for a target word."""
    # LLaVA tokenizer uses leading spaces (e.g., ' dog' vs 'dog')
    token_candidates = tokenizer.encode(f" {word}", add_special_tokens=False)
    target_id = token_candidates[-1]
    
    # Reverse search to find token in the ASSISTANT response
    for idx in reversed(range(len(input_ids))):
        if input_ids[idx] == target_id:
            return idx
    raise ValueError(f"Target word '{word}' not found in prompt token IDs.")

# ------------------------ MULTI-WORD GENERATION -------------------------
output_data = {}

for word in TARGET_WORDS:
    token_idx = find_token_index(word, input_ids, processor.tokenizer)
    print(f"Generating relevancy map for '{word}' at sequence position {token_idx}...")
    
    heatmap = explainer.generate_relevance(
        inputs,
        target_token_idx=token_idx,
        img_start_idx=img_start_idx,
        img_num_tokens=img_num_tokens
    )
    
    # Key matches format expected by auroc_downstream.py: (prompt_idx, word, layer)
    output_data[(0, word, 32)] = heatmap

# -------------------------------- SAVE ----------------------------------
pkl_path = os.path.join(OUT_DIR, "attention_heatmaps.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(output_data, f)

print(f"\nSaved {len(output_data)} heatmaps for {list(output_data.keys())} -> {pkl_path}")