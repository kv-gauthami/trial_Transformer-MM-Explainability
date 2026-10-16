
#Tie the pipeline together and save directly to attention_heatmaps.pkl for downstream AUC calculation:

import os
import sys
import pickle
import torch

# Add parent directory (trial_Transformer-MM-Explainability) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from llava_explainability.hook_handler import AttentionHookManager
from llava_explainability.explainer import LlavaExplainer

# Load model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, # float32 ensures stable gradients during hook operations
    device_map="cuda",
    attn_implementation="eager"
)

# out_dir = "/rdata/mithun/gauthamikv/Relevancy_Maps/trial_Transformer-MM-Explainability/llava_folder"
# Dynamically get the directory where this script is located
out_dir = os.path.dirname(os.path.abspath(__file__))

hook_mgr = AttentionHookManager(model)
explainer = LlavaExplainer(model, hook_mgr)

# Example sample loading
image = Image.open("man_dog_1.png")
prompt = "USER: <image>\nWhat is in the image? ASSISTANT: A dog"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

# Locate target token index and image patch boundaries
input_ids = inputs["input_ids"][0].tolist()

# Dynamically find exactly where the 576 image tokens begin
IMAGE_TOKEN_ID = model.config.image_token_index
img_start_idx = input_ids.index(IMAGE_TOKEN_ID)

target_token_idx = len(input_ids) - 1 # Index for the target token

# *******************************************************************************
# heatmap_24x24 = explainer.generate_relevance(
#     inputs, 
#     target_token_idx=target_token_idx, 
#     img_start_idx=img_start_idx,
#     img_num_tokens=576
# )

# # Save dict in the exact structure expected by auroc_downstream.py
# # output_data = {
# #     "heatmap": heatmap_24x24,
# #     "target_token": "dog"
# # }
# output_data = {
#     (0, "dog", 32): heatmap_dog_24x24,  # prompt_idx=0, word="dog", layer=32 (or "relevancy")
#     (0, "man", 32): heatmap_man_24x24,  # prompt_idx=0, word="man", layer=32
# }

# with open(os.path.join(out_dir, "attention_heatmaps.pkl"), "wb") as f:
#     pickle.dump(all_heatmaps, f)
# print(f"\nSaved {len(all_heatmaps)} heatmaps -> {out_dir}/attention_heatmaps.pkl")

# print("Saved attention_heatmaps.pkl successfully.")
# *******************************************************************************


# Generate the heatmap for the target token ('dog')
heatmap_dog_24x24 = explainer.generate_relevance(
    inputs, 
    target_token_idx=target_token_idx, 
    img_start_idx=img_start_idx,
    img_num_tokens=576
)

# Save dict in the exact structure expected by auroc_downstream.py
output_data = {
    (0, ["dog", "man"], ['0', '4', '8', '12', '16', '20', '24', '28']): heatmap_dog_24x24,  # prompt_idx=0, word="dog", layer=32
    # Note: We are only saving 'dog' for now since we only generated one map.
    # auroc_downstream.py will evaluate 'dog' and safely skip 'man' if it's missing from the pickle.
}

# Fix: Use 'output_data' instead of 'all_heatmaps'
with open(os.path.join(out_dir, "attention_heatmaps.pkl"), "wb") as f:
    pickle.dump(output_data, f)
    
print(f"\nSaved {len(output_data)} heatmaps -> {os.path.join(out_dir, 'attention_heatmaps.pkl')}")