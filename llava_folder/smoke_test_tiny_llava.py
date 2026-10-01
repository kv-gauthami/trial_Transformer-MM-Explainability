"""CPU smoke test (no downloads): random tiny LLaVA with the real 576-token layout.
Checks hooks, indexing, shapes, save format, plots, and that row t has zero gradient.
Run: python llava_folder/smoke_test_tiny_llava.py
"""
import sys, os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from PIL import Image
from transformers import LlavaConfig, LlavaForConditionalGeneration, CLIPVisionConfig, LlamaConfig
from llava_explainability.pipeline import prepare_model_for_relevance, run_layerwise_maps
from llava_explainability.storage import load_maps, select
from llava_explainability.hook_handler import AttentionHookManager
from llava_explainability.explainer import LlavaLayerwiseExplainer

torch.manual_seed(0)
OUT = os.path.join(HERE, "_smoke_test_out")
vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=336, patch_size=14)
tc = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, vocab_size=32064)
cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=32000, vision_feature_select_strategy="default", vision_feature_layer=-2)
model = LlavaForConditionalGeneration(cfg)
model.config._attn_implementation = "eager"
model.set_attn_implementation("eager") if hasattr(model,"set_attn_implementation") else None
prepare_model_for_relevance(model)

VOCAB = {"▁man": 767, "▁and": 322, "▁a": 263, "▁dog": 11203, ".": 29889, "▁": 29871}
INV = {v:k for k,v in VOCAB.items()}
class MockTok:
    def encode(self, text, add_special_tokens=False):
        w = text.strip(); return ([29871] if text.startswith(" ") else []) + [VOCAB["▁"+w]]
    def convert_ids_to_tokens(self, i):
        return [INV.get(x, f"<{x}>") for x in i] if isinstance(i, list) else INV.get(i, f"<{i}>")

ids = [1, 3148, 1001, 29901] + [32000]*576 + [13, 5618, 338, 297, 278, 1967, 29973, 319, 1799, 29901, 767, 322, 263, 11203, 29889]
img = Image.open(os.path.join(HERE, "man_dog_1.png")).convert("RGB")
W,H = img.size; s = 336/min(W,H); img = img.resize((round(W*s), round(H*s)), Image.BICUBIC)
W,H = img.size; l,t = (W-336)//2, (H-336)//2; img = img.crop((l,t,l+336,t+336))
x = (np.asarray(img)/255.0 - [0.4815,0.4578,0.4082])/[0.2686,0.2613,0.2758]
inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1,len(ids),dtype=torch.long),
          "pixel_values": torch.tensor(x, dtype=torch.float32).permute(2,0,1)[None]}

rec, meta = run_layerwise_maps(model, MockTok(), inputs, ["man","dog"], OUT)
print(meta["words"], meta["grid"], meta["num_layers"])
r = load_maps(os.path.join(OUT, "relevancy_maps.pkl"))
print(len(r), sorted(set(k[2] for k in r)))
for mt in ["A","R_layer","R_cum","R_final"]:
    d = select(r, 0, "dog", mt); v = list(d.values())
    print(mt, list(d.keys()), [f"{a.std():.2e}" for a in v])
# final == last cum
assert np.allclose(select(r,0,"dog","R_final")[4], select(r,0,"dog","R_cum")[3])
# check: row t (old code) is identically zero, row t-1 is not
hm = AttentionHookManager(model); ex = LlavaLayerwiseExplainer(model, hm); out = ex.forward(inputs)
t = ids.index(11203); g = torch.autograd.grad(out.logits[0,t-1,11203], hm.attentions, retain_graph=True)
g_t = max(x[0,:,t].abs().max().item() for x in g)
assert g_t == 0.0
print("grad row t max:", max(x[0,:,t].abs().max().item() for x in g), " row t-1 max:", max(x[0,:,t-1].abs().max().item() for x in g))
hm.remove()
print("SMOKE TEST PASSED")
