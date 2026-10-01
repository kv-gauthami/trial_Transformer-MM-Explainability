"""End-to-end: one forward pass -> per-word layer-wise A/R maps -> save + plot."""
import os
import platform

import torch

from .explainer import LlavaLayerwiseExplainer
from .hook_handler import AttentionHookManager
from .storage import maps_to_records, save_maps
from .token_utils import find_word_position, image_token_span
from . import visualize as viz


def prepare_model_for_relevance(model):
    """Freeze weights; gradients flow to attention maps through the input embeddings."""
    model.eval()
    model.requires_grad_(False)
    model.enable_input_require_grads()
    if getattr(model.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("Load the model with attn_implementation='eager'.")
    return model


def run_layerwise_maps(model, tokenizer, inputs, words, out_dir, prompt_idx=0,
                       image_token_id=None, occurrence="last", scales=("per_layer", "global"),
                       save_name="relevancy_maps", extra_meta=None, make_plots=True):
    image_token_id = image_token_id if image_token_id is not None else model.config.image_token_index
    ids = inputs["input_ids"][0].tolist()
    img_start, n_img, grid = image_token_span(ids, image_token_id)
    response_from = img_start + n_img       # words are searched after the image block

    hook_mgr = AttentionHookManager(model)
    explainer = LlavaLayerwiseExplainer(model, hook_mgr)
    try:
        explainer.forward(inputs)
        image_rgb = viz.pixel_values_to_image(inputs["pixel_values"])

        records, per_word = {}, {}
        for word in words:
            t = find_word_position(ids, tokenizer, word, search_from=response_from, occurrence=occurrence)
            maps = explainer.explain_word(inputs["input_ids"], word, t, img_start, n_img, grid)
            records.update(maps_to_records(prompt_idx, maps))
            per_word[word] = {"target_pos": t, "query_pos": maps.query_pos,
                              "target_token": tokenizer.convert_ids_to_tokens(ids[t]),
                              "query_token": tokenizer.convert_ids_to_tokens(ids[t - 1]),
                              "target_prob": maps.target_prob, "target_logit": maps.target_logit}
            print(f"[{word}] t={t} row={t-1} ({per_word[word]['query_token']!r} -> "
                  f"{per_word[word]['target_token']!r})  p={maps.target_prob:.4f}")

            if make_plots:
                wdir = os.path.join(out_dir, "plots", f"p{prompt_idx}_{word}")
                for mtype, arr in (("A", maps.A), ("R_layer", maps.R_layer), ("R_cum", maps.R_cum)):
                    for sc in scales:
                        viz.plot_layer_grid(image_rgb, arr, f"{mtype} '{word}'",
                                            os.path.join(wdir, f"{mtype}_{sc}.png"), scale=sc)
                viz.plot_summary(image_rgb, maps, os.path.join(wdir, "summary.png"))
                viz.plot_final(image_rgb, maps, os.path.join(wdir, "R_final.png"))
    finally:
        hook_mgr.remove()

    import transformers
    meta = {
        "prompt_idx": prompt_idx, "words": per_word, "row_convention": "t-1",
        "image_token_start": img_start, "num_image_tokens": n_img, "grid": grid,
        "num_layers": hook_mgr.num_layers, "r_final_layer_key": hook_mgr.num_layers,
        "map_types": ["A", "R_layer", "R_cum", "R_final"],
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": platform.python_version()},
        **(extra_meta or {}),
    }
    path = save_maps(records, meta, out_dir, save_name)
    print(f"Saved {len(records)} maps -> {path}")
    return records, meta
