"""
Type-aware storage.

<name>.pkl        : {(prompt_idx, word, map_type, layer): np.ndarray[g, g]}
<name>_meta.json  : run metadata (model, prompt, positions, versions, ...)

map_type in {"A", "R_layer", "R_cum", "R_final"}.
layer is the 0-based decoder layer. R_final is stored with layer = num_layers,
meaning "aggregated over all layers"; it equals R_cum at layer num_layers - 1.
Keys stay flat tuples of (int, str, str, int), so they sort and filter easily.
"""
import json
import os
import pickle

MAP_TYPES = ("A", "R_layer", "R_cum", "R_final")


def maps_to_records(prompt_idx, maps):
    """LayerwiseMaps -> {(prompt_idx, word, map_type, layer): array}."""
    rec = {}
    for l in range(maps.num_layers):
        rec[(prompt_idx, maps.word, "A", l)] = maps.A[l]
        rec[(prompt_idx, maps.word, "R_layer", l)] = maps.R_layer[l]
        rec[(prompt_idx, maps.word, "R_cum", l)] = maps.R_cum[l]
    rec[(prompt_idx, maps.word, "R_final", maps.num_layers)] = maps.R_final
    return rec


def save_maps(records, meta, out_dir, name="relevancy_maps"):
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, f"{name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    with open(os.path.join(out_dir, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return pkl_path


def load_maps(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def select(records, prompt_idx=None, word=None, map_type=None):
    """Filter records; returns {layer: array} sorted by layer when a single
    (prompt, word, map_type) is selected, else the filtered dict."""
    out = {k: v for k, v in records.items()
           if (prompt_idx is None or k[0] == prompt_idx)
           and (word is None or k[1] == word)
           and (map_type is None or k[2] == map_type)}
    if prompt_idx is not None and word is not None and map_type is not None:
        return {k[3]: v for k, v in sorted(out.items(), key=lambda kv: kv[0][3])}
    return out
