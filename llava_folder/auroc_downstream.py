
# AUROC; Input: a heatmap pkl (attention_heatmaps.pkl from specific_word_attn_v2.py,
# or relevancy_maps.pkl from run_layerwise_maps.py), runs
# YOLO-World to get one box per target word, maps each box into LLaVA's 336
# center-crop / 24x24 grid, and reports AUROC(heatmap, mask) per
# (prompt, word, layer).

# Run order:
#     1) python run_layerwise_maps.py         # writes relevancy_maps.pkl
#     2) python auroc_downstream.py --pkl man_and_dog/relevancy_maps.pkl
#
# Keys: (prompt, word, layer)            -> legacy A-map pkl, map_type = "A"
#       (prompt, word, map_type, layer)  -> type-aware pkl (A, R_layer, R_cum, R_final)


import os
import csv
import argparse
import pickle

import numpy as np
from PIL import Image
from ultralytics import YOLOWorld
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dynamically get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------- CONFIG --------------------------------
IMAGE_PATH     = os.path.join(BASE_DIR, "man_dog_1.png")
HEATMAP_PKL    = os.path.join(BASE_DIR, "attention_heatmaps.pkl")
TARGET_WORDS   = ["dog", "man"]
OUT_DIR        = os.path.join(BASE_DIR, "man_and_dog")
YOLO_WORLD     = "yolov8s-world.pt"     # or "yolov8x-worldv2.pt"
YOLO_CONF      = 0.15
DEVICE         = "cuda"
CROP           = 336                    # LLaVA-1.5 CLIP center-crop size (fixed by the model)
SAVE_BOX_PLOTS = True


# ---------------------------- YOLO-WORLD BOXES --------------------------
def get_yoloworld_bboxes(world_model, image_path, target_words, conf=0.15, device="cuda"):
    """{word: (x_min, y_min, x_max, y_max)} in ORIGINAL pixels, or None if missing."""
    world_model.set_classes(target_words)
    results = world_model.predict(image_path, imgsz=640, conf=conf, device=device, verbose=False)

    best = {w: (None, -1.0) for w in target_words}
    for r in results:
        for box, cls, c in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            word = target_words[int(cls)]
            c = float(c)
            if c > best[word][1]:
                best[word] = (tuple(box.cpu().numpy()), c)

    out = {w: b for w, (b, _) in best.items()}
    for w, b in out.items():
        if b is None:
            print(f"  [YOLO-World] '{w}' not detected (try lower conf or a richer phrase).")
    return out


def plot_bbox(image_path, bbox, word, out_path):
    if bbox is None:
        return
    x_min, y_min, x_max, y_max = bbox
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=3, edgecolor="lime", facecolor="none"))
    ax.text(x_min, y_min - 8, word, color="lime", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {out_path}")


# ---------------------- BBOX -> GRID MASK (336 space) -------------------
def clip_bbox_to_grid(bbox, orig_size, grid_size, target=336):
    """bbox (original px) -> inclusive grid cells in the `target` center-crop space,
       or None if the object fell outside the crop."""
    W, H = orig_size
    x_min, y_min, x_max, y_max = bbox

    scale = target / min(W, H)                 # resize shortest edge -> target
    x_off = (W * scale - target) / 2.0         # center-crop offsets
    y_off = (H * scale - target) / 2.0

    bx_min, bx_max = x_min * scale - x_off, x_max * scale - x_off
    by_min, by_max = y_min * scale - y_off, y_max * scale - y_off

    bx_min, bx_max = max(0, bx_min), min(target, bx_max)
    by_min, by_max = max(0, by_min), min(target, by_max)
    if bx_max <= bx_min or by_max <= by_min:
        return None

    cell = target / grid_size
    return (int(bx_min // cell), int(by_min // cell),
            int((bx_max - 1e-6) // cell), int((by_max - 1e-6) // cell))


def bbox_to_binary_mask(bbox, orig_size, grid_size, target=336):
    mask = np.zeros((grid_size, grid_size), dtype=int)      # [row(y), col(x)]
    g = clip_bbox_to_grid(bbox, orig_size, grid_size, target)
    if g is not None:
        c0, r0, c1, r1 = g
        mask[r0:r1 + 1, c0:c1 + 1] = 1
    return mask


def spatial_auroc(heatmap_2d, binary_mask):
    """P(random in-object patch scores higher than random background patch).
       Heatmap renormalization is irrelevant — AUROC only sees the ranking."""
    y_true  = binary_mask.flatten()
    y_score = heatmap_2d.flatten()
    if len(np.unique(y_true)) < 2:       # mask all-0 or all-1 -> undefined
        return None
    return roc_auc_score(y_true, y_score)


# -------------------------------- OUTPUT --------------------------------
def print_table(rows):
    if not rows:
        print("\nNo AUROC rows produced.")
        return
    print("\n" + "=" * 56)
    print(f"{'prompt':>6} | {'word':>5} | {'type':>7} | {'layer':>5} | {'AUROC':>6}")
    print("-" * 56)
    for r in sorted(rows, key=lambda x: (x["prompt"], x["word"], x["map_type"], x["layer"])):
        a = "n/a" if r["auroc"] is None else f"{r['auroc']:.3f}"
        print(f"{r['prompt']:>6} | {r['word']:>5} | {r['map_type']:>7} | {r['layer']:>5} | {a:>6}")
    print("=" * 46)


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "word", "map_type", "layer", "auroc"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved AUROC table -> {path}")


# --------------------------------- MAIN ---------------------------------
def main():
    global HEATMAP_PKL, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=HEATMAP_PKL)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    HEATMAP_PKL, OUT_DIR = args.pkl, args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load the heatmaps the attention script wrote
    if not os.path.exists(HEATMAP_PKL):
        raise FileNotFoundError(
            f"{HEATMAP_PKL} not found. Run specific_word_attn_v2.py first to generate it.")
    with open(HEATMAP_PKL, "rb") as f:
        heatmaps = pickle.load(f)          # {(prompt_idx, word, layer): 24x24 array}
    print(f"Loaded {len(heatmaps)} heatmaps from {HEATMAP_PKL}")

    # grid size is inferred from the arrays themselves -> no hidden coupling to the
    # upstream script's grid constant. Only CROP (336 px) stays a config value.
    grid_size = next(iter(heatmaps.values())).shape[0]
    print(f"Inferred grid size: {grid_size}x{grid_size}")

    image = Image.open(IMAGE_PATH).convert("RGB")
    orig_size = image.size                                  # (W, H)

    # 2. Boxes -> masks
    world_model = YOLOWorld(YOLO_WORLD)
    boxes = get_yoloworld_bboxes(world_model, IMAGE_PATH, TARGET_WORDS, conf=YOLO_CONF, device=DEVICE)
    print("Boxes:", boxes)

    # fail fast: every target word must have a detected box
    missing = [w for w, b in boxes.items() if b is None]
    if missing:
        raise ValueError(
            f"YOLO-World found no box for {missing}. Lower YOLO_CONF (now {YOLO_CONF}), "
            f"try a richer phrase, or switch to yolov8x-worldv2.pt.")

    if SAVE_BOX_PLOTS:
        for word, box in boxes.items():
            plot_bbox(IMAGE_PATH, box, word, out_path=os.path.join(OUT_DIR, f"{word}_box.png"))

    masks = {}
    for w, b in boxes.items():
        g = clip_bbox_to_grid(b, orig_size, grid_size, target=CROP)
        if g is None:
            raise ValueError(
                f"'{w}' box {b} fell entirely outside the {CROP} center-crop — mask "
                f"would be empty. Check the image aspect ratio and the crop assumption.")
        m = bbox_to_binary_mask(b, orig_size, grid_size, target=CROP)
        assert m.sum() > 0, f"'{w}' produced an all-zero mask despite in-crop cells {g}."
        masks[w] = m

    # 3. AUROC per (prompt, word, layer) — fail fast on any mismatch
    rows = []
    for key, hmap in sorted(heatmaps.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if len(key) == 3:
            idx, word, layer = key
            map_type = "A"
        else:
            idx, word, map_type, layer = key
        if word not in TARGET_WORDS:
            continue                 
        if word not in masks:
            raise KeyError(
                f"heatmap word '{word}' (prompt {idx}, layer {layer}) has no mask. "
                f"masks available for {list(masks)}; TARGET_WORDS = {TARGET_WORDS}. "
                f"Make sure the words scored upstream match the ones detected here.")
        if hmap.shape != masks[word].shape:
            raise ValueError(
                f"shape mismatch for '{word}' (prompt {idx}, layer {layer}): "
                f"heatmap {hmap.shape} vs mask {masks[word].shape}.")
        if np.asarray(hmap).std() == 0:
            raise ValueError(f"constant heatmap for '{word}' ({map_type}, layer {layer}) -> AUROC meaningless.")
        a = spatial_auroc(hmap, masks[word])
        if a is None:
            raise ValueError(
                f"AUROC undefined for '{word}' (prompt {idx}, layer {layer}): mask has "
                f"{masks[word].sum()} positive cells of {masks[word].size} — all-0 or all-1.")
        rows.append({"prompt": idx, "word": word, "map_type": map_type, "layer": layer,
                     "auroc": round(a, 4)})

    # 4. Report
    print_table(rows)
    save_csv(rows, os.path.join(OUT_DIR, "auroc_results.csv"))


if __name__ == "__main__":
    main()