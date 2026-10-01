"""
Visualisation of layer-wise maps.

show_cam_on_image and the upsample / min-max steps follow the repo's
CLIP/example.py and CLIP_explainability.ipynb (show_image_relevance). There
the function is nested inside other functions, so it cannot be imported; it
is copied verbatim here.

The background image is the model's own preprocessed input (pixel_values,
i.e. the 336x336 center crop), so the overlays align with the patch grid.
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---- verbatim from the repo (CLIP/example.py) ----
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam
# ---------------------------------------------------


def pixel_values_to_image(pixel_values):
    """[1,3,H,W] normalised tensor -> HxWx3 RGB float in [0,1] (repo's min-max)."""
    image = pixel_values[0].permute(1, 2, 0).detach().float().cpu().numpy()
    return (image - image.min()) / (image.max() - image.min())


def _upsample(grid_map, size):
    """g x g map -> size x size, bilinear (as in the repo)."""
    t = torch.as_tensor(np.asarray(grid_map, dtype=np.float32)).reshape(1, 1, *grid_map.shape)
    return torch.nn.functional.interpolate(t, size=size, mode="bilinear").reshape(size, size).numpy()


def _normalise(x, vmin=None, vmax=None):
    vmin = x.min() if vmin is None else vmin
    vmax = x.max() if vmax is None else vmax
    if vmax - vmin < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - vmin) / (vmax - vmin), 0, 1)


def overlay(image_rgb, grid_map, vmin=None, vmax=None):
    """RGB uint8 overlay. vmin/vmax=None -> per-map min-max (repo default).

    applyColorMap returns BGR, so the image is fed to show_cam_on_image in BGR
    and the result converted back to RGB.
    """
    mask = _normalise(_upsample(grid_map, image_rgb.shape[0]), vmin, vmax)
    vis = show_cam_on_image(image_rgb[..., ::-1], mask)
    return cv2.cvtColor(np.uint8(255 * vis), cv2.COLOR_BGR2RGB)


def plot_layer_grid(image_rgb, maps, title, out_path, scale="per_layer", ncols=8,
                    mode="overlay"):
    """All layers of one map type in one figure.

    maps : [L, g, g]
    scale: "per_layer" -> each panel min-max normalised (spatial pattern);
           "global"    -> one range for all layers (magnitude across depth).
    mode : "overlay" (on the image) or "raw" (patch grid, no interpolation).
    """
    L = maps.shape[0]
    nrows = int(np.ceil(L / ncols))
    gmin, gmax = (float(maps.min()), float(maps.max())) if scale == "global" else (None, None)

    fig, axs = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.3 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for l in range(len(axs)):
        ax = axs[l]
        ax.axis("off")
        if l >= L:
            continue
        if mode == "overlay":
            ax.imshow(overlay(image_rgb, maps[l], gmin, gmax))
        else:
            ax.imshow(maps[l], cmap="jet", vmin=gmin, vmax=gmax, interpolation="nearest")
        ax.set_title(f"L{l}", fontsize=9)
    fig.suptitle(f"{title}  [{scale}]", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary(image_rgb, maps, out_path, layers=None):
    """One row per map type (A, R_layer, R_cum) at selected layers, plus R_final."""
    L = maps.num_layers
    layers = layers or sorted(set(np.linspace(0, L - 1, 6).round().astype(int).tolist()))
    rows = [("A", maps.A), ("R_layer", maps.R_layer), ("R_cum", maps.R_cum)]
    ncols = len(layers) + 1
    fig, axs = plt.subplots(len(rows), ncols, figsize=(2.3 * ncols, 2.4 * len(rows)))
    for r, (name, arr) in enumerate(rows):
        axs[r, 0].axis("off")
        if r == 0:
            axs[r, 0].imshow(image_rgb); axs[r, 0].set_title("input crop", fontsize=9)
        elif r == 1:
            axs[r, 0].imshow(overlay(image_rgb, maps.R_final)); axs[r, 0].set_title("R_final", fontsize=9)
        for c, l in enumerate(layers, start=1):
            axs[r, c].imshow(overlay(image_rgb, arr[l]))
            axs[r, c].set_title(f"{name} L{l}", fontsize=9)
            axs[r, c].axis("off")
    fig.suptitle(f"'{maps.word}'  (row t-1 = {maps.query_pos}, p={maps.target_prob:.3f})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_final(image_rgb, maps, out_path):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.6))
    axs[0].imshow(image_rgb); axs[0].set_title("input crop")
    axs[1].imshow(overlay(image_rgb, maps.R_final)); axs[1].set_title(f"R_final '{maps.word}'")
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
