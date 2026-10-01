"""
Layer-wise A and R maps for LLaVA from ONE forward pass.

For a target word at sequence position t, everything is read from row
q = t - 1: the position whose logit predicts the word ("where does the model
look while producing the word"). Row t itself has zero gradient under the
causal mask, so its relevance to the image is identically zero.

Per layer l (0-based), the image-patch columns of row q give:
  A[l]       raw attention, averaged over heads
  R_layer[l] gradient-gated attention A_bar_l (Eq. 5), that layer alone
  R_cum[l]   accumulated relevance after layers 0..l (Eq. 6)
  R_final  = R_cum[L-1]
"""
from dataclasses import dataclass

import numpy as np
import torch

from .relevancy_rules import avg_heads, self_attention_update


@dataclass
class LayerwiseMaps:
    word: str
    target_pos: int          # position of the word's first sub-token (t)
    query_pos: int           # row used for every map (t - 1)
    target_logit: float
    target_prob: float       # softmax prob of the word at query_pos (teacher-forced)
    A: np.ndarray            # [L, g, g]
    R_layer: np.ndarray      # [L, g, g]
    R_cum: np.ndarray        # [L, g, g]

    @property
    def R_final(self) -> np.ndarray:
        return self.R_cum[-1]

    @property
    def num_layers(self) -> int:
        return self.A.shape[0]


class LlavaLayerwiseExplainer:
    def __init__(self, model, hook_manager):
        self.model = model
        self.hook_manager = hook_manager
        self.outputs = None

    def forward(self, inputs):
        """Single forward pass; caches attentions (via hooks) and logits."""
        self.hook_manager.clear()
        self.outputs = self.model(**inputs, output_attentions=True, use_cache=False)
        n = len(self.hook_manager.attentions)
        if n != self.hook_manager.num_layers:
            raise RuntimeError(f"Captured {n} attention maps for {self.hook_manager.num_layers} layers.")
        if not self.hook_manager.attentions[0].requires_grad:
            raise RuntimeError("Attention maps do not require grad. Call "
                               "model.enable_input_require_grads() (or keep params trainable).")
        return self.outputs

    def explain_word(self, input_ids, word, target_pos, img_start, img_num_tokens, grid):
        """Backward from the target word's logit and build all layer-wise maps.
        Re-uses the cached forward pass, so call forward() once per prompt."""
        if self.outputs is None:
            raise RuntimeError("Call forward(inputs) first.")
        q = target_pos - 1
        target_id = int(input_ids[0, target_pos])
        logits_q = self.outputs.logits[0, q]
        target_logit = logits_q[target_id]
        target_prob = torch.softmax(logits_q.float(), dim=-1)[target_id].item()

        attns = self.hook_manager.attentions
        grads = torch.autograd.grad(target_logit, attns, retain_graph=True)

        img = slice(img_start, img_start + img_num_tokens)
        seq_len = attns[0].shape[-1]
        R = torch.eye(seq_len, device=attns[0].device, dtype=torch.float32)

        A_maps, R_layer_maps, R_cum_maps = [], [], []
        for attn, grad in zip(attns, grads):
            cam = attn.detach().float()           # [1, H, S, S]
            grad = grad.detach().float()
            cam_bar = avg_heads(cam, grad)        # [S, S]   (Eq. 5, repo)
            R = self_attention_update(R, cam_bar) # [S, S]   (Eq. 6, repo)

            A_maps.append(cam[0].mean(dim=0)[q, img])
            R_layer_maps.append(cam_bar[q, img])
            R_cum_maps.append(R[q, img])

        to_np = lambda xs: torch.stack(xs).reshape(len(xs), grid, grid).cpu().numpy()
        return LayerwiseMaps(
            word=word, target_pos=target_pos, query_pos=q,
            target_logit=float(target_logit.item()), target_prob=target_prob,
            A=to_np(A_maps), R_layer=to_np(R_layer_maps), R_cum=to_np(R_cum_maps),
        )
