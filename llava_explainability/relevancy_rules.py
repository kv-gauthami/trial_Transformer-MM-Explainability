"""
Relevancy rules for LLaVA, reused from the original repo.

Source: lxmert/lxmert/src/ExplanationGenerator.py 

  avg_heads                  -> Eq. 5 : A_bar = E_h[(grad ⊙ A)^+]
  apply_self_attention_rules -> Eq. 6/7 : returns the additions A_bar @ R
  handle_residual            -> Eq. 8/9 : normalisation (not used by default)
  compute_rollout_attention  -> Abnar & Zuidema rollout baseline (optional)

LLaVA-1.5 is decoder-only: image patches and text share one self-attention
stream. so only the R_ss update 
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lxmert.lxmert.src.ExplanationGenerator import (  
    avg_heads,
    apply_self_attention_rules as _repo_apply_self_attention_rules,
    handle_residual,
    compute_rollout_attention,
)

__all__ = ["avg_heads", "self_attention_update", "handle_residual", "compute_rollout_attention"]


def self_attention_update(R_ss, cam_bar):
    """Eq. 6: R_ss <- R_ss + A_bar @ R_ss.
    """
    # R_ss[:, :1] : passing a dummy argument to match the repo's interface.
    R_ss_addition, _ = _repo_apply_self_attention_rules(R_ss, R_ss[:, :1], cam_bar)
    return R_ss + R_ss_addition
