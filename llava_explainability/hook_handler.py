"""
Captures the post-softmax attention matrix A of every language-model layer
during the forward pass. The tensors stay attached to the autograd graph, so
gradients dy/dA can be taken later with torch.autograd.grad (the same pattern
the repo's CLIP_explainability notebook uses).
"""
import torch


class AttentionHookManager:
    def __init__(self, model):
        self.model = model
        self.attentions = []      # one tensor per layer: [batch, heads, seq, seq]
        self.hooks = []
        self.layers = self._find_text_layers()
        for layer in self.layers:
            self.hooks.append(layer.self_attn.register_forward_hook(self._forward_hook))

    def _find_text_layers(self):
        # Works for both module layouts:
        #   older transformers: language_model.model.layers
        #   newer transformers: model.language_model.layers
        for name, module in self.model.named_modules():
            if (name.endswith("layers") and isinstance(module, torch.nn.ModuleList)
                    and "vision" not in name):
                return module
        raise AttributeError("Could not locate the language-model decoder layers.")

    def _forward_hook(self, module, inp, output):
        attn = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if attn is None:
            raise RuntimeError(
                "Attention weights are None. Load the model with "
                "attn_implementation='eager' and call it with output_attentions=True.")
        self.attentions.append(attn)

    @property
    def num_layers(self):
        return len(self.layers)

    def clear(self):
        self.attentions = []

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
