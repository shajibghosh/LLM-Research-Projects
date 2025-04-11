# -*- coding: utf-8 -*-
# LoRA (Low-Rank Adaptation) with complete parameter-saving optimization
# Injects LoRA into all Linear layers and saves only LoRA weights.

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32, dropout=0.05, bias=True):
        """
        A LoRA-augmented linear layer.
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Simulate frozen pretrained weights (but not saved)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01, requires_grad=False)

        # Learnable LoRA adapters
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Optional trainable bias
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        weight = self.weight.to(device=device, dtype=dtype)
        bias = self.bias.to(device=device, dtype=dtype) if self.bias is not None else None
        lora_A = self.lora_A.to(device=device, dtype=dtype)
        lora_B = self.lora_B.to(device=device, dtype=dtype)

        base_output = nn.functional.linear(x, weight, bias)
        lora_output = self.dropout(x) @ lora_A.T @ lora_B.T
        return base_output + self.scaling * lora_output

    def state_dict(self, *args, **kwargs):
        """
        Override to save only LoRA weights (not full base weights).
        """
        state = {
            "lora_A": self.lora_A,
            "lora_B": self.lora_B
        }
        if self.bias is not None:
            state["bias"] = self.bias
        return state


def inject_lora_adapters(model: nn.Module, r=8, alpha=32, dropout=0.05):
    """
    Replace all nn.Linear layers in the model with LoRALinear.
    """
    replaced = []

    for name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                lora_linear = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    bias=child.bias is not None
                )
                lora_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora_linear.bias.data.copy_(child.bias.data)

                setattr(module, attr_name, lora_linear)
                replaced.append(f"{name}.{attr_name}" if name else attr_name)

    freeze_non_lora_params(model)
    print(f"[LoRA] Injected into {len(replaced)} linear layers.")
    for name in replaced:
        print(" -", name)


def freeze_non_lora_params(model: nn.Module):
    """
    Freeze all parameters except LoRA adapter weights.
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False