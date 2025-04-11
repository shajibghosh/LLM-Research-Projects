# -*- coding: utf-8 -*-
# QLoRA: Quantized Low-Rank Adaptation for PyTorch
# Fully replaces all nn.Linear layers with QLoRALinear and quantizes them using NF4.

import torch
import torch.nn as nn
import math
from quant.nf4 import NF4Quantizer


class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32, dropout=0.05, bias=True, group_size=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.quantizer = NF4Quantizer(group_size=group_size)

        self.fp_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.fp_weight, a=math.sqrt(5))

        self.register_buffer("q_weight", None)
        self.register_buffer("q_scale", None)
        self.register_buffer("q_zero_point", None)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def quantize_once(self):
        if self.q_weight is not None or self.q_scale is not None:
            return  # already quantized
        with torch.no_grad():
            q_weight, q_scale = self.quantizer.quantize(self.fp_weight.data)
            self.q_weight = q_weight.to(self.fp_weight.device)
            self.q_scale = q_scale.to(self.fp_weight.device)

    def forward(self, x):
        if self.q_weight is None or self.q_scale is None:
            raise ValueError("Call quantize_once() before using QLoRALinear.")

        device = x.device
        dtype = x.dtype

        w = self.quantizer.dequantize(self.q_weight, self.q_scale).to(device=device, dtype=dtype)
        bias = self.bias.to(device=device, dtype=dtype) if self.bias is not None else None

        base = nn.functional.linear(x, w, bias)
        lora_out = self.dropout(x) @ self.lora_A.T.to(device, dtype) @ self.lora_B.T.to(device, dtype)
        return base + self.scaling * lora_out

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Save only quantized weight and LoRA adapters — skip fp_weight
        sd = {
            f"{prefix}q_weight": self.q_weight,
            f"{prefix}q_scale": self.q_scale,
            f"{prefix}lora_A": self.lora_A,
            f"{prefix}lora_B": self.lora_B,
        }
        if self.bias is not None:
            sd[f"{prefix}bias"] = self.bias
        return sd


def inject_qlora_adapters(model, r=8, alpha=32, dropout=0.05, group_size=64):
    """
    Replace all nn.Linear layers with QLoRALinear and quantize them.
    """
    replaced = []

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name

                qlora_layer = QLoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    bias=child.bias is not None,
                    group_size=group_size
                )

                # Copy original weights and bias
                qlora_layer.fp_weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    qlora_layer.bias.data.copy_(child.bias.data)

                # Replace and quantize
                setattr(module, child_name, qlora_layer)
                qlora_layer.quantize_once()
                qlora_layer.fp_weight = None  # free memory
                replaced.append(full_name)

    freeze_non_lora_params(model)
    print(f"[QLoRA] Injected and quantized {len(replaced)} Linear layers:")
    for name in replaced:
        print(f" - {name}")


def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False