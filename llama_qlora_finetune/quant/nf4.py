# -*- coding: utf-8 -*-
#
# Simulated NormalFloat4 (NF4) Quantization for PyTorch
# This module implements a simulated NF4 quantization scheme for 4-bit quantization of weights.
# It uses per-row (group-wise) scaling to quantize weights into the range [-8, 7], simulating signed 4-bit integers (int4).
# The quantization is done in a way that allows for dequantization back to the original floating-point values.
# The quantization and dequantization processes are designed to be efficient and maintain numerical stability.
#

import torch
import math


class NF4Quantizer:
    """
    Simulated NormalFloat4 (NF4) quantization module.
    This simulates 4-bit quantization by using per-row (group-wise) scale factors.
    - Each row (or group of rows) is scaled individually.
    - Weights are quantized into the range [-8, 7], simulating signed 4-bit integers (int4).
    """

    def __init__(self, group_size=64):
        # Number of rows (output features) to include per group for scaling
        self.group_size = group_size

    def quantize(self, weight: torch.Tensor):
        """
        Quantizes a floating-point weight matrix to simulated 4-bit precision.

        Args:
            weight (Tensor): A matrix of shape (out_features, in_features)

        Returns:
            q_weight (Tensor): Quantized weights (simulated int4, but stored in float32/float16)
            q_scale (Tensor): Per-row or per-group scale used to quantize the weights
        """
        device = weight.device          # Store the device (e.g., CUDA) for later use
        dtype = weight.dtype            # Store the data type (e.g., float32, float16) for later use
        out_dim = weight.size(0)        # Number of output features (rows)                                                       

        # Split the weight matrix into row chunks for group-wise quantization
        chunks = weight.chunk(math.ceil(out_dim / self.group_size), dim=0)   

        # Initialize lists to hold quantized chunks and corresponding scales                               
        q_chunks = []
        scale_chunks = []

        for chunk in chunks:
            # For each row, get the maximum absolute value (per row), shape: (chunk_rows, 1)
            max_val = chunk.abs().amax(dim=1, keepdim=True) + 1e-6  # Add epsilon to avoid division by zero
            
            # Calculate scale: max_val / 7 so that values fall into the [-8, 7] range after scaling
            scale = max_val / 7.0

            # Quantize: divide by scale → round → clamp to int4 range [-8, 7]
            # The quantization simulates int4 by rounding and clamping to the range [-8, 7]
            q = (chunk / scale).round().clamp(-8, 7)

            # Append quantized and scale tensors, cast back to original dtype for consistency
            q_chunks.append(q.to(dtype))                    
            scale_chunks.append(scale.to(dtype))

        # Reassemble full quantized matrix and scale matrix from chunks
        q_weight = torch.cat(q_chunks, dim=0).to(device).to(dtype)
        q_scale = torch.cat(scale_chunks, dim=0).to(device).to(dtype)

        # Return quantized weight and corresponding scale per row
        return q_weight, q_scale

    def dequantize(self, q_weight: torch.Tensor, scale: torch.Tensor):
        """
        Dequantizes weights using the stored scale per row.

        Args:
            q_weight (Tensor): Simulated quantized weights of shape (out_features, in_features)
            scale (Tensor): Scale per row, typically shape (out_features, 1)

        Returns:
            Tensor: Dequantized weights in original float format
        """
        # Multiply quantized values by their corresponding row scales to get back to original float values
        return q_weight * scale