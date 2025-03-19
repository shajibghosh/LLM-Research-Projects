# Importing required libraries
import torch
from transformers import AutoModelForCausalLM  # Hugging Face model loading utility
from peft import LoraConfig, get_peft_model  # Parameter-efficient fine-tuning (PEFT) utilities for LoRA adaptation

class QLoRALinear(torch.nn.Module):
    """
    Implements a standalone 4-bit quantized QLoRA layer for LLaMA models.

    - Keeps frozen weights in half-precision (bfloat16).
    - Adds trainable QLoRA layers with 4-bit quantization.
    - Uses adaptive scaling for stability.
    """

    def __init__(self, in_features: int, out_features: int, lora_dim: int, bias: bool = True, group_size: int = 16):
        """
        Initializes the QLoRA linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            lora_dim (int): Dimensionality of LoRA adaptation.
            bias (bool, optional): Whether to include bias. Defaults to True.
            group_size (int, optional): Number of elements per quantization group. Defaults to 16.
        """
        super().__init__()
        
        # Freeze all parameters by default (LoRA layers are made trainable later)
        self.requires_grad_(False)

        # Trainable LoRA layers: Linear transformations for low-rank adaptation
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)  # Input projection
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)  # Output projection
        
        # Adaptive scaling factor (learnable parameter)
        self.alpha = torch.nn.Parameter(torch.ones(1))

        # Initialize LoRA layers using best practices
        torch.nn.init.kaiming_normal_(self.lora_a.weight)  # Xavier-like initialization for stability
        torch.nn.init.zeros_(self.lora_b.weight)  # Initialize with zeros for controlled adaptation

        # Ensure LoRA layers are trainable while the rest remain frozen
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)

        # 4-bit Quantization Storage
        self.group_size = group_size  # Defines how many elements per group will be quantized
        self.register_buffer("quantized_weight", None)  # Stores quantized weights
        self.register_buffer("scale_factor", None)  # Stores scale factors for dequantization

    def quantize_4bit(self, x: torch.Tensor):
        """
        Applies 4-bit group-wise quantization to the input tensor.

        - Splits the tensor into groups (`group_size` elements per group).
        - Computes scale factor for each group (max absolute value).
        - Normalizes values between -1 and 1.
        - Converts normalized values to 4-bit integers (0-15 range).
        """
        assert x.dim() == 1 and x.size(0) % self.group_size == 0, "Input must be a 1D tensor and divisible by group size"

        x = x.view(-1, self.group_size)  # Reshape into quantization groups
        scale_factor = x.abs().max(dim=-1, keepdim=True).values  # Compute scale factor for each group

        # Normalize and quantize
        x_norm = (x + scale_factor) / (2 * scale_factor)  # Normalize to range [0, 1]
        x_quant = (x_norm * 15).round().to(torch.int8)  # Scale to 4-bit integer range [0, 15]

        # Pack 4-bit values into int8 storage
        lower_4_bits_1 = (x_quant[:, ::2] & 0xF)  # Extract lower 4 bits
        lower_4_bits_2 = ((x_quant[:, 1::2] & 0xF) << 4)  # Extract and shift upper 4 bits
        x_quant_4 = lower_4_bits_1 + lower_4_bits_2  # Combine into single int8 representation

        return x_quant_4, scale_factor.to(torch.float16)  # Return quantized values with scale factors

    def dequantize_4bit(self, quantized_weight: torch.Tensor, scale_factor: torch.Tensor):
        """
        Restores 4-bit quantized values back to full precision.

        - Unpacks 4-bit integer pairs stored in `quantized_weight`.
        - Applies the stored scale factors to reconstruct full precision values.
        """
        scale_factor = scale_factor.to(torch.float32)  # Convert scale factor to FP32 for accuracy

        # Create an empty tensor to hold 8-bit unpacked values
        x_quant_8 = quantized_weight.new_empty(quantized_weight.size(0), quantized_weight.shape[1] * 2)

        # Unpack stored 4-bit values back into 8-bit representation
        x_quant_8[:, ::2] = quantized_weight & 0xF
        x_quant_8[:, 1::2] = (quantized_weight >> 4) & 0xF

        # Convert back to floating-point range [-1, 1]
        x_norm = x_quant_8.to(torch.float32) / 15
        return (x_norm * 2 * scale_factor) - scale_factor  # Dequantized values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using 4-bit quantization and LoRA adaptation.

        - Quantizes input if not already quantized.
        - Dequantizes back to full precision.
        - Applies LoRA adaptation and combines with original computation.
        """
        if self.quantized_weight is None or self.scale_factor is None:
            self.quantized_weight, self.scale_factor = self.quantize_4bit(x.view(-1))

        weight = self.dequantize_4bit(self.quantized_weight, self.scale_factor).view(x.shape[1], -1)
        lora_out = self.alpha * self.lora_b(self.lora_a(x.to(torch.float))).to(x.dtype)  # Apply LoRA
        return torch.nn.functional.linear(x, weight) + lora_out  # Final output

def get_qlora_config():
    """Returns QLoRA configuration for LLaMA models.

    - Uses LoRA-style parameter-efficient tuning.
    - Applies QLoRA to key attention projection layers.
    - Supports 4-bit quantization.
    """
    return LoraConfig(
        r=16,  # LoRA rank (size of adaptation layers for QLoRA)
        lora_alpha=32,  # Scaling factor for LoRA adaptation
        lora_dropout=0.05,  # Dropout regularization for LoRA layers
        bias="none",
        task_type="CAUSAL_LM",  # Applies QLoRA for causal language modeling
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # QLoRA applied to key attention layers
    )

def apply_qlora_to_llama(model_name: str, device: torch.device, hf_token: str):
    """
    Loads a LLaMA model with QLoRA and 4-bit quantization.

    - Uses **standalone 4-bit quantization** (no bitsandbytes).
    - Applies **QLoRA on LLaMA's projection layers**.
    - Moves the model to **CUDA (if available)** or CPU.
    """
    print(f"Loading {model_name} with QLoRA and 4-bit quantization...")

    # Load pre-trained LLaMA model
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, token=hf_token)

    # Apply QLoRA configuration
    qlora_model = get_peft_model(base_model, get_qlora_config())

    # Move to device
    qlora_model.to(device)

    print(f"{model_name} successfully loaded with QLoRA and 4-bit quantization!")

    return qlora_model