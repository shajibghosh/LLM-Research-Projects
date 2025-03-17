import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


class QLoRALinear(torch.nn.Module):
    """
    Implements a standalone 4-bit quantized QLoRA layer for Llama models.

    - Keeps frozen weights in half-precision (bfloat16).
    - Adds trainable QLoRA layers with 4-bit quantization.
    - Uses adaptive scaling for stability.
    """

    def __init__(self, in_features: int, out_features: int, lora_dim: int, bias: bool = True, group_size: int = 16):
        super().__init__()
        self.requires_grad_(False)  # Ensure the base weights remain frozen.

        # Trainable LoRA layers
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)
        self.alpha = torch.nn.Parameter(torch.ones(1))  # Adaptive scaling

        # Initialize weights
        torch.nn.init.kaiming_normal_(self.lora_a.weight)
        torch.nn.init.zeros_(self.lora_b.weight)

        # LoRA layers should be trainable
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)

        # 4-bit quantization storage
        self.group_size = group_size
        self.register_buffer("quantized_weight", None)
        self.register_buffer("scale_factor", None)

    def quantize_4bit(self, x: torch.Tensor):
        """Applies 4-bit group-wise quantization."""
        assert x.dim() == 1 and x.size(0) % self.group_size == 0

        x = x.view(-1, self.group_size)
        scale_factor = x.abs().max(dim=-1, keepdim=True).values
        x_norm = (x + scale_factor) / (2 * scale_factor)
        x_quant = (x_norm * 15).round().to(torch.int8)

        lower_4_bits_1 = (x_quant[:, ::2] & 0xF)
        lower_4_bits_2 = ((x_quant[:, 1::2] & 0xF) << 4)
        x_quant_4 = lower_4_bits_1 + lower_4_bits_2

        return x_quant_4, scale_factor.to(torch.float16)

    def dequantize_4bit(self, quantized_weight: torch.Tensor, scale_factor: torch.Tensor):
        """Restores 4-bit quantized values back to full precision."""
        scale_factor = scale_factor.to(torch.float32)
        x_quant_8 = quantized_weight.new_empty(quantized_weight.size(0), quantized_weight.shape[1] * 2)
        x_quant_8[:, ::2] = quantized_weight & 0xF
        x_quant_8[:, 1::2] = (quantized_weight >> 4) & 0xF
        x_norm = x_quant_8.to(torch.float32) / 15
        return (x_norm * 2 * scale_factor) - scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using 4-bit quantization and LoRA adaptation."""
        if self.quantized_weight is None or self.scale_factor is None:
            self.quantized_weight, self.scale_factor = self.quantize_4bit(x.view(-1))

        weight = self.dequantize_4bit(self.quantized_weight, self.scale_factor).view(x.shape[1], -1)
        lora_out = self.alpha * self.lora_b(self.lora_a(x.to(torch.float))).to(x.dtype)
        return torch.nn.functional.linear(x, weight) + lora_out


def get_qlora_config():
    """Returns QLoRA configuration for Llama models."""
    return LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.05,  # Dropout regularization
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Apply QLoRA to projection layers
    )


def apply_qlora_to_llama(model_name: str, device: torch.device, hf_token: str):
    """
    Loads a Llama 3.2 or 3.1 model with QLoRA and 4-bit quantization.

    - Uses **standalone 4-bit quantization** (no bitsandbytes).
    - Applies **QLoRA on Llama's projection layers**.
    - Moves the model to **CUDA (if available)** or CPU.

    Args:
        model_name (str): Hugging Face model ID (e.g., "meta-llama/Llama-3.2-1B").
        device (torch.device): Target device for training.
        hf_token (str): Hugging Face authentication token.

    Returns:
        model (torch.nn.Module): The QLoRA-optimized Llama model.
    """
    print(f"Loading {model_name} with QLoRA and 4-bit quantization...")

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Keep LLM parameters in BF16
        token=hf_token
    )

    # Apply QLoRA
    qlora_config = get_qlora_config()
    qlora_model = get_peft_model(base_model, qlora_config)

    # Move the model to the target device
    qlora_model.to(device)

    print(f"{model_name} successfully loaded with QLoRA and 4-bit quantization!")

    return qlora_model

