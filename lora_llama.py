from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class LoRALinear(torch.nn.Module):
    """LoRA-based linear projection module for Llama 3.2 models."""
    
    def __init__(self, in_features: int, out_features: int, lora_dim: int, bias: bool = True):
        """
        Implements LoRA: Low-Rank Adaptation of Llama Projection Layers.
        
        - Keeps frozen weights in half-precision
        - Only trains LoRA layers (A and B projections)
        - Uses float32 for LoRA layers to maintain stability
        """
        super().__init__()
        
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)
        
        torch.nn.init.kaiming_normal_(self.lora_a.weight)  # Xavier initialization
        torch.nn.init.zeros_(self.lora_b.weight)  # Zero initialize LoRA B layer
        
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        - Keeps base Llama weights frozen (FP16)
        - Adds LoRA-modified residual connections
        """
        y_out_frozen = x.to(torch.float16)  # Cast input to half-precision
        y_LoRA = self.lora_b(self.lora_a(x))  # LoRA forward pass
        return y_out_frozen + y_LoRA  # Return LoRA-enhanced activations


def get_lora_config():
    """Returns LoRA configuration tuned for Llama 3.2 models."""
    return LoraConfig(
        r=8,  # Low-rank factor (smaller saves memory)
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.05,  # Dropout regularization
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Apply LoRA to Llama projection layers
    )


def apply_lora_to_llama(model_name: str, device: torch.device, hf_token: str):
    """
    Loads a Llama 3.2 model and applies LoRA-based optimization.

    - Uses LoRA adaptation for select projection layers
    - Keeps most weights frozen to save memory
    - Moves model to the selected device (CUDA or CPU)
    """
    print(f"Loading {model_name} with LoRA optimization...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # BF16 for better memory efficiency
        token=hf_token
    )
    
    lora_config = get_lora_config()
    lora_model = get_peft_model(base_model, lora_config)  # Apply LoRA
    
    lora_model.to(device)  # Move to selected device
    print(f"{model_name} loaded and optimized with LoRA!")

    return lora_model
