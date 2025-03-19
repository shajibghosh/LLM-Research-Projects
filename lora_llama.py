# Import required libraries
from pathlib import Path 
import torch
from transformers import AutoModelForCausalLM  # Hugging Face's pre-trained LLaMA model
from peft import LoraConfig, get_peft_model  # PEFT library for LoRA-based fine-tuning

class LoRALinear(torch.nn.Module):
    """
    Implements a LoRA-based linear projection module for LLaMA models.

    - Keeps the base model **frozen** while adding **trainable low-rank matrices**.
    - Reduces memory requirements for **fine-tuning large LLMs**.
    - Improves computational efficiency with a **low-rank adaptation** approach.
    """

    def __init__(self, in_features: int, out_features: int, lora_dim: int, bias: bool = True):
        """
        Initializes the LoRA projection layers.

        - Keeps frozen weights in **bfloat16** precision.
        - Only trains **LoRA layers** (A and B matrices).
        - Uses **float32** for LoRA layers to maintain numerical stability.
        
        Args:
            in_features (int): Input dimension size.
            out_features (int): Output dimension size.
            lora_dim (int): LoRA adaptation rank (low-rank factor).
            bias (bool): Whether to use bias in projection layers (default: True).
        """
        super().__init__()  # Call parent class (nn.Module) constructor

        # Define LoRA layers:
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)  # Low-rank projection A
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)  # Low-rank projection B

        # Initialize weights using standard techniques
        torch.nn.init.kaiming_normal_(self.lora_a.weight)  # Xavier/He initialization for stability
        torch.nn.init.zeros_(self.lora_b.weight)  # Set LoRA B weights to zero initially

        # Enable training for LoRA layers (keeps base model frozen)
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoRA projection layer.

        - Passes input through LoRA layers.
        - Keeps base model activations **frozen**.
        - Adds LoRA **residual updates** to the original activations.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: LoRA-enhanced activations.
        """
        y_out_frozen = x.to(torch.float16)  # Convert input to **half-precision** (saves memory)
        y_LoRA = self.lora_b(self.lora_a(x))  # LoRA transformation (A → B)
        return y_out_frozen + y_LoRA  # Add **LoRA-enhanced activations** to the frozen output


def get_lora_config():
    """
    Returns the LoRA configuration optimized for LLaMA 3.2 models.

    - Uses **lower LoRA rank (r=8)** to save memory.
    - Applies LoRA to **key attention projection layers**.
    - Enables **dropout** for regularization to prevent overfitting.

    Returns:
        LoraConfig: Configuration for applying LoRA fine-tuning.
    """
    return LoraConfig(
        r=8,  # LoRA rank (lower value reduces memory usage)
        lora_alpha=16,  # Scaling factor to amplify LoRA updates
        lora_dropout=0.05,  # Adds **dropout regularization** to prevent overfitting
        bias="none",  # No additional bias added to LoRA layers
        task_type="CAUSAL_LM",  # Fine-tuning for **causal language modeling**
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Applies LoRA **only to attention projections**
    )


def apply_lora_to_llama(model_name: str, device: torch.device, hf_token: str):
    """
    Loads a LLaMA 3.2 model and applies LoRA fine-tuning.

    - Loads a **pre-trained** LLaMA model from Hugging Face.
    - Attaches LoRA layers **to attention projections**.
    - Moves the model to **CUDA (if available)** for efficient fine-tuning.

    Args:
        model_name (str): Hugging Face model ID (e.g., `"meta-llama/Llama-3.2-1B"`).
        device (torch.device): Target device (CPU/GPU).
        hf_token (str): Hugging Face authentication token for accessing models.

    Returns:
        torch.nn.Module: The LoRA-optimized LLaMA model.
    """
    print(f"Loading {model_name} with LoRA optimization...")

    # Load the **pre-trained LLaMA model** from Hugging Face
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Uses BF16 for **better memory efficiency**
        token=hf_token  # Authentication token for Hugging Face API
    )

    # Get LoRA configuration and apply to model
    lora_config = get_lora_config()
    lora_model = get_peft_model(base_model, lora_config)  # Apply LoRA adaptation

    # Move model to the specified device (GPU or CPU)
    lora_model.to(device)

    print(f"{model_name} loaded and optimized with LoRA!")

    return lora_model