# -*- coding: utf-8 -*-
#           
# # LLaMA 3.1 Model Loader for PyTorch
# # This code is designed to load the LLaMA 3.1 model and tokenizer from Hugging Face's model hub.  
# # It includes safe defaults for model loading, including GPU-efficient configurations and padding settings.
# # The model is set up for training with gradient checkpointing and proper handling of padding tokens.     

import torch                                                            
from transformers import AutoModelForCausalLM, AutoTokenizer                
from typing import Optional         
import os


def load_llama_model(
    model_name: str = "meta-llama/Llama-3.1-8B",                    
    torch_dtype: torch.dtype = torch.float16,
    device_map: Optional[str] = "auto"
):
    """
    Load the base LLaMA 3.1 8B model and tokenizer with safe defaults and GPU-efficient config.
    """
    print(f"[INFO] Loading model: {model_name}")

    hf_token = os.getenv("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,                                     # Load tokenizer from Hugging Face model hub
        use_fast=True,                                  # Use fast tokenizer for better performance
        padding_side="right",                           # Fix: ensure padding on the right side
        truncation_side="right",                        # Fix: ensure truncation on the right side
        use_auth_token=hf_token                         # handles gated models
    )

    # Ensure pad_token exists for training
    if tokenizer.pad_token is None:                                 
        tokenizer.pad_token = tokenizer.eos_token               # Set pad_token to eos_token if not defined 

    model = AutoModelForCausalLM.from_pretrained(                       
        model_name,                                             # Load model from Hugging Face model hub    
        torch_dtype=torch_dtype,                                # Set torch dtype for GPU efficiency
        device_map=device_map,                                  # Automatically map model to available devices
        use_auth_token=hf_token                                 # required for access-restricted models
    )

    # Apply tokenizer settings to model config                      
    model.config.pad_token_id = tokenizer.pad_token_id              # Set pad_token_id in model config
    model.config.use_cache = False                                  # important for gradient checkpointing during training

    print(f"[INFO] Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}") # Print total number of parameters
    return model, tokenizer                                                                                             
