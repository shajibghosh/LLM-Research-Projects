# Import required libraries
import os
import torch
import pytorch_lightning as pl
import json
import time
import pandas as pd
import multiprocessing
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from getpass import getpass
from pytorch_lightning.utilities import rank_zero_only  # Ensures only rank 0 prints logs
from lora_llama import apply_lora_to_llama  # Import LoRA function

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.compile_fx")

# Hugging Face Authentication
def setup_hf_auth():
    """Set up authentication for Hugging Face API."""
    hf_token = os.getenv("HF_TOKEN", None)  # Try to get the token from the environment variable
    if not hf_token:
        hf_token = getpass("Enter your Hugging Face token: ")  # Prompt user if not found
        os.environ["HF_TOKEN"] = hf_token  # Store it in environment for later use
    return hf_token

HF_TOKEN = setup_hf_auth()  # Authenticate with Hugging Face

# Detect GPU Count & Setup Distributed Training
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1  # Get the number of available GPUs
RANK = int(os.getenv("LOCAL_RANK", 0))  # Get the local rank in multi-GPU training

@rank_zero_only
def print_rank_0(*args, **kwargs):
    """Ensures only rank 0 prints logs to avoid duplication in multi-GPU settings."""
    print(*args, **kwargs)

if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)  # Get the name of the first available GPU
    print_rank_0(f"Training on {NUM_GPUS} GPUs ({GPU_NAME}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)")
    torch.cuda.empty_cache()  # Clear cached memory to free up GPU space

# Model & Training Configurations
MODELS = {
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",  # Small model
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B"   # Larger model
}

BATCH_SIZE = 16  # Number of samples per batch
GRAD_ACCUM_STEPS = 32  # Gradient accumulation steps to effectively increase batch size
LR = 5e-4  # Learning rate for training
NUM_EPOCHS = 5  # Number of epochs for training
LOG_DIR = "logs_lora_general"  # Directory for training logs
MODEL_SAVE_DIR = "trained_models_lora_general"  # Directory to save fine-tuned models
RESULTS_DIR = "logs_lora_general"  # Directory to store training results

# Ensure necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Generate timestamp for saving logs
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
RESULTS_FILE = f"{RESULTS_DIR}/training_results_lora_{TIMESTAMP}.json"

# Supported Benchmark Datasets
BENCHMARK_DATASETS = {
    "wikitext": "wikitext",
    "oasst1": "OpenAssistant/oasst1"
}

def load_benchmark_dataset(dataset_name):
    """Loads benchmark dataset from Hugging Face."""
    if dataset_name not in BENCHMARK_DATASETS:
        raise ValueError(f"Invalid dataset name. Choose from {list(BENCHMARK_DATASETS.keys())}")

    print_rank_0(f"Loading benchmark dataset: {dataset_name}...")

    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1")["train"]
    elif dataset_name == "oasst1":
        dataset = load_dataset("OpenAssistant/oasst1")["train"]

    return dataset

# Tokenization Function
def tokenize_data(dataset, tokenizer, block_size=128):
    """Tokenizes dataset and ensures padding token exists."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token if pad token is missing
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=block_size)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

# Fine-Tuning Model Class
class LoRAFinetuner(pl.LightningModule):
    def __init__(self, model_name, train_dataset):
        """Initialize the fine-tuning model with LoRA applied."""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.pad_token})

        # Apply LoRA to the model and move it to GPU
        self.model = apply_lora_to_llama(model_name, torch.device("cuda"), HF_TOKEN)
        self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency
        self.training_losses = []  # Store training loss values
        self.train_dataset = train_dataset
        self.writer = SummaryWriter(LOG_DIR) if RANK == 0 else None  # Log metrics only for rank 0

    def training_step(self, batch, batch_idx):
        """Defines a single training step."""
        batch = {k: v.to("cuda") for k, v in batch.items()}  # Move batch to GPU
        outputs = self.model(**batch)  # Forward pass
        loss = outputs.loss  # Compute loss
        self.training_losses.append(loss.item())  # Store loss for later analysis

        if self.writer:
            self.writer.add_scalar("Train Loss", loss.item(), self.global_step)  # Log loss to TensorBoard

        self.log("train_loss", loss, prog_bar=True)  # Log loss for monitoring
        return loss

    def configure_optimizers(self):
        """Configures the optimizer for training."""
        return torch.optim.AdamW(self.parameters(), lr=LR)

    def train_dataloader(self):
        """Creates DataLoader with optimal number of workers."""
        NUM_WORKERS = min(8, multiprocessing.cpu_count() // 2)  # Auto-adjust worker count

        return DataLoader(
            self.train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            collate_fn=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            num_workers=NUM_WORKERS,  # Parallel data loading
            pin_memory=True if torch.cuda.is_available() else False  # Pin memory if GPU is used
        )

# Training Execution
results = {}

if __name__ == "__main__":
    for dataset_name in BENCHMARK_DATASETS.keys():
        dataset = load_benchmark_dataset(dataset_name)  # Load dataset
        tokenizer = AutoTokenizer.from_pretrained(MODELS["Llama-3.2-1B"], token=HF_TOKEN)  # Load tokenizer
        tokenized_dataset = tokenize_data(dataset, tokenizer)  # Tokenize dataset

        for model_name, model_path in MODELS.items():
            print_rank_0(f"\nTraining {model_name} on {dataset_name} dataset with LoRA...")

            model = LoRAFinetuner(model_path, tokenized_dataset)  # Initialize model

            trainer = pl.Trainer(
                max_epochs=NUM_EPOCHS,
                accumulate_grad_batches=GRAD_ACCUM_STEPS,
                precision="bf16-mixed",  # Use bf16 mixed precision for better performance
                strategy="ddp" if NUM_GPUS > 1 else "auto",  # Enable multi-GPU training
                devices=NUM_GPUS
            )

            start_time = time.time()
            trainer.fit(model)  # Train model
            training_time = time.time() - start_time  # Compute training time

            if RANK == 0:
                results[f"{model_name}_{dataset_name}"] = {
                    "dataset": dataset_name,
                    "final_loss": model.training_losses[-1] if model.training_losses else None,
                    "training_time": training_time
                }

                model_save_path = f"{MODEL_SAVE_DIR}/{model_name}_lora_{dataset_name}_{TIMESTAMP}"
                model.model.save_pretrained(model_save_path)  # Save model
                print_rank_0(f"Model saved at {model_save_path}")

    if RANK == 0:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=4)  # Save results

        print_rank_0(f"\nTraining results saved to {RESULTS_FILE}")