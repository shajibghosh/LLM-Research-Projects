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
from qlora_llama import apply_qlora_to_llama  # Import QLoRA function

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.compile_fx")

# Hugging Face Authentication
def setup_hf_auth():
    hf_token = os.getenv("HF_TOKEN", None)
    if not hf_token:
        hf_token = getpass("Enter your Hugging Face token: ")
        os.environ["HF_TOKEN"] = hf_token
    return hf_token

HF_TOKEN = setup_hf_auth()

# Detect GPU Count & Setup Distributed Training
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
RANK = int(os.getenv("LOCAL_RANK", 0))  # Get Local Rank in Multi-GPU DDP Training

@rank_zero_only
def print_rank_0(*args, **kwargs):
    """Ensures only rank 0 prints logs to avoid duplication."""
    print(*args, **kwargs)

if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    print_rank_0(f"Training on {NUM_GPUS} GPUs ({GPU_NAME}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)")
    torch.cuda.empty_cache()

# Model & Training Config
MODELS = {
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B"
}

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 32
LR = 5e-4
NUM_EPOCHS = 5
LOG_DIR = "logs_qlora_general"
MODEL_SAVE_DIR = "trained_models_qlora_general"
RESULTS_DIR = "logs_qlora_general"

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Timestamped file naming for logs
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
RESULTS_FILE = f"{RESULTS_DIR}/training_results_qlora_{TIMESTAMP}.json"

# Supported Benchmark Datasets
BENCHMARK_DATASETS = {
    "wikitext": "wikitext",
    "oasst1": "OpenAssistant/oasst1"
}

def load_benchmark_dataset(dataset_name):
    """Loads benchmark dataset from Hugging Face"""
    if dataset_name not in BENCHMARK_DATASETS:
        raise ValueError(f"Invalid dataset name. Choose from {list(BENCHMARK_DATASETS.keys())}")

    print_rank_0(f"Loading benchmark dataset: {dataset_name}...")

    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1")["train"]
    elif dataset_name == "oasst1":
        dataset = load_dataset("OpenAssistant/oasst1")["train"]

    return dataset

# Tokenization
def tokenize_data(dataset, tokenizer, block_size=128):
    """Tokenizes dataset and ensures padding token exists."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=block_size)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

# Fine-Tuning Model Class
class QLoRAFinetuner(pl.LightningModule):
    def __init__(self, model_name, train_dataset):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.pad_token})

        self.model = apply_qlora_to_llama(model_name, torch.device("cuda"), HF_TOKEN)
        self.model.gradient_checkpointing_enable()
        self.training_losses = []
        self.train_dataset = train_dataset
        self.writer = SummaryWriter(LOG_DIR) if RANK == 0 else None  # Only rank 0 writes logs

    def training_step(self, batch, batch_idx):
        """Training step for fine-tuning."""
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = outputs.loss
        self.training_losses.append(loss.item())

        if self.writer:
            self.writer.add_scalar("Train Loss", loss.item(), self.global_step)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)

    def train_dataloader(self):
        """Dynamically set num_workers for optimal performance."""
        NUM_WORKERS = min(8, multiprocessing.cpu_count() // 2)

        return DataLoader(
            self.train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            collate_fn=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            num_workers=NUM_WORKERS,  # Optimized for performance
            pin_memory=True if torch.cuda.is_available() else False
        )

# Training Execution
results = {}

if __name__ == "__main__":
    for dataset_name in BENCHMARK_DATASETS.keys():
        dataset = load_benchmark_dataset(dataset_name)
        tokenizer = AutoTokenizer.from_pretrained(MODELS["Llama-3.2-1B"], token=HF_TOKEN)
        tokenized_dataset = tokenize_data(dataset, tokenizer)

        for model_name, model_path in MODELS.items():
            print_rank_0(f"\nTraining {model_name} on {dataset_name} dataset with QLoRA...")

            model = QLoRAFinetuner(model_path, tokenized_dataset)

            trainer = pl.Trainer(
                max_epochs=NUM_EPOCHS,
                accumulate_grad_batches=GRAD_ACCUM_STEPS,
                precision="bf16-mixed",
                strategy="ddp" if NUM_GPUS > 1 else "auto",  # Multi-GPU enabled
                devices=NUM_GPUS
            )

            start_time = time.time()
            trainer.fit(model)
            training_time = time.time() - start_time  # Compute training duration

            # Ensure model results dictionary exists
            key = f"{model_name}_{dataset_name}"
            if key not in results:
                results[key] = {}

            # Store results only on rank 0
            if RANK == 0:
                results[key].update({
                    "dataset": dataset_name,
                    "final_loss": model.training_losses[-1] if model.training_losses else None,
                    "training_time": training_time,
                    "total_tokens": sum(len(tokens) for tokens in tokenized_dataset["input_ids"]),
                    "gpu_memory_usage": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else "N/A",
                    "loss_curve": model.training_losses
                })

                # Save fine-tuned model with timestamp
                model_save_path = f"{MODEL_SAVE_DIR}/{model_name}_qlora_{dataset_name}_{TIMESTAMP}"
                model.model.save_pretrained(model_save_path)
                print_rank_0(f"Model saved at {model_save_path}")

    # Save results to JSON only on rank 0
    if RANK == 0:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=4)

        print_rank_0(f"\nTraining results saved to {RESULTS_FILE}")
        print_rank_0("\nTo visualize training logs, run: tensorboard --logdir logs_qlora_general")