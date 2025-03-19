"""
Instructions to run this script:

Fine-tune LLaMA with QLoRA:
    python finetune_llama.py --method qlora --dataset wikitext

Fine-tune LLaMA with LoRA:
    python finetune_llama.py --method lora --dataset llm-pie

Limit GPU usage (if multiple GPUs available):
    CUDA_VISIBLE_DEVICES=0 python finetune_llama.py --method lora --dataset llm-pie
"""

import os
import torch
import json
import time
import argparse
import fitz  # PyMuPDF
import multiprocessing
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset as HFDataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities import rank_zero_only
from qlora_llama import apply_qlora_to_llama
from lora_llama import apply_lora_to_llama

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch._inductor.compile_fx"
)


torch.cuda.empty_cache()

# Enable Float Precision Optimization
torch.set_float32_matmul_precision("high")

# Argument Parsing
parser = argparse.ArgumentParser(
    description="Fine-tune LLaMA 3.1 8B with LoRA or QLoRA"
)
parser.add_argument(
    "--method",
    choices=["lora", "qlora"],
    required=True,
    help="Fine-tuning method: LoRA or QLoRA",
)
parser.add_argument(
    "--dataset",
    choices=["wikitext", "openassistant", "llm-pie"],
    required=True,
    help="Dataset to use",
)
args = parser.parse_args()


# Hugging Face Authentication
def setup_hf_auth():
    hf_token = os.getenv("HF_TOKEN", None)
    if not hf_token:
        hf_token = input("Enter your Hugging Face token: ").strip()
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
    print_rank_0(
        f"Training on {NUM_GPUS} GPUs ({GPU_NAME}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)"
    )
    torch.cuda.empty_cache()

# Model & Training Config
MODEL_NAME = "meta-llama/Llama-3.1-8B"

BATCH_SIZE = 1  # Reduced batch size for OOM prevention
GRAD_ACCUM_STEPS = 64
LR = 5e-4
NUM_EPOCHS = 3
LOG_DIR = "logs_llama"
MODEL_SAVE_DIR = "trained_models_llama"
RESULTS_DIR = "logs_llama"

# Local Dataset Paths
DATASET_DIR = "datasets/llm-pie"
DATASET_SAVE_PATH = "datasets/llm-pie-tokenized"

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Timestamped file naming for logs
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
RESULTS_FILE = f"{RESULTS_DIR}/training_results_llama_{TIMESTAMP}.json"


# Load or Extract Dataset
def prepare_dataset(dataset_name):
    """Loads dataset from Hugging Face or extracts text from local PDFs."""

    if dataset_name == "wikitext":
        return load_dataset("wikitext", "wikitext-103-v1")["train"]

    elif dataset_name == "openassistant":
        return load_dataset("OpenAssistant/oasst1")["train"]

    elif dataset_name == "llm-pie":
        # Check if dataset is already processed and saved
        if os.path.exists(DATASET_SAVE_PATH):
            return load_from_disk(DATASET_SAVE_PATH)

        texts = []

        # Iterate through subfolders, skipping non-directory files
        for subfolder in tqdm(os.listdir(DATASET_DIR), desc="Processing Subfolders"):
            subfolder_path = os.path.join(DATASET_DIR, subfolder)

            if not os.path.isdir(subfolder_path):  # Skip files like `.DS_Store`
                continue

            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)

                if file.endswith(".pdf"):
                    try:
                        with fitz.open(file_path) as doc:
                            texts.append("".join(page.get_text() for page in doc))
                    except Exception as e:
                        print(f"Error reading {file}: {e}")

        # Convert text to Hugging Face Dataset and save it
        if texts:
            dataset = HFDataset.from_dict({"text": texts})
            dataset.save_to_disk(DATASET_SAVE_PATH)
            return dataset
        else:
            raise RuntimeError("No valid PDFs found in the dataset directory.")

    else:
        raise ValueError(
            "Invalid dataset choice! Choose from 'wikitext', 'openassistant', or 'llm-pie'."
        )


# Tokenization
def tokenize_data(dataset, tokenizer, block_size=128):
    """Tokenizes dataset and ensures padding token exists."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})

    def tokenize_function(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=block_size
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    return tokenized_dataset


# Fine-Tuning Model Class
class LLaMAFinetuner(pl.LightningModule):
    def __init__(self, model_name, train_dataset, method):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.pad_token})

        # Choose optimization method
        if method == "qlora":
            self.model = apply_qlora_to_llama(
                model_name, torch.device("cuda"), HF_TOKEN
            )
        else:
            self.model = apply_lora_to_llama(model_name, torch.device("cuda"), HF_TOKEN)

        self.model.gradient_checkpointing_enable()
        self.training_losses = []
        self.train_dataset = train_dataset
        self.writer = (
            SummaryWriter(LOG_DIR) if RANK == 0 else None
        )  # Only rank 0 writes logs
        self.method = method

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
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
            num_workers=NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )


# Training Execution
if __name__ == "__main__":
    dataset = prepare_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    print_rank_0(
        f"\nTraining LLaMA 3.1 8B on {args.dataset} dataset using {args.method.upper()}..."
    )

    model = LLaMAFinetuner(MODEL_NAME, tokenized_dataset, args.method)

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accumulate_grad_batches=GRAD_ACCUM_STEPS,
        precision="bf16-mixed",
        strategy="ddp" if NUM_GPUS > 1 else "auto",
        devices=NUM_GPUS,
    )

    start_time = time.time()
    trainer.fit(model)
    training_time = time.time() - start_time

    # Ensure model results dictionary exists
    results = {
        "dataset": args.dataset,
        "method": args.method,
        "final_loss": model.training_losses[-1] if model.training_losses else None,
        "training_time": training_time,
        "total_tokens": sum(len(tokens) for tokens in tokenized_dataset["input_ids"]),
        "gpu_memory_usage": (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else "N/A"
        ),
        "loss_curve": model.training_losses,  # Adding loss curve for evaluation
    }

    # Save fine-tuned model with timestamp
    model_save_path = f"{MODEL_SAVE_DIR}/{args.method}_llama_{args.dataset}_{TIMESTAMP}"
    model.model.save_pretrained(model_save_path)
    print_rank_0(f"Model saved at {model_save_path}")

    # Save results to JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print_rank_0(f"\nTraining results saved to {RESULTS_FILE}")
    print_rank_0("\nTo visualize training logs, run: tensorboard --logdir logs_llama")
