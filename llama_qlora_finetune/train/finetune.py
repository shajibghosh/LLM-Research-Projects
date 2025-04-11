# -*- coding: utf-8 -*-
"""
Fine-tunes LLaMA 3.1 8B using LoRA or QLoRA on PDF data.
Includes:
- TensorBoard + W&B logging
- Auto-resume from latest checkpoint
- Best/final/epochal checkpointing
- Isolated run directories
- TFLOP and performance summary
"""

import os, time, math, argparse, warnings, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from dotenv import load_dotenv
load_dotenv()

from models.llama import load_llama_model
from models.lora import inject_lora_adapters
from models.qlora import inject_qlora_adapters
from data.pdf_loader import load_pdf_dataset_from_dir
from optim.paged_adamw import PagedAdamW
from utils.logging import Logger

# Environment flags
USE_WANDB = os.getenv("USE_WANDB", "false").lower() == "true"
USE_HF_UPLOAD = os.getenv("USE_HF_UPLOAD", "false").lower() == "true"
SAVE_EVERY_EPOCH = int(os.getenv("SAVE_EVERY_EPOCH", 1))
warnings.filterwarnings("ignore")


class TokenizedDataset(Dataset):
    """Wraps list of tokenized samples into PyTorch Dataset"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3.1 8B with LoRA/QLoRA")
    parser.add_argument("--adapter", choices=["lora", "qlora"], required=True)
    parser.add_argument("--pdf_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_path", type=str, default="finetuned_model")
    return parser.parse_args()


def get_latest_checkpoint_dir(base_path: str):
    """Scans directory for latest checkpoint directory."""
    checkpoints = [d for d in os.listdir(base_path) if d.startswith("checkpoint_epoch_")]
    if not checkpoints:
        return None
    latest = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1]
    return os.path.join(base_path, latest)


def train():
    args = parse_args()
    torch.cuda.reset_peak_memory_stats()

    # Create unique run folder using timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.adapter}_run_{timestamp}"
    args.save_path = os.path.join(args.save_path, run_id)
    os.makedirs(args.save_path, exist_ok=True)

    model, tokenizer = load_llama_model(args.model_name, torch_dtype=torch.float16)
    print(f"[INFO] Model device map:\n{model.hf_device_map}")

    # Inject adapter of choice
    if args.adapter == "lora":
        inject_lora_adapters(model)
    else:
        inject_qlora_adapters(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_samples = load_pdf_dataset_from_dir(args.pdf_dir, tokenizer=tokenizer, max_tokens=args.max_tokens)
    print(f"[INFO] Loaded {len(tokenized_samples)} tokenized samples from {args.pdf_dir}")
    dataset = TokenizedDataset(tokenized_samples)

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["input_ids"] for x in batch])
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = PagedAdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model.train()
    logger = Logger(project="llama-finetune", run_name=run_id)
    total_tokens = len(tokenized_samples) * args.max_tokens
    start_time = time.time()

    # === Auto-resume from latest ===
    start_epoch = 0
    latest_ckpt = get_latest_checkpoint_dir(args.save_path)
    if latest_ckpt:
        print(f"[Resume] Loading from checkpoint: {latest_ckpt}")
        model.load_state_dict(torch.load(os.path.join(latest_ckpt, "model.pt")))
        optimizer.load_state_dict(torch.load(os.path.join(latest_ckpt, "optimizer.pt")))
        meta = torch.load(os.path.join(latest_ckpt, "meta.pt"))
        start_epoch = meta.get("epoch", 0)

    # === Training ===
    best_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(model.device)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            # Per-step logging
            logger.log(step=epoch * len(dataloader) + step, loss=loss.item(), lr=args.lr)
            print(f"[Epoch {epoch+1}] Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f">>> Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
            ckpt_dir = os.path.join(args.save_path, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
            torch.save({"epoch": epoch + 1}, os.path.join(ckpt_dir, "meta.pt"))
            print(f"[Checkpoint] Saved to {ckpt_dir}")

        # Save best-performing model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.save_path, "best")
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"[Best] Saved best model to {best_path}")

    # === Final Metrics ===
    total_time = time.time() - start_time
    tokens_per_sec = total_tokens / total_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    model_size_gb = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 ** 3)

    hidden_size = getattr(model.config, "hidden_size", 4096)
    num_layers = getattr(model.config, "num_hidden_layers", 32)
    flops_total = 2 * args.max_tokens * hidden_size * num_layers * total_tokens
    flops_per_sec = flops_total / total_time / 1e12

    print("\n=== Performance Summary ===")
    print(f"Adapter Type     : {args.adapter.upper()}")
    print(f"Trainable Params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total Params     : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model Size       : {model_size_gb:.2f} GB")
    print(f"Peak GPU Memory  : {peak_mem:.2f} GB")
    print(f"Final Loss       : {avg_loss:.4f}")
    print(f"Time Taken       : {total_time:.2f}s | Tokens/sec: {tokens_per_sec:.2f} | TFLOPs: {flops_per_sec:.2f}")

    # === Save final model ===
    final_path = os.path.join(args.save_path, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[Final] Saved final model to {final_path}")

    logger.close()

    print(f"\nTraining completed. Logs saved to {args.save_path}")
    print("Launch TensorBoard with:\n    tensorboard --logdir=runs")


if __name__ == "__main__":
    train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Training script finished.")
    torch.cuda.synchronize()    