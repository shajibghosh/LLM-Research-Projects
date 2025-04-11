# train.py
import os
import torch
from tqdm.auto import tqdm
from dotenv import load_dotenv
from datetime import datetime
from torch import nn, optim
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import Accelerator
import deepspeed

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["WANDB_PROJECT"] = "llama3-qlora-finetune"

# Check CUDA version
def check_cuda_version():
    try:
        torch_cuda_version = torch.version.cuda
        deepspeed_cuda_version = deepspeed.ops.comm.torch_cuda_version
        if torch_cuda_version != deepspeed_cuda_version:
            print(f"Warning: Torch CUDA version ({torch_cuda_version}) "
                  f"does not match DeepSpeed CUDA version ({deepspeed_cuda_version}). "
                  f"This might cause compatibility issues.")
    except AttributeError as e:
        print(f"Could not determine CUDA version. Please ensure CUDA is properly configured.")

check_cuda_version()

# Custom 4-bit Quantized Linear Layer
class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Simulate 4-bit storage with 8-bit quantization
        quantized = torch.quantize_per_tensor(
            self.weight, 
            scale=self.scale.item(), 
            zero_point=0, 
            dtype=torch.qint8
        )
        return nn.functional.linear(x, quantized.dequantize())

# Custom QLoRA Model Wrapper
def prepare_qlora_model(model_id, rank=8):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )
    
    # Replace linear layers with quantized versions
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, QuantLinear(module.in_features, module.out_features))
    
    # Add LoRA adapters
    config = LoraConfig(
        r=rank,
        lora_alpha=rank*2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)

# Custom Progress Callback
class QLoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar = None
    
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=self.state.max_steps,
                desc="Training",
                unit="step",
                dynamic_ncols=True
            )
        self.progress_bar.update(1)
        self.progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{self.state.optimizer.param_groups[0]['lr']:.2e}"
        })
        return loss

# Main Training Function
def main():
    # Configuration
    set_seed(42)
    model_id = "meta-llama/Llama-3.1-8B"
    dataset_id = "wikitext"
    batch_size = 1
    gradient_accumulation = 16
    max_length = 512  # Define the maximum sequence length

    # Initialize accelerator
    accelerator = Accelerator()

    # Load model and tokenizer
    model = prepare_qlora_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze all parameters *except* LoRA adapters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    # ? Sanity check: Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"? Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.4f}%)")

    # Load and prepare dataset
    dataset = load_dataset(dataset_id, "wikitext-2-raw-v1", split="train")

    def tokenize_function(examples):
        # Tokenize and truncate
        result = tokenizer(examples["text"], truncation=True, max_length=max_length)
        return result

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training arguments
    args = TrainingArguments(
        output_dir=f"./checkpoints-{datetime.now().strftime('%Y%m%d-%H%M')}",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        report_to=["wandb"],
        deepspeed="ds_config.json",
        remove_unused_columns=False  # Keep columns needed for DataCollator
    )

    # Initialize trainer
    trainer = QLoRATrainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start training
    accelerator.print("Starting training...")
    trainer.train()
    accelerator.wait_for_everyone()

    # Save final model
    if accelerator.is_main_process:
        model.save_pretrained("./final_model")
        tokenizer.save_pretrained("./final_model")

if __name__ == "__main__":
    main()
