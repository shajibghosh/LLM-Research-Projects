# utils/logging.py

import os
from torch.utils.tensorboard import SummaryWriter
import wandb


class Logger:
    """
    Logging utility that writes to TensorBoard and optionally to Weights & Biases.
    """

    def __init__(self, project="llama-finetune", run_name=None, log_dir="runs"):
        self.enabled = os.getenv("USE_WANDB", "false").lower() == "true"

        # Create a dedicated TensorBoard subdir per run
        self.tb = SummaryWriter(log_dir=os.path.join(log_dir, run_name or "default"))

        if self.enabled:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            self.wandb = wandb.init(
                project=os.getenv("WANDB_PROJECT", project),
                name=os.getenv("WANDB_NAME", run_name),
                reinit=True  # safer than deprecated `finish_previous`
            )
        else:
            self.wandb = None

    def log(self, step, loss, lr):
        """Log loss and LR to TensorBoard + Weights & Biases (if enabled)."""
        self.tb.add_scalar("loss", loss, step)
        self.tb.add_scalar("learning_rate", lr, step)

        if self.enabled and self.wandb:
            self.wandb.log({"loss": loss, "lr": lr, "step": step})

    def close(self):
        """Gracefully shutdown both loggers."""
        self.tb.close()
        if self.enabled and self.wandb:
            self.wandb.finish()