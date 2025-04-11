# -*- coding: utf-8 -*-
# PagedAdamW Optimizer for PyTorch
# This optimizer is designed to simulate bitsandbytes' paged_adamw_32bit functionality.
# It optimizes only trainable (LoRA) parameters, keeps optimizer state in FP32 for numerical stability,
# and supports simulation of CPU offloading with minimal memory pressure.
# The optimizer is designed to be used with PyTorch models and is compatible with the Hugging Face Transformers library. 

import torch
from torch.optim import Optimizer


class PagedAdamW(Optimizer):
    """
    Custom optimizer simulating bitsandbytes' paged_adamw_32bit.
    Key features:
    - Optimizes only trainable (LoRA) parameters.
    - Keeps optimizer state in FP32 for numerical stability.
    - Supports simulation of CPU offloading with minimal memory pressure.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # Initialize optimizer defaults like learning rate, betas, epsilon, and weight decay
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:   
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if not 0.0 < betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")  
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # Pass to the base Optimizer class
        super(PagedAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.
        Optionally supports a closure that reevaluates the model.
        """
        # Call closure if provided, to reevaluate the model and get loss
        # This is useful for optimizers that need to compute gradients  
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            # Iterate over each parameter group
            # Each group can have different hyperparameters (lr, betas, etc.)
            for p in group["params"]:
                # Skip if parameter is not a tensor or is not trainable
                if not p.requires_grad:
                    continue
                # Skip if parameter is not a tensor
                if not isinstance(p, torch.Tensor):
                    raise TypeError(f"Parameter {p} is not a tensor")
                if p.grad is None:
                    # Skip if gradient is None (not computed)
                    # This can happen if the parameter is not involved in the loss computation
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    # Skip sparse gradients (not supported)
                    # Sparse gradients are not supported in this implementation
                    raise RuntimeError("PagedAdamW does not support sparse gradients")

                state = self.state[p]

                # Lazy init state
                if len(state) == 0:
                    # Initialize state for the parameter
                    # This is where we store the optimizer state for each parameter
                    state["step"] = 0
                    # Optimizer states in FP32 for numerical stability
                    # This is important for AdamW to avoid numerical issues
                    # when updating the parameters
                    # Initialize exp_avg and exp_avg_sq to zero tensors of the same shape as the parameter
                    # This is where we store the moving averages of the gradients and squared gradients
                    # We use FP32 to avoid numerical issues 
                    # during the optimization step
                    # and to ensure compatibility with CPU offloading   

                    # Exponential moving average of gradients (FP32)
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32, device=p.device)         
                    # Exponential moving average of squared gradients (FP32)                  
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32, device=p.device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Convert grad to FP32 if needed
                grad = grad.to(torch.float32)                   
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)                         
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)                

                # Update the state step counter
                # This is important for the moving averages to be correctly scaled
                state["step"] += 1
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]

                if state["step"] % 1000 == 0:
                    # Print debug information every 1000 steps
                    print(f"[DEBUG] Step {state['step']}: exp_avg={exp_avg}, exp_avg_sq={exp_avg_sq}")  

                update = exp_avg / denom
                if group["weight_decay"] != 0:
                    # Apply weight decay if specified
                    # This is important for AdamW to correctly apply weight decay
                    update += group["weight_decay"] * p.data.to(torch.float32)
                
                # Update the parameter with the computed update
                # Convert the update to the same dtype as the parameter
                p.data.add_(-step_size * update.to(p.data.dtype))

        return loss # Return the loss if closure was provided, else None