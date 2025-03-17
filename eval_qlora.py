import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Directory for QLoRA Training Logs
LOGS_DIR = "logs_qlora_general"
RESULTS_FILE_PATTERN = "training_results_qlora_*.json"  # Match multiple runs

# Load Training Results
def load_results():
    """Load results from multiple JSON files."""
    results = {}
    log_files = sorted([f for f in os.listdir(LOGS_DIR) if f.startswith("training_results_qlora") and f.endswith(".json")])

    if not log_files:
        print("No training results found! Ensure you have trained the models.")
        return None

    for log_file in log_files:
        file_path = os.path.join(LOGS_DIR, log_file)
        with open(file_path, "r") as f:
            data = json.load(f)
            results.update(data)  # Merge all logs

    # Compute additional metrics
    for key, metrics in results.items():
        if "speed_tokens_per_sec" not in metrics:
            metrics["speed_tokens_per_sec"] = metrics["total_tokens"] / metrics["training_time"] if metrics["training_time"] > 0 else 0
        if "memory_efficiency" not in metrics:
            metrics["memory_efficiency"] = metrics["total_tokens"] / metrics["gpu_memory_usage"] if metrics["gpu_memory_usage"] > 0 else 0

    return results

# GPU Info
def print_gpu_info():
    """Display available GPU details."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        print(f"\nGPU Detected: {gpu_name} ({total_memory:.2f} GB VRAM)\n")
    else:
        print("\nNo GPU detected, running on CPU.\n")

# Plot Training Loss Curves
def plot_training_loss(results):
    """Plot training loss curves for different models and datasets."""
    plt.figure(figsize=(12, 6))

    for key, data in results.items():
        if "loss_curve" in data:
            plt.plot(data["loss_curve"], label=f"{key} (Dataset: {data['dataset']})")
        else:
            print(f"No loss data found for {key}!")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("QLoRA Training Loss Comparison Across Datasets")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# Print Performance Summary Table
def print_comparison_table(results):
    """Print a summary table comparing models trained on different datasets."""
    data = []
    for key, metrics in results.items():
        model_name, dataset_name = key.split("_", 1)
        data.append([
            model_name,
            dataset_name,
            round(metrics["final_loss"], 5),
            f"{metrics['training_time']:.2f} sec",
            f"{metrics['total_tokens']:,} tokens",
            f"{metrics['speed_tokens_per_sec']:.2f} tokens/sec",
            f"{metrics['gpu_memory_usage']:.2f} GB",
            f"{metrics['memory_efficiency']:.5e} tokens/GB"
        ])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        "Model",
        "Dataset",
        "Final Loss",
        "Training Time",
        "Total Tokens",
        "Speed (tokens/sec)",
        "GPU Memory Usage",
        "Memory Efficiency (tokens/GB)"
    ])

    print("\n**QLoRA Model Performance Summary Across Datasets**\n")
    print(df.to_markdown(index=False))

# Load and Visualize Results
if __name__ == "__main__":
    print_gpu_info()
    
    results = load_results()
    if results:
        plot_training_loss(results)
        print_comparison_table(results)
