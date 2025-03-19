# Import Required Libraries
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Directory Setup for Training Logs
LOGS_DIR = "logs_lora_general"  # Directory where LoRA training logs are stored
RESULTS_FILE_PATTERN = "training_results_lora_*.json"  # Pattern to match multiple training result files

# Load Training Results
def load_results():
    """
    Loads training results from multiple JSON files and computes additional performance metrics.

    - Searches for log files matching `training_results_lora_*.json`
    - Merges data from multiple training runs
    - Computes:
        - `speed_tokens_per_sec` (Tokens processed per second)
        - `memory_efficiency` (Tokens processed per GB of GPU memory)
    
    Returns:
        dict: Dictionary containing aggregated training results.
    """
    results = {}  # Dictionary to store merged results

    # Find all JSON log files in the logs directory
    log_files = sorted([
        f for f in os.listdir(LOGS_DIR)
        if f.startswith("training_results_lora") and f.endswith(".json")
    ])

    # If no log files are found, display an error message
    if not log_files:
        print("No training results found! Ensure you have trained the models.")
        return None

    # Load and merge results from multiple JSON files
    for log_file in log_files:
        file_path = os.path.join(LOGS_DIR, log_file)
        with open(file_path, "r") as f:
            data = json.load(f)
            results.update(data)  # Merge all logs

    # Compute additional performance metrics
    for key, metrics in results.items():
        if "speed_tokens_per_sec" not in metrics:
            # Compute token processing speed (tokens per second)
            metrics["speed_tokens_per_sec"] = (
                metrics["total_tokens"] / metrics["training_time"]
                if metrics["training_time"] > 0 else 0
            )
        if "memory_efficiency" not in metrics:
            # Compute memory efficiency (tokens processed per GB of GPU memory)
            metrics["memory_efficiency"] = (
                metrics["total_tokens"] / metrics["gpu_memory_usage"]
                if metrics["gpu_memory_usage"] > 0 else 0
            )

    return results  # Return the processed results dictionary

# GPU Information Display
def print_gpu_info():
    """
    Displays information about the available GPU.

    - Prints GPU name and available VRAM if CUDA is available.
    - If no GPU is detected, displays a message indicating CPU execution.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)  # Get GPU name
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        print(f"\nGPU Detected: {gpu_name} ({total_memory:.2f} GB VRAM)\n")
    else:
        print("\nNo GPU detected, running on CPU.\n")

# Plot Training Loss Curves
def plot_training_loss(results):
    """
    Plots training loss curves for different models and datasets.

    - Extracts `loss_curve` from training results.
    - Each model-dataset combination is plotted separately.
    - Uses a labeled legend for clarity.

    Args:
        results (dict): Dictionary containing training results with loss curves.
    """
    plt.figure(figsize=(12, 6))  # Set plot size

    for key, data in results.items():
        if "loss_curve" in data:
            # Plot loss curve
            plt.plot(data["loss_curve"], label=f"{key} (Dataset: {data['dataset']})")
        else:
            print(f"No loss data found for {key}!")  # Warn if no loss curve data is available

    # Label the axes and title
    plt.xlabel("Training Steps")  # X-axis label
    plt.ylabel("Loss")  # Y-axis label
    plt.title("LoRA Training Loss Comparison Across Datasets")  # Plot title
    plt.legend()  # Display legend with model-dataset names
    plt.grid(True, linestyle="--", alpha=0.6)  # Add grid for readability
    plt.show()  # Display the plot

# Print Performance Summary Table
def print_comparison_table(results):
    """
    Prints a summary table comparing models trained on different datasets.

    - Extracts performance metrics for each model-dataset pair.
    - Formats data into a tabular structure using Pandas.

    Args:
        results (dict): Dictionary containing training results.
    """
    data = []  # List to store table data

    for key, metrics in results.items():
        # Extract model and dataset name from the key
        model_name, dataset_name = key.split("_", 1)

        # Append data as a row
        data.append([
            model_name,
            dataset_name,
            round(metrics["final_loss"], 5),  # Round loss to 5 decimal places
            f"{metrics['training_time']:.2f} sec",  # Training time in seconds
            f"{metrics['total_tokens']:,} tokens",  # Format token count with commas
            f"{metrics['speed_tokens_per_sec']:.2f} tokens/sec",  # Tokens processed per second
            f"{metrics['gpu_memory_usage']:.2f} GB",  # GPU memory usage in GB
            f"{metrics['memory_efficiency']:.5e} tokens/GB"  # Memory efficiency in scientific notation
        ])

    # Convert the data list into a Pandas DataFrame
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

    # Print the formatted table
    print("\n**LoRA Model Performance Summary Across Datasets**\n")
    print(df.to_markdown(index=False))  # Display table in Markdown format

# Load and Visualize Results
if __name__ == "__main__":
    print_gpu_info()  # Display GPU information

    results = load_results()  # Load training results

    if results:  # If results are available, proceed with visualization and analysis
        plot_training_loss(results)  # Plot loss curves
        print_comparison_table(results)  # Print performance summary table