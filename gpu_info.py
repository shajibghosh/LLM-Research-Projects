import torch
import subprocess

def get_gpu_details():
    if not torch.cuda.is_available():
        print("No GPU detected.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Total GPUs available: {num_gpus}")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / 1e9  # GB
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # GB
        
        print(f"\nGPU {i}: {gpu_name}")
        print(f"  - Compute Capability: {capability[0]}.{capability[1]}")
        print(f"  - Total Memory: {total_memory:.2f} GB")
        print(f"  - Memory Allocated (used by tensors): {memory_allocated:.2f} GB")
        print(f"  - Memory Reserved (cached for PyTorch): {memory_reserved:.2f} GB")

        # Get real-time utilization and temperature from nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, text=True, check=True
            )
            lines = result.stdout.strip().split("\n")
            if i < len(lines):  # Ensure we don't access an out-of-bounds index
                values = lines[i].split(", ")
                if len(values) == 2:
                    utilization, temperature = values
                    print(f"  - GPU Utilization: {utilization}%")
                    print(f"  - Temperature: {temperature}°C")
                else:
                    print(f"  - Unexpected format from nvidia-smi: {values}")
            else:
                print(f"  - No utilization/temperature data available for GPU {i}")
        except Exception as e:
            print(f"  - Failed to retrieve utilization & temperature: {e}")

    # Get system-wide CUDA & Driver version
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.split("\n")
        for line in lines:
            if "Driver Version" in line and "CUDA Version" in line:
                print(f"\nSystem Info: {line.strip()}")
                break
    except Exception as e:
        print("\nFailed to retrieve CUDA and Driver version:", e)

if __name__ == "__main__":
    get_gpu_details()