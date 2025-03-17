# **LoRA & QLoRA Fine-Tuning and Evaluation for LLaMA**
This repository provides a comprehensive setup for fine-tuning **Meta-LLaMA** models using **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized Low-Rank Adaptation)** techniques. It includes training scripts, evaluation tools, and performance benchmarking across different datasets.

## **Table of Contents**
- [Overview](#overview)
- [Installation](#installation)
- [LoRA vs. QLoRA](#lora-vs-qlora)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
  - [1. Setting Up Environment](#1-setting-up-environment)
  - [2. Hugging Face Authentication](#2-hugging-face-authentication)
  - [3. Fine-Tuning](#3-fine-tuning)
  - [4. Evaluating Models](#4-evaluating-models)
- [Results Interpretation](#results-interpretation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## **Overview**
This repository contains scripts to fine-tune **LLaMA models** using LoRA and QLoRA for efficient parameter adaptation. The key features include:
- **Efficient fine-tuning** with reduced memory footprint.
- **Multi-GPU support** using PyTorch Lightning.
- **Evaluation scripts** to compare training loss, speed, and memory efficiency.

---

## **Installation**
Before running any script, install the required dependencies:

```bash
pip install torch transformers datasets pytorch_lightning tensorboard accelerate matplotlib pandas
```

Additionally, **ensure your system has CUDA installed** for GPU-based training.

---

## **LoRA vs. QLoRA**

| Feature      | LoRA | QLoRA |
|-------------|------|-------|
| Parameter Efficiency | ✅ Uses fewer trainable parameters | ✅ Further reduces memory usage |
| Quantization | ❌ No quantization | ✅ Uses 4-bit quantization |
| Speed | 🔵 Faster | 🔵 Even faster due to reduced memory footprint |
| Memory Usage | 🔵 Moderate | 🟢 Lower |

---

## **File Descriptions**

### **1. Adapter Modules**
- `lora_llama.py` → Implements LoRA-based fine-tuning for LLaMA models.
- `qlora_llama.py` → Implements QLoRA fine-tuning with 4-bit quantization.

### **2. Fine-Tuning Scripts**
- `finetune_lora.py` → Fine-tunes LLaMA using **LoRA**.
- `finetune_qlora.py` → Fine-tunes LLaMA using **QLoRA**.
- Features:
  - Multi-GPU training support (Distributed Data Parallel).
  - Supports datasets like **Wikitext** and **OpenAssistant**.
  - Logs training metrics using **TensorBoard**.

### **3. Evaluation Scripts**
- `eval_lora.py` → Evaluates LoRA fine-tuning results.
- `eval_qlora.py` → Evaluates QLoRA fine-tuning results.
- Features:
  - **Plots training loss curves**.
  - **Displays GPU memory usage and efficiency**.
  - **Prints a comparison table** summarizing training performance.

---

## **Usage**

### **1. Setting Up Environment**
Clone the repository and navigate to the directory:

```bash
git clone https://github.com/shajibghosh/LLM-Research-and-Projects.git
cd LLM-Research-and-Projects
```

---

### **2. Hugging Face Authentication**
Since models are downloaded from **Hugging Face**, you need to authenticate:

```bash
export HF_TOKEN="your_huggingface_token"
```

Or, the script will prompt you to enter it manually.

---

### **3. Fine-Tuning**

#### **Fine-Tune LLaMA with LoRA**
```bash
python finetune_lora.py
```

#### **Fine-Tune LLaMA with QLoRA**
```bash
python finetune_qlora.py
```

- Models will be saved in:
  - `trained_models_lora/`
  - `trained_models_qlora/`
- Logs are stored in:
  - `logs_lora_general/`
  - `logs_qlora_general/`

---

### **4. Evaluating Models**

#### **Evaluate LoRA Fine-Tuned Models**
```bash
python eval_lora.py
```

#### **Evaluate QLoRA Fine-Tuned Models**
```bash
python eval_qlora.py
```

These scripts generate:
- **Training loss plots** for all models.
- **Performance summary tables** comparing models.
- **GPU memory usage reports**.

---

## **Results Interpretation**
The evaluation scripts generate detailed training statistics. Key metrics include:

| Metric | Description |
|--------|------------|
| **Final Loss** | Last recorded training loss |
| **Training Time** | Time taken to complete training |
| **Total Tokens Processed** | Number of tokens used during training |
| **Tokens per Second** | Training speed in tokens/sec |
| **GPU Memory Usage** | Peak memory usage during training |
| **Memory Efficiency** | Number of tokens processed per GB of memory |

---

## **Acknowledgements**
This project acknowledges the contributions of:
- **Meta AI** for developing the **LLaMA** models.
  - [LLaMA 3.2 1B on Meta AI](https://ai.meta.com/resources/models-and-libraries/llama-3/)
  - [LLaMA 3.2 3B on Meta AI](https://ai.meta.com/resources/models-and-libraries/llama-3/)
- **Hugging Face** for providing an extensive ecosystem for model training, hosting, and dataset management.
  - [LLaMA 3.2 1B on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B)
  - [LLaMA 3.2 3B on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B)

---

## **License**
This project is open-source and licensed under the **MIT License**.