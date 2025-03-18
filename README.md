# **LLaMA Fine-Tuning with LoRA & QLoRA**
This repository provides scripts for fine-tuning **Meta LLaMA 3.1 8B, 3.2 1B, and 3.2 3B models** using **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized Low-Rank Adaptation)**. It supports multiple datasets and multi-GPU training.

## **Table of Contents**
- [Overview](#overview)
- [Installation](#installation)
- [LoRA vs. QLoRA](#lora-vs-qlora)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
  - [1. Setting Up the Environment](#1-setting-up-the-environment)
  - [2. Hugging Face Authentication](#2-hugging-face-authentication)
  - [3. Dataset Preparation](#3-dataset-preparation)
  - [4. Fine-Tuning](#4-fine-tuning)
  - [5. Evaluating Models](#5-evaluating-models)
- [Results Interpretation](#results-interpretation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## **Overview**
This repository enables efficient fine-tuning of **LLaMA models** using **LoRA and QLoRA**. Features include:
- **Multi-GPU training support** using PyTorch Lightning.
- **Efficient parameter tuning** with LoRA and QLoRA.
- **Support for multiple datasets** (Wikitext, OpenAssistant, LLM-PIE).
- **Evaluation tools** for benchmarking performance.

---

## **Installation**
Before running any script, install the required dependencies:

```bash
conda create -n llama-finetune python=3.10 -y
conda activate llama-finetune

pip install torch transformers datasets pytorch_lightning tensorboard accelerate matplotlib pandas pymupdf
```

Ensure your system has **CUDA installed** for GPU-based training.

---

## **LoRA vs. QLoRA**

| Feature      | LoRA | QLoRA |
|-------------|------|-------|
| Trainable Parameters | ✅ Fewer | ✅ Even fewer |
| Quantization | ❌ No quantization | ✅ 4-bit quantization |
| Speed | 🟢 Fast | 🟢 Even faster |
| Memory Usage | 🔵 Moderate | 🟢 Lower |

---

## **File Descriptions**

### **1. Adapter Modules**
- `lora_llama.py` → Implements **LoRA fine-tuning** for LLaMA models.
- `qlora_llama.py` → Implements **QLoRA fine-tuning** with 4-bit quantization.

### **2. Fine-Tuning Scripts**
- `finetune_lora.py` → Fine-tunes **LLaMA** using **LoRA**.
- `finetune_qlora.py` → Fine-tunes **LLaMA** using **QLoRA**.
- `finetune_llama.py` → Supports both **LoRA & QLoRA** with **dataset selection**.
- Features:
  - **Multi-GPU training** with **Distributed Data Parallel (DDP)**.
  - **Dataset selection** (`wikitext`, `openassistant`, `llm-pie`).
  - **Automatic tokenization and preprocessing**.
  - **TensorBoard logging** for monitoring training.

### **3. Evaluation Scripts**
- `eval_lora.py` → Evaluates **LoRA** fine-tuning results.
- `eval_qlora.py` → Evaluates **QLoRA** fine-tuning results.
- `eval_llama.py` → Evaluates **both LoRA & QLoRA fine-tuned models**.
- Features:
  - **Plots training loss curves**.
  - **Compares GPU memory usage**.
  - **Summarizes training performance**.

### **4. System Utilities**
- `gpu_info.py` → Displays **GPU details** (name, memory, utilization, temperature).

---

## **Usage**

### **1. Setting Up the Environment**
Clone the repository and navigate to the directory:

```bash
git clone https://github.com/shajibghosh/LLM-Research-and-Projects.git
cd LLM-Research-and-Projects
conda activate llama-finetune
```

---

### **2. Hugging Face Authentication**
Since models are downloaded from **Hugging Face**, authentication is required:

```bash
export HF_TOKEN="your_huggingface_token"
```

Or, the script will prompt for manual entry.

---

### **3. Dataset Preparation**
#### **Load Predefined Datasets**
```bash
python finetune_llama.py --method lora --dataset wikitext
```

#### **Use Custom Dataset (PDF Files)**
Place **PDF documents** inside the `datasets/llm-pie/` folder. The script will extract text automatically.

---

### **4. Fine-Tuning**
#### **Fine-Tune LLaMA with LoRA**
```bash
python finetune_lora.py
```

#### **Fine-Tune LLaMA with QLoRA**
```bash
python finetune_qlora.py
```

#### **Fine-Tune LLaMA (Select Method & Dataset)**
```bash
python finetune_llama.py --method qlora --dataset openassistant
```

- Models are saved in:
  - `trained_models_lora/`
  - `trained_models_qlora/`
  - `trained_models_llama/`
- Logs are stored in:
  - `logs_lora_general/`
  - `logs_qlora_general/`
  - `logs_llama/`

---

### **5. Evaluating Models**
#### **Evaluate LoRA Fine-Tuned Models**
```bash
python eval_lora.py
```

#### **Evaluate QLoRA Fine-Tuned Models**
```bash
python eval_qlora.py
```

#### **Evaluate All LLaMA Fine-Tuned Models**
```bash
python eval_llama.py
```

These scripts generate:
- **Training loss plots** for all models.
- **Performance summary tables** comparing models.
- **GPU memory usage reports**.

---

## **Results Interpretation**
The evaluation scripts generate training statistics, including:

| Metric | Description |
|--------|------------|
| **Final Loss** | Last recorded training loss |
| **Training Time** | Time taken to complete training |
| **Total Tokens Processed** | Number of tokens used during training |
| **Tokens per Second** | Training speed in tokens/sec |
| **GPU Memory Usage** | Peak memory usage during training |
| **Memory Efficiency** | Tokens processed per GB of memory |

---

## **Acknowledgements**
This project acknowledges contributions from:
- **[Meta AI](https://ai.meta.com/research/models/llama/)** for developing **LLaMA** models.
- **[Hugging Face](https://huggingface.co/meta-llama/)** for hosting models and providing dataset tools.

### **LLaMA Model Links**
- **[LLaMA 3.2 1B (Meta)](https://huggingface.co/meta-llama/Llama-3.2-1B)**
- **[LLaMA 3.2 3B (Meta)](https://huggingface.co/meta-llama/Llama-3.2-3B)**
- **[LLaMA 3.1 8B (Meta)](https://huggingface.co/meta-llama/Llama-3.1-8B)**

---

## **License**
This project is open-source and licensed under the **MIT License**.

---

## **Citing**
If you use this repository for research or development, please consider citing:

```bibtex
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Hugo Touvron and others},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```
