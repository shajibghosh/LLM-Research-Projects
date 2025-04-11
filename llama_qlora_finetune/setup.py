# setup.py

from setuptools import setup, find_packages

setup(
    name="llama-finetune",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.37",
        "pymupdf",
        "wandb",
        "matplotlib",
        "tensorboard",
        "pandas"
    ],
    entry_points={
        "console_scripts": [
            "llama-finetune=train.finetune:train",
            "benchmark-lora=scripts.benchmark:main",
            "plot-benchmark=scripts.benchmark_plot:plot_benchmark",
            "merge-lora=scripts.merge_lora:main"
        ]
    },
    author="Shajib Ghosh",
    description="LoRA & QLoRA fine-tuning for LLaMA 3.1 8B without bitsandbytes",
    keywords="lora qlora llama finetuning nlp huggingface",
    python_requires=">=3.8",
)

