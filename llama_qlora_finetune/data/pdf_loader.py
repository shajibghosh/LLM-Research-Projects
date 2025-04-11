# -*- coding: utf-8 -*-
# PDF Loader for PyTorch-based LLM fine-tuning pipelines.
# This module extracts and tokenizes text from PDFs using PyMuPDF and Hugging Face Transformers.

import os
import fitz  # PyMuPDF
from typing import List, Dict
from transformers import PreTrainedTokenizer
import torch 


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text.
    """
    doc: fitz.Document = fitz.open(pdf_path)  # Open the PDF document
    text = ""
    for page in doc:  # Iterate over pages
        text += page.get_text()  # Extract text from each page
    doc.close()
    return text.strip()  # Remove leading/trailing whitespace


def chunk_text(text: str, max_tokens: int = 512, stride: int = 256, tokenizer: PreTrainedTokenizer = None) -> List[Dict]:
    """
    Split a long text into overlapping token chunks and tokenize.
    Args:
        text (str): The input text.
        max_tokens (int): Max tokens per chunk.
        stride (int): Token overlap between chunks.
        tokenizer (PreTrainedTokenizer): HF tokenizer to use.
    Returns:
        List[Dict]: List of tokenized chunks (dicts with input_ids and attention_mask).
    """
    if not text or tokenizer is None:
        return []

    # Tokenize with chunking and overlap using return_overflowing_tokens
    tokenized = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors=None  # Return Python lists instead of tensors
    )

    tokenized_samples = []

    # Extract token chunks from tokenizer output
    input_ids_list = tokenized["input_ids"]
    attention_mask_list = tokenized["attention_mask"]

    for i, input_ids in enumerate(input_ids_list):
        tokenized_samples.append({
            "input_ids": torch.tensor(input_ids),  # convert to tensor manually
            "attention_mask": torch.tensor(attention_mask_list[i])
        })

    return tokenized_samples


def load_pdf_dataset_from_dir(pdf_dir: str, tokenizer: PreTrainedTokenizer, max_tokens: int = 512, stride: int = 256) -> List[Dict]:
    """
    Load all PDFs from a directory and return tokenized training-ready samples.
    Args:
        pdf_dir (str): Directory containing PDF files.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        max_tokens (int): Max tokens per chunk.
        stride (int): Overlap between chunks.
    Returns:
        List[Dict]: List of tokenized samples.
    """
    all_samples = []

    # List only .pdf files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    for filename in pdf_files:
        path = os.path.join(pdf_dir, filename)
        text = extract_text_from_pdf(path)

        if not text:
            print(f"[WARN] Skipping empty PDF: {filename}")
            continue

        # Tokenize the extracted text into chunks
        chunks = chunk_text(text, max_tokens=max_tokens, stride=stride, tokenizer=tokenizer)

        if not chunks:
            print(f"[WARN] No chunks extracted from: {filename}")
        else:
            all_samples.extend(chunks)

    if not all_samples:
        raise ValueError("No tokenized samples could be generated from the provided PDFs.")

    print(f"[INFO] Loaded {len(all_samples)} tokenized samples from {len(pdf_files)} PDFs.")
    return all_samples
