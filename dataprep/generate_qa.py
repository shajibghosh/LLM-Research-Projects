# -*- coding: utf-8 -*-
"""
Generate Q&A pairs from JSON files using a LLaMA model.
This script processes JSON files containing text data, generates question-answer pairs using a LLaMA model,
and saves the generated pairs into separate JSON files. It also handles checkpointing to avoid reprocessing
files that have already been processed.
"""
import os, json, random, hashlib, time, requests, warnings, argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline, logging,
    LlamaConfig
)
from datetime import datetime

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
load_dotenv()


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate Q&A from JSON files using a LLaMA model")
    parser.add_argument("--input_json_folder", default="jsons", help="Folder with input JSON files")
    parser.add_argument("--output_raw", default="data-llm/llama31/individual_qas", help="Folder to save raw QAs")
    parser.add_argument("--output_curated", default="data-llm/llama31/output", help="Folder to save curated train/val/test splits")
    parser.add_argument("--checkpoint_file", default=None, help="Optional: checkpoint file to resume or avoid reprocessing")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID on HuggingFace")
    parser.add_argument("--n_qa", type=int, default=100, help="Number of Q&A pairs to generate per document")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for saving splits")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for model inference if available")
    return parser.parse_args()


def generate_checkpoint_path():
    """
    Generate a unique checkpoint file name based on the current timestamp.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"processed_files_{ts}.txt"


def load_model(model_id, token, use_gpu):
    """
    Load the LLaMA model and tokenizer from Hugging Face.
    """
    print("Loading and patching model config...")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    config_data = requests.get(url, headers=headers).json()
    config_data["rope_scaling"] = {"type": "dynamic", "factor": 1.1}
    config = LlamaConfig.from_dict(config_data)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        device_map="auto" if use_gpu else "cpu",
        torch_dtype=torch.float16 if use_gpu else "auto",
        token=token
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Model loaded.\n")
    return generator


def format_prompt(text, n):
    """
    Format the prompt for the model to generate Q&A pairs.
    Use a direct, plain-text format to avoid generation issues.
    """
    return f"""You are a helpful assistant tasked with generating exactly {n} question and answer pairs from the provided content. 
Each answer should be between 10–100 words. Use the following format:

Q: <question>
A: <answer>

Only use information from the following text:

{text[:3000]}
"""


def parse_qa(text, n):
    """
    Parse the generated text to extract Q&A pairs.
    Cleans out prompt remnants and tolerates slight formatting deviations.
    """
    qas = []
    lines = text.splitlines()
    q, a = None, None

    for line in lines:
        line = line.strip()

        # Stop parsing at common artifacts or echoed prompts
        if line.lower().startswith("text:") or "[/inst]" in line.lower():
            break

        # Identify Q and A blocks
        if line.lower().startswith("q:"):
            if q and a:
                qas.append([q.strip(), a.strip()])
            q = line[2:].strip()
            a = ""
        elif line.lower().startswith("a:"):
            a = line[2:].strip()
        elif q is not None and a is not None:
            a += " " + line.strip()

    if q and a:
        qas.append([q.strip(), a.strip()])

    return qas[:n]


def generate_qa(text, generator, n, fallback_dir=None, file_id=None):
    """
    Generate Q&A pairs from the given text using the model.
    Optionally saves failed generations to file for debugging.
    """
    prompt = format_prompt(text, n)
    start = time.time()
    response = generator(prompt, max_new_tokens=1500, do_sample=False)[0]["generated_text"]
    print(f"\nGenerated Q&A in {round(time.time() - start, 2)} sec")

    qas = parse_qa(response, n)

    if not qas and fallback_dir and file_id:
        Path(fallback_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(fallback_dir, f"{file_id}_raw.txt"), "w", encoding="utf-8") as f:
            f.write(response)

    return qas


def hash_qa(qa):
    """
    Generate a hash for the Q&A pairs to check for duplicates.
    """
    return hashlib.md5(" ".join(q + a for q, a in qa).encode("utf-8")).hexdigest()


def split_and_save(qas, out_dir):
    """
    Split the Q&A pairs into train, validation, and test sets and save them to JSON files.
    """
    random.shuffle(qas)
    total = len(qas)
    train = qas[:int(.8 * total)]
    valid = qas[int(.8 * total):int(.9 * total)]
    test = qas[int(.9 * total):]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, split in tqdm(zip(["train", "valid", "test"], [train, valid, test]), total=3, desc="Saving splits"):
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)

    return train + valid + test


def read_checkpoint(file):
    """
    Read the checkpoint file to get the list of already processed files.
    """
    if Path(file).exists():
        with open(file) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def write_checkpoint(file, filename):
    """
    Write the processed file name to the checkpoint file.
    """
    with open(file, "a") as f:
        f.write(f"{filename}\n")


def process_all_json(args):
    """
    Process all JSON files in the input folder, generate Q&A pairs, and save them.
    """
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    use_gpu = args.gpu and torch.cuda.is_available()
    generator = load_model(args.model_id, HF_TOKEN, use_gpu)

    all_qas = []
    seen = set()
    Path(args.output_raw).mkdir(parents=True, exist_ok=True)
    debug_dir = "debug_outputs"

    checkpoint_file = args.checkpoint_file or generate_checkpoint_path()
    processed = read_checkpoint(checkpoint_file)
    files = list(Path(args.input_json_folder).glob("*.json"))
    print(f"\nTotal files: {len(files)} | Already processed: {len(processed)}")

    for file in tqdm(files, desc="Processing files"):
        if file.name in processed:
            continue

        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)

            sections = data.get("sections", [])
            text = f"{data.get('title', '')}\n\n{data.get('abstract', '')}\n\n"
            for s in tqdm(sections, desc=f"Reading sections in {file.name}", leave=False):
                text += s.get("content", "") + "\n\n"

        except Exception as e:
            print(f"\nSkipping broken file: {file.name} ({e})")
            write_checkpoint(checkpoint_file, file.name)
            continue

        if not text.strip():
            print(f"\nEmpty content in: {file.name}")
            write_checkpoint(checkpoint_file, file.name)
            continue

        try:
            qa = generate_qa(text, generator, args.n_qa, fallback_dir=debug_dir, file_id=file.stem)
            if not qa:
                print(f"\nNo Q&A generated: {file.name}")
                write_checkpoint(checkpoint_file, file.name)
                continue

            h = hash_qa(qa)
            if h in seen:
                print(f"\nDuplicate detected: {file.name}")
                write_checkpoint(checkpoint_file, file.name)
                continue

            seen.add(h)
            all_qas.extend(qa)

            with open(os.path.join(args.output_raw, file.stem + ".json"), "w") as f:
                json.dump(qa, f, indent=2, ensure_ascii=False)

            write_checkpoint(checkpoint_file, file.name)

            if len(all_qas) >= args.batch_size * args.n_qa:
                _ = split_and_save(all_qas, args.output_curated)
                all_qas.clear()

        except Exception as e:
            print(f"\nError processing {file.name}: {e}")
            continue

    if all_qas:
        final = split_and_save(all_qas, args.output_curated)
        print("\nSample generated Q&A pairs:")
        for i, qa_pair in enumerate(random.sample(final, min(3, len(final)))):
            print(f"\nQ{i+1}: {qa_pair[0]}\nA{i+1}: {qa_pair[1]}")
    else:
        print("\nNo QAs to save.")

    print(f"\nAll processing complete. Checkpoint: {checkpoint_file}")


if __name__ == "__main__":
    args = parse_args()
    process_all_json(args)