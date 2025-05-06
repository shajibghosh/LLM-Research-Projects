# This script filters and curates QA pairs from individual outputs.
# -*- coding: utf-8 -*-
import os, json, random, re, hashlib, argparse
from pathlib import Path
from tqdm import tqdm
from textstat import flesch_reading_ease
import matplotlib.pyplot as plt

# Some academic phrases that are not suitable for QA pairs
# These phrases are often used in academic writing and may not be appropriate for general QA pairs
BAD_QUESTION_PHRASES = [
    "in this study", "in this paper", "this paper presents", "this study investigates",
    "we propose", "we investigate", "we present", "this work", "this research",
    "we demonstrate", "our approach", "our method", "our results show",
    "the purpose of this study", "this article discusses", "the aim of this research",
    "the focus of this paper", "we introduce", "this document", "our findings indicate",
    "the paper aims to", "this analysis", "in the study", "in the paper",
    "in the proposed system", "in the proposed solution", "of the paper"
]

# Initialize a filter log to keep track of the filtering process
# This log will help in understanding how many QA pairs were filtered out and why
# The keys represent different filtering criteria, and the values are the counts of each criterion
# The "kept" key will store the number of QA pairs that were kept after filtering
# The "total" key will store the total number of QA pairs processed
# The "placeholder" key will store the number of QA pairs that contained placeholders
# The "non_question" key will store the number of QA pairs that did not end with a question mark
# The "short_answer" key will store the number of QA pairs that had short answers
# The "empty" key will store the number of QA pairs that were empty
# The "low_score" key will store the number of QA pairs that had a low score
# The "academic_phrase" key will store the number of QA pairs that contained academic phrases
# The "repetitive_answer" key will store the number of QA pairs that had repetitive answers
# The "ambiguous_answer" key will store the number of QA pairs that had ambiguous answers
# The "duplicate" key will store the number of QA pairs that were duplicates

filter_log = {
    "placeholder": 0, "non_question": 0, "short_answer": 0, "empty": 0, "low_score": 0,
    "academic_phrase": 0, "repetitive_answer": 0, "ambiguous_answer": 0,
    "duplicate": 0, "kept": 0, "total": 0
}


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Filter and curate QA pairs from individual outputs")
    parser.add_argument("--base_dir", default="data-llm-pie/llama31", help="Base directory for outputs")
    parser.add_argument("--input_dir", default="data-llm-pie/llama31/individual_qas", help="Directory of raw QA JSONs")
    parser.add_argument("--score_threshold", type=float, default=0.25, help="Minimum score to keep a QA pair")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    return parser.parse_args()


def score_answer(answer, min_words=10, max_words=100):
    """
    Score the answer based on completeness and readability.
    Completeness is based on the number of words in the answer,
    and readability is based on the Flesch Reading Ease score.
    """
    words = answer.split()
    word_count = len(words)
    completeness = min(1.0, max(0.0, (word_count - min_words) / (max_words - min_words)))
    try:
        readability_raw = flesch_reading_ease(answer)
        readability = min(1.0, max(0.0, readability_raw / 100))
    except:
        readability = 0.0
    score = round(0.6 * completeness + 0.4 * readability, 3)     # Calculate the final score
    return {
        "completeness": round(completeness, 3),
        "readability": round(readability, 3),
        "score": score
    }


def clean_qa_pair(q, a, args):
    """
    Clean and filter the QA pair.
    This function checks for various conditions such as:
    - Placeholder text
    - Non-question format
    - Short answer
    - Empty question or answer
    - Low score
    - Academic phrases
    - Repetitive answer
    - Ambiguous answer
    """
    filter_log["total"] += 1
    q = re.sub(r"\[/?INST\]", "", q.replace("Q:", "").strip())
    a = re.sub(r"(\[/?INST\]|Text:.*)", "", a.replace("A:", "").strip())

    if "<question>" in q.lower() or "<answer>" in a.lower():
        filter_log["placeholder"] += 1
        return None
    if not q.endswith("?"):
        filter_log["non_question"] += 1
        return None
    if len(a.split()) < 3 or not re.search(r"\w", a):
        filter_log["short_answer"] += 1
        return None
    if not q or not a:
        filter_log["empty"] += 1
        return None

    if any(re.search(rf"\b{re.escape(p)}\b", q.lower()) for p in BAD_QUESTION_PHRASES):
        filter_log["academic_phrase"] += 1
        if args.debug:
            print("\n? Academic phrase:\n", q)
        return None

    score_data = score_answer(a)
    if score_data["score"] < args.score_threshold:
        filter_log["low_score"] += 1
        if args.debug:
            print("\n? Low score:\n", q, "\n", a, "\n", score_data)
        return None

    q_norm = re.sub(r"[^\w\s]", "", q.lower()).strip()
    a_norm = re.sub(r"[^\w\s]", "", a.lower()).strip()
    if a_norm.startswith(("yes", "no", "maybe")) and (q_norm in a_norm or a_norm.endswith(q_norm)):
        filter_log["repetitive_answer"] += 1
        return None
    if a_norm in ["yes", "no", "maybe", "it depends", "not sure"]:
        filter_log["ambiguous_answer"] += 1
        return None

    return q.strip(), a.strip(), score_data


def process_individual_files(args, curated_dir):
    """
    Process individual QA files and filter them based on various criteria.
    This function reads the JSON files, cleans the QA pairs, and saves the filtered data.
    """
    all_qas = []
    seen_hashes = set()
    files = list(Path(args.input_dir).glob("*.json"))
    print(f"Processing {len(files)} individual QA files...\n")

    for file in tqdm(files, desc="Reformatting files"):
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
            for pair in data:
                if isinstance(pair, list) and len(pair) == 2:
                    cleaned = clean_qa_pair(pair[0], pair[1], args)
                    if cleaned:
                        q, a, score_data = cleaned
                        pair_hash = hashlib.md5((q + a).encode("utf-8")).hexdigest()
                        if pair_hash in seen_hashes:
                            filter_log["duplicate"] += 1
                            continue
                        seen_hashes.add(pair_hash)
                        all_qas.append({
                            "question": q, "answer": a,
                            "score": score_data,
                            "reviewed": False,
                            "accepted": None
                        })
                        filter_log["kept"] += 1
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    Path(curated_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(curated_dir, "scored_data.json"), "w", encoding="utf-8") as f:
        json.dump(all_qas, f, indent=2, ensure_ascii=False)
    with open(os.path.join(curated_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump([[q["question"], q["answer"]] for q in all_qas], f, indent=2, ensure_ascii=False)
    return all_qas


def split_dataset(data, curated_dir):
    """
    Split the dataset into training, validation, and test sets.
    The split is done in a 80-10-10 ratio.
    """
    random.shuffle(data)
    pairs = [[q["question"], q["answer"]] for q in data]
    total = len(pairs)
    train = pairs[:int(total * 0.8)]
    valid = pairs[int(total * 0.8):int(total * 0.9)]
    test = pairs[int(total * 0.9):]

    with open(os.path.join(curated_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(os.path.join(curated_dir, "valid.json"), "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)
    with open(os.path.join(curated_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    print(f"\ntrain.json: {len(train)}\nvalid.json: {len(valid)}\ntest.json: {len(test)}")
    return train, valid, test


def save_filter_log(curated_dir):
    """
    Save the filter log to a JSON file.                         
    This log contains the counts of various filtering criteria.
    """
    with open(os.path.join(curated_dir, "filter_log.json"), "w") as f:
        json.dump(filter_log, f, indent=2)
    print("\nFilter summary:")
    for k, v in filter_log.items():
        if k != "kept":
            print(f" - {k}: {v}")
    print(f"\nKept: {filter_log['kept']} / {filter_log['total']} ({round((filter_log['kept']/max(1, filter_log['total']))*100, 2)}%)")


def compute_stats(dataset):
    """
    Compute statistics for the dataset.
    This function calculates the average, maximum, and minimum lengths of questions and answers.
    """
    q_lengths = [len(q.split()) for q, _ in dataset]
    a_lengths = [len(a.split()) for _, a in dataset]
    return {
        "count": len(dataset),
        "avg_q_len": round(sum(q_lengths) / len(q_lengths), 2),
        "avg_a_len": round(sum(a_lengths) / len(a_lengths), 2),
        "max_q_len": max(q_lengths),
        "max_a_len": max(a_lengths),
        "min_q_len": min(q_lengths),
        "min_a_len": min(a_lengths)
    }


def compute_score_summary(dataset):
    """
    Compute a summary of scores for the dataset.
    This function calculates the average score, readability, and completeness of the answers.
    """
    scores = [score_answer(a) for _, a in dataset]
    return {
        "avg_score": round(sum(s["score"] for s in scores) / len(scores), 3),
        "avg_readability": round(sum(s["readability"] for s in scores) / len(scores), 3),
        "avg_completeness": round(sum(s["completeness"] for s in scores) / len(scores), 3)
    }


def save_all_stats(train, valid, test, curated_dir):
    """
    Save all statistics to a JSON file.
    This function includes the statistics for training, validation, and test sets,
    as well as the filter summary.
    """
    stats = {
        "train": compute_stats(train),
        "valid": compute_stats(valid),
        "test": compute_stats(test),
        "train_quality": compute_score_summary(train),
        "valid_quality": compute_score_summary(valid),
        "test_quality": compute_score_summary(test),
        "filter_summary": filter_log
    }
    with open(os.path.join(curated_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("\nDataset and score stats saved.")

def save_plot(data, title, xlabel, filename, out_dir):
    """
    Save a histogram plot of the given data.
    This function creates a histogram of the data and saves it to the specified directory.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=20, color="#4A90E2", edgecolor="black", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close()

def visualize_stats(qas, out_dir):
    """
    Visualize the statistics of the dataset.
    This function creates histograms for question lengths, answer lengths, scores, readability, and completeness.
    """
    print("\nGenerating plots...")
    questions = [len(q["question"].split()) for q in qas]
    answers = [len(q["answer"].split()) for q in qas]
    scores = [q["score"]["score"] for q in qas]
    readability = [q["score"]["readability"] for q in qas]
    completeness = [q["score"]["completeness"] for q in qas]

    save_plot(questions, "Question Length Distribution", "Words", "question_lengths.png", out_dir)
    save_plot(answers, "Answer Length Distribution", "Words", "answer_lengths.png", out_dir)
    save_plot(scores, "Overall Score Distribution", "Score", "scores.png", out_dir)
    save_plot(readability, "Readability Score", "Score", "readability.png", out_dir)
    save_plot(completeness, "Completeness Score", "Score", "completeness.png", out_dir)


def main():
    """
    Main function to run the script.
    This function parses the command-line arguments, processes the individual files,
    splits the dataset, saves the filter log, computes and saves statistics,
    and generates visualizations.
    """
    print("\nCurating QA pairs from individual outputs...\n")
    args = parse_args()
    curated_dir = os.path.join(args.base_dir, "curated")

    merged = process_individual_files(args, curated_dir)
    train, valid, test = split_dataset(merged, curated_dir)
    save_filter_log(curated_dir)
    save_all_stats(train, valid, test, curated_dir)
    plot_dir = os.path.join(curated_dir, "plots")
    visualize_stats(merged, plot_dir)
    print("\nAll stats saved.")



if __name__ == "__main__":
    main()