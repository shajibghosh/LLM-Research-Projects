# 🧠 Academic PDF to Q&A Dataset Generator

This project is a full pipeline that converts academic PDF documents into high-quality question-answer datasets using LLMs. It supports both extractable-text and scanned PDFs via OCR, generates QA pairs using a Hugging Face-hosted LLaMA model, and performs detailed filtering and scoring of the results to produce a clean, usable dataset.

---

## 📁 Project Structure

```
├── .env                      # Hugging Face API token (HUGGINGFACE_TOKEN=...)
├── run_dataprep.sh          # Shell script to execute the pipeline
├── pdf_to_json.py           # PDF parsing with OCR fallback
├── generate_qa.py           # LLaMA-based QA generation
├── curate_qa.py             # Filtering, scoring, and dataset splitting
├── Pipfile                  # Pipenv-managed dependencies
├── jsons/                   # Output JSONs from PDF conversion
├── data-llm/
│   └── llama31/
│       ├── individual_qas/  # Generated raw QA pairs
│       ├── output/          # Curated train/valid/test splits
│       └── curated/         # Final scored/filtered QA sets, plots, and logs
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-project-directory>
```

### 2. Install Pipenv (if not already)

```bash
pip install pipenv
```

### 3. Install Dependencies

```bash
pipenv install
```

### 4. Activate Virtual Environment

```bash
pipenv shell
```

### 5. Set Hugging Face Token

Create a `.env` file in the root directory with:

```env
HUGGINGFACE_TOKEN=your_hf_token_here
```

---

## ⚙️ Run the Full Pipeline

```bash
chmod +x run_dataprep.sh
./run_dataprep.sh
```

> 💡 You can enable GPU acceleration for OCR and inference by uncommenting lines in `run_dataprep.sh`.

---

## 🧩 Run Individual Steps

### Convert PDFs to Structured JSON

```bash
python pdf_to_json.py --input_dir pdfs --output_dir jsons --gpu
```

### Generate Q&A Pairs from JSONs

```bash
python generate_qa.py --gpu \
  --input_json_folder jsons \
  --output_raw data-llm/llama31/individual_qas \
  --output_curated data-llm/llama31/output
```

### Filter, Score, and Split QA Dataset

```bash
python curate_qa.py \
  --base_dir data-llm/llama31 \
  --input_dir data-llm/llama31/individual_qas
```

---

## 🧠 Model Details

- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Prompting**: Direct plain-text format, no examples
- **Device**: GPU supported (torch + transformers)
- **Context Scaling**: Dynamic RoPE patching
- **Tokens per Gen**: 1500 max per prompt

---

## 📊 Output Artifacts

| Path                                      | Description                              |
|------------------------------------------|------------------------------------------|
| `train.json`, `valid.json`, `test.json`  | Final curated dataset splits             |
| `scored_data.json`                       | Full QA list with readability & score    |
| `filter_log.json`                        | Summary of discarded QA pairs            |
| `stats.json`                             | Token and quality distribution metadata  |
| `plots/*.png`                            | Histograms of scores and lengths         |

---

## 🧼 QA Filtering Rules

QA pairs are **discarded** if:
- Question is not a question (missing `?`)
- Includes placeholder tokens (`<question>`, `<answer>`)
- Academic writing phrases (e.g., “This paper investigates…”)
- Very short or ambiguous answers (“Yes”, “Maybe”)
- Repetitive or duplicated QA pairs

---

## 🧮 Scoring Formula

```
score = 0.6 * completeness + 0.4 * readability
```

- **Completeness**: Word count normalized between 10–100
- **Readability**: Based on Flesch Reading Ease
- **Score ≥ 0.25** is retained

---

## 🪪 License

MIT License

```
MIT License

Copyright (c) 2025 Shajib Ghosh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push to your fork
5. Open a pull request

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Textstat](https://pypi.org/project/textstat/)
