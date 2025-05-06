#!/bin/bash
set -e  # Exit immediately on error

echo "Step 1: Converting PDFs to JSON..."
# Uncomment below to use GPU for OCR (EasyOCR)
python pdf_to_json.py --gpu

# Default: use CPU
# python pdf_to_json.py

echo "Step 2: Generating Q&A from JSON..."
# Uncomment below to use GPU for transformer model
python generate_qa.py --gpu \
   --input_json_folder jsons \
   --output_raw data-llm/llama31/individual_qas \
   --output_curated data-llm/llama31/output

# Default: use CPU
# python generate_qa.py \
#  --input_json_folder jsons \
#  --output_raw data-llm/llama31/individual_qas \
#  --output_curated data-llm/llama31/output

echo "Step 3: Curating and splitting dataset..."
# No GPU needed for this step
python curate_qa.py \
  --base_dir data-llm/llama31 \
  --input_dir data-llm/llama31/individual_qas

echo "All steps completed successfully!"

# -------------------------------
# Usage Notes:
# 1. Uncomment GPU lines if your system supports it.
# 2. Ensure this script and all Python scripts are in the same directory.
# 3. Make executable:
#      chmod +x run_dataprep.sh
# 4. Run it:
#      ./run_dataprep.sh
# -------------------------------