# This script converts PDF files to structured JSON files with optional OCR for scanned documents.
# -*- coding: utf-8 -*-
import os
import json
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from tqdm import tqdm
import easyocr
import argparse

import warnings
warnings.filterwarnings("ignore")


def extract_text_from_pdf(pdf_path, reader):
    """
    Extract text from a PDF file. If the text is not extractable (e.g., scanned document),
    use OCR to extract text from images.
    """
    doc = fitz.open(pdf_path)                  # Open the PDF file
    full_text = ""                             # Initialize an empty string for full text
    image_based = False                        # Flag to check if OCR is needed

    for page in doc:
        text = page.get_text()                 # Extract text from the page
        if text.strip():                       # If text is extractable, append it to full_text
            full_text += text + "\n"           # Add a newline for separation
        else:                                  # If no text is found, mark as image-based
            image_based = True
            break                              # Exit early to switch to OCR for all pages

    if image_based:                                                                   # If the document is image-based, use OCR
        print(f"[OCR] Scanned document detected: {os.path.basename(pdf_path)}")       # Log the OCR process
        full_text = ""                                                                # Reset full_text for OCR result
        for page in doc:                                                              # Iterate through each page
            pix = page.get_pixmap(dpi=300)                                            # Get a pixmap of the page
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)        # Convert to PIL Image
            result = reader.readtext(np.array(img), detail=0, paragraph=True)         # Use OCR to extract text
            full_text += "\n".join(map(str, result)) + "\n"                           # Append the OCR result to full_text
    else:
        print(f"[INFO] Extracted text from: {os.path.basename(pdf_path)}")            # Log the text extraction

    return full_text.strip()                                                          # Return the full text, stripping whitespace


def guess_title(text):
    """
    Guess the title of the document based on the first few lines.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]                            # Split the text into lines and remove empty lines
    candidates = [line for line in lines[:10] if line.istitle() or line.isupper()]        # Find lines that are title-like (capitalized or all uppercase)
    return candidates[0] if candidates else lines[0] if lines else "Untitled Document"    # Return a guess for the title


def guess_abstract(text):
    """
    Guess the abstract of the document based on the presence of the word "abstract".
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]                            # Split the text into lines and remove empty lines
    joined = "\n".join(lines).lower()                                                     # Join the lines into a single string and convert to lowercase
    if "abstract" in joined:                                                              # Check if "abstract" is in the text
        abstract_start = joined.find("abstract")
        intro_start = joined.find("introduction", abstract_start)                        # Find the start of the introduction section
        end = intro_start if intro_start > abstract_start else abstract_start + 800      # Set the end point for the abstract
        return text[abstract_start + len("abstract"):end].strip()                        # Return the abstract text
    else:
        return " ".join(lines[:5])                                                       # Return the first 5 lines as fallback


def extract_sections(text):
    """
    Extract sections from the text based on known headings.
    """
    lines = text.split('\n')                                          # Split the text into lines
    lines = [l.strip() for l in lines if l.strip()]                   # Remove empty lines
    sections = []                                                     # Initialize an empty list for sections
    current_heading = None                                            # Initialize the current heading
    current_content = []                                              # Initialize the current content

    known_headings = [                                                     # List of known headings to identify sections
        "abstract", "introduction", "background", "experimental",
        "methods", "results", "discussion", "conclusion", "references"
    ]

    for line in lines:                                                                             # Iterate through each line
        line = line.strip()                                                                        # Strip leading/trailing whitespace
        if line.lower() in known_headings or line.lower().startswith(tuple("ivxlc1234567890")):    # Check if the line is a heading
            if current_heading:                                                                    # Save the previous section if any
                sections.append({
                    "heading": current_heading.lower(),
                    "content": " ".join(current_content).strip()
                })
            current_heading = line
            current_content = []
        else:
            current_content.append(line)

    if current_heading and current_content:                                                        # Append any remaining section
        sections.append({
            "heading": current_heading.lower(),
            "content": " ".join(current_content).strip()
        })

    if not sections:                                                                               # Fallback if no sections detected
        sections = [{
            "heading": "content",
            "content": text.strip()
        }]
    return sections


def process_pdf_file(pdf_path, reader):
    """
    Process a single PDF file and extract its title, abstract, and sections.
    """
    try:
        text = extract_text_from_pdf(pdf_path, reader)                                             # Extract text from the PDF
        title = guess_title(text)                                                                  # Guess the title
        abstract = guess_abstract(text)                                                            # Guess the abstract
        sections = extract_sections(text)                                                          # Extract sections

        return {                                                                                   # Create output JSON structure
            "file_name": os.path.basename(pdf_path).replace(".pdf", ".json"),
            "title": title,
            "abstract": abstract,
            "sections": sections
        }
    except Exception as e:                                                                         # Catch and report errors
        print(f"Error processing {pdf_path}: {e}")
        return None


def main():
    """
    Main function to process all PDF files in the input directory and save them as JSON files.
    """
    parser = argparse.ArgumentParser(description="Convert PDFs to structured JSONs with optional OCR")
    parser.add_argument("--input_dir", default="pdfs", help="Directory containing PDF files")
    parser.add_argument("--output_dir", default="jsons", help="Directory to save JSON files")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR if available")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)                                                    # Create output folder if needed
    reader = easyocr.Reader(['en'], gpu=args.gpu)                                                  # Initialize OCR reader

    pdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf")]              # Get list of PDFs
    pdf_files.sort()                                                                               # Optional: consistent order

    for filename in tqdm(pdf_files, desc="Processing PDFs"):                                       # Process each file
        pdf_path = os.path.join(args.input_dir, filename)
        json_data = process_pdf_file(pdf_path, reader)
        if json_data:
            output_path = os.path.join(args.output_dir, filename.replace(".pdf", ".json"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()