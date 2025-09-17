#!/usr/bin/env python3

# (c) by Shajib Ghosh @10:42 AM, 2025-09-17

import argparse
import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from PyPDF2 import PdfReader
from tqdm import tqdm

# -----------------------------
# Topic definitions (edit to fit your corpus)
# -----------------------------
TOPICS = {
    "CT_microCT": [
        r"\bcomputed tomography\b", r"\bmicro[-\s]?ct\b", r"\bcone[-\s]?beam\b",
        r"\bcbct\b", r"\btomograph(y|ic)\b"
    ],
    "Radiography_NDT": [
        r"\bradiograph(y|ic)\b", r"\bdigital radiography\b", r"\bndt\b",
        r"\bnon[-\s]?destructive testing\b", r"\bnondestructive testing\b",
        r"\bindustrial x[-\s]?ray\b", r"\bx[-\s]?ray inspection\b", r"\bx[-\s]?ray imaging\b"
    ],
    "Semiconductor_PCB": [
        r"\bsolder joint\b", r"\bvoid(ing)?\b", r"\bbga\b", r"\bball grid array\b",
        r"\bpcb\b", r"\belectronics assembly\b", r"\bpackage inspection\b"
    ],
    "Security_Baggage": [
        r"\bsecurity screening\b", r"\bbaggage\b", r"\bcarry[-\s]?on\b", r"\bluggage\b",
        r"\bthreat detection\b", r"\bdual[-\s]?energy\b"
    ],
    "Medical": [
        r"\bdiagnostic\b", r"\bradiology\b", r"\bmammograph(y|ic)\b",
        r"\bfluoroscopy\b", r"\binterventional\b"
    ],
    "Materials_Additive": [
        r"\bweld(ing|s)?\b", r"\bcasting\b", r"\bporosity\b", r"\bmetallurg(y|ical)\b",
        r"\bcomposite(s)?\b", r"\badditive manufacturing\b", r"\b3d printing\b",
        r"\bpowder bed fusion\b", r"\blpbf\b"
    ],
    "Food_Agri": [
        r"\bfood\b", r"\bforeign object detection\b", r"\bcontamination\b",
        r"\bagricultur(al|e)\b"
    ],
    "XRD_Crystallography": [
        r"\bx[-\s]?ray diffraction\b", r"\bxrd\b", r"\bdiffraction\b"
    ],
    "XRF_Spectroscopy": [
        r"\bx[-\s]?ray fluorescence\b", r"\bxrf\b", r"\bfluorescence\b"
    ],
}

# Gate terms to ensure it's actually X-ray related
XRAY_GUARD = [
    r"\bx[-\s\u2011\u2013]?ray(s)?\b",      # x-ray, x–ray, x ray
    r"\bradiograph(y|ic)\b",
    r"\bcomputed tomography\b", r"\bct\b", r"\bmicro[-\s]?ct\b"
]

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def matches_any(text: str, patterns: List[str]) -> List[str]:
    hits = []
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(pat)
    return hits

def classify(text: str) -> Tuple[str, List[str], int]:
    """Return (topic, matched_patterns, score)."""
    guard_hits = matches_any(text, XRAY_GUARD)
    if not guard_hits:
        return "Uncategorized", [], 0

    best_topic = "Xray_General"
    best_hits = guard_hits[:]
    best_score = len(guard_hits)

    for topic, patterns in TOPICS.items():
        thits = matches_any(text, patterns)
        score = len(thits) + len(guard_hits)
        if score > best_score:
            best_topic = topic
            best_hits = guard_hits + thits
            best_score = score

    return best_topic, best_hits, best_score

def read_pdf_text(path: Path, max_pages: int, show_page_bar: bool, bar_position: int = 1) -> str:
    """Extract text from the first max_pages using PyPDF2, with optional per-PDF page bar."""
    chunks = []
    try:
        reader = PdfReader(str(path))
        n_pages = len(reader.pages)
        pages_to_read = min(max_pages, n_pages if n_pages else 0)
        page_iter = range(pages_to_read)
        if show_page_bar and pages_to_read > 0:
            page_iter = tqdm(
                page_iter,
                total=pages_to_read,
                desc=f"Reading pages: {path.name}",
                unit="pg",
                leave=False,
                position=bar_position
            )
        for i in page_iter:
            try:
                chunks.append(reader.pages[i].extract_text() or "")
            except Exception:
                # unreadable page — skip
                pass
    except Exception:
        # encrypted or bad file — return empty
        return ""
    return "\n".join(chunks)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Sort PDFs by X-ray inspection topics with progress bars.")
    ap.add_argument("--src", required=True, help="Source folder containing PDFs")
    ap.add_argument("--dst", required=True, help="Destination root folder")
    ap.add_argument("--action", choices=["copy", "move"], default="copy",
                    help="Copy or move matched PDFs into topic folders (default: copy)")
    ap.add_argument("--max-pages", type=int, default=6,
                    help="Number of pages to scan from each PDF (default: 6)")
    ap.add_argument("--threshold", type=int, default=1,
                    help="Minimum total matches (guard+topic) to classify (default: 1)")
    ap.add_argument("--no-page-bar", action="store_true",
                    help="Disable the inner page-level progress bar")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    ensure_dir(dst)

    report_rows = []
    counts = defaultdict(int)

    pdfs = sorted([p for p in src.rglob("*.pdf")])
    if not pdfs:
        print(f"No PDFs found in: {src}")
        return

    # Top-level progress bar over files
    for pdf in tqdm(pdfs, desc="Processing PDFs", unit="file", position=0):
        text = read_pdf_text(
            pdf,
            args.max_pages,
            show_page_bar=not args.no_page_bar,
            bar_position=1
        )

        topic, hits, score = classify(text)

        if score < args.threshold:
            topic = "Uncategorized"
            hits = []
            score = 0

        out_dir = dst / topic
        ensure_dir(out_dir)
        target = out_dir / pdf.name

        try:
            if args.action == "copy":
                shutil.copy2(pdf, target)
            else:
                shutil.move(pdf, target)
        except Exception as e:
            tqdm.write(f"[WARN] Failed to {args.action} '{pdf.name}': {e}")
            continue

        counts[topic] += 1
        report_rows.append({
            "file": str(pdf),
            "topic": topic,
            "score": score,
            "matched_terms": "; ".join(hits),
            "dest": str(target)
        })

    # Write CSV report
    report_path = dst / "sort_report.csv"
    try:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "topic", "score", "matched_terms", "dest"])
            writer.writeheader()
            writer.writerows(report_rows)
    except Exception as e:
        tqdm.write(f"[WARN] Could not write report: {e}")

    # Summary
    print(f"\nProcessed {len(pdfs)} PDFs from: {src}")
    for topic, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {topic}: {n}")
    print(f"\nReport: {report_path}")
    print(f"Sorted files under: {dst}")

if __name__ == "__main__":
    main()