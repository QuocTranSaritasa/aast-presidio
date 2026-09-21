#!/usr/bin/env python
"""Evaluate our custom recognizers against obi/deid_roberta_i2b2 - a
transformer fine-tuned on the i2b2 clinical de-identification dataset -
instead of generic spaCy NER. Runs the real production pass
(entities=CUSTOM_ENTITIES) plus this NER model side by side, for both the
text file and the image.

This needs torch/transformers/spacy-huggingface-pipelines, which conflict
with this project's opencv/numpy versions on this platform (opencv-python
needs numpy>=2, this platform's max torch build needs numpy<2). So it runs
from a separate virtualenv - see requirements-clinical.txt - and the image
path uses pytesseract directly (redact.ocr_utils) instead of
presidio-image-redactor, to avoid the opencv dependency entirely.

Usage (from .venv-clinical):
    .venv-clinical/bin/python redact_clinical_eval.py
    .venv-clinical/bin/python redact_clinical_eval.py --score-threshold 0.7
    .venv-clinical/bin/python redact_clinical_eval.py --cpt-codes-file data/cpt_codes.csv
"""
import argparse
from pathlib import Path

from redact.clinical_image_redact import redact_image_file_clinical
from redact.engine import (
    CLINICAL_ENTITIES,
    CLINICAL_SCORE_THRESHOLD,
    build_clinical_analyzer_engine,
    load_cpt_codes,
)
from redact.text_redact import redact_markdown_file

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = DATA_DIR / "redacted"
DEFAULT_CPT_CODES_FILE = DATA_DIR / "cpt_codes.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=CLINICAL_SCORE_THRESHOLD,
        help=(
            "Minimum obi/deid_roberta_i2b2 confidence for a hit to be kept "
            f"(default: {CLINICAL_SCORE_THRESHOLD}, see CLINICAL_SCORE_THRESHOLD "
            "in redact/engine.py). Lower to see more low-confidence gaps, "
            "raise to see only the hits the model is most sure about."
        ),
    )
    parser.add_argument(
        "--cpt-codes-file",
        type=Path,
        default=DEFAULT_CPT_CODES_FILE,
        help=(
            f"CSV file with a 'code' column, used to drop CPT/HCPCS billing "
            f"codes the model confidently mistags as ID (default: "
            f"{DEFAULT_CPT_CODES_FILE}, a placeholder seeded from "
            "data/encounters.md - point this at your real billing-system "
            "export/vendor feed instead). Pass a nonexistent path or an "
            "empty file to disable this filtering."
        ),
    )
    args = parser.parse_args()

    known_codes = load_cpt_codes(args.cpt_codes_file) if args.cpt_codes_file.exists() else set()

    print("Loading obi/deid_roberta_i2b2 (custom recognizers + clinical NER eval)...")
    analyzer = build_clinical_analyzer_engine()

    text_src = DATA_DIR / "encounters.md"
    text_dst = OUT_DIR / "encounters.clinical_eval.md"
    print(f"Redacting {text_src} -> {text_dst} (score_threshold={args.score_threshold})")
    redact_markdown_file(
        text_src,
        text_dst,
        eval_entities=CLINICAL_ENTITIES,
        eval_score_threshold=args.score_threshold,
        eval_known_codes=known_codes,
        analyzer=analyzer,
    )

    image_src = DATA_DIR / "Patient_Intake.jpg"
    image_dst = OUT_DIR / "Patient_Intake.clinical_eval.jpg"
    print(f"Redacting {image_src} -> {image_dst} (score_threshold={args.score_threshold})")
    redact_image_file_clinical(
        image_src,
        image_dst,
        eval_entities=CLINICAL_ENTITIES,
        eval_score_threshold=args.score_threshold,
        eval_known_codes=known_codes,
        analyzer=analyzer,
    )

    print("Done.")


if __name__ == "__main__":
    main()
