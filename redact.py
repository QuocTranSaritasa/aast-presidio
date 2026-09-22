#!/usr/bin/env python
"""Redact PHI from the sample medical records in data/ using Presidio.

Usage:
    python redact.py                 # redact both data/encounters.md and data/Patient_Intake.jpg
    python redact.py --text-only
    python redact.py --image-only
"""
import argparse
from pathlib import Path

from redact.engine import SPACY_ENTITIES
from redact.image_redact import redact_image_file
from redact.recognizers import load_patient_demographics
from redact.text_redact import redact_markdown_file

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = DATA_DIR / "redacted"
DEFAULT_PATIENT_JSON = DATA_DIR / "patient_demographics.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument(
        "--accident-date",
        default=None,
        help=(
            "Date of accident (e.g. 03/15/2023), used as the anchor for "
            "relative date redaction when a document doesn't itself "
            "contain a 'Date of Accident' field for it to be found in. "
            "If omitted, each document's own 'Date of Accident' field is "
            "used as before."
        ),
    )
    parser.add_argument(
        "--spacy-eval",
        action="store_true",
        help=(
            "Also run Presidio's stock SpacyRecognizer (spaCy's generic NER) "
            "and mark anything it caught that our custom recognizers missed "
            "with double brackets, e.g. [[PERSON]], instead of the normal "
            "[PATIENT NAME]-style single-bracket labels."
        ),
    )
    parser.add_argument(
        "--patient-json",
        type=Path,
        default=DEFAULT_PATIENT_JSON,
        help=(
            "JSON file with one patient's known demographic fields (see "
            f"data/patient_demographics.json for the expected shape, "
            f"default: {DEFAULT_PATIENT_JSON}). Each non-empty field is "
            "redacted via naive verbatim search/replace, as a supplement "
            "to the shape/context recognizers - see "
            "redact/recognizers.py:PatientDemographicRecognizer. Pass a "
            "nonexistent path to disable this."
        ),
    )
    args = parser.parse_args()

    do_text = not args.image_only
    do_image = not args.text_only
    eval_entities = SPACY_ENTITIES if args.spacy_eval else None
    patient_demographics = (
        load_patient_demographics(args.patient_json) if args.patient_json.exists() else None
    )

    if do_text:
        src = DATA_DIR / "encounters.md"
        dst = OUT_DIR / ("encounters.spacy_eval.md" if args.spacy_eval else "encounters.redacted.md")
        print(f"Redacting {src} -> {dst}")
        redact_markdown_file(
            src,
            dst,
            accident_date=args.accident_date,
            eval_entities=eval_entities,
            patient_demographics=patient_demographics,
        )

    if do_image:
        src = DATA_DIR / "Patient_Intake.jpg"
        dst = OUT_DIR / ("Patient_Intake.spacy_eval.jpg" if args.spacy_eval else "Patient_Intake.redacted.jpg")
        print(f"Redacting {src} -> {dst}")
        redact_image_file(
            src,
            dst,
            accident_date=args.accident_date,
            eval_entities=eval_entities,
            patient_demographics=patient_demographics,
        )

    print("Done.")


if __name__ == "__main__":
    main()
