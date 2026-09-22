from pathlib import Path
from typing import Dict, List, Optional, Set

from presidio_anonymizer import AnonymizerEngine

from .engine import AnalyzerEngine, build_analyzer_engine, redact_text


def redact_markdown_file(
    input_path: Path,
    output_path: Path,
    accident_date: Optional[str] = None,
    eval_entities: Optional[List[str]] = None,
    eval_score_threshold: Optional[float] = None,
    eval_known_codes: Optional[Set[str]] = None,
    analyzer: Optional[AnalyzerEngine] = None,
    patient_demographics: Optional[Dict[str, str]] = None,
) -> None:
    """patient_demographics is only applied when this function builds its
    own analyzer (i.e. `analyzer` is not supplied) - a caller passing in a
    pre-built analyzer (e.g. redact_clinical_eval.py) must bake the
    demographics into that analyzer itself via
    build_clinical_analyzer_engine(patient_demographics=...)."""
    text = input_path.read_text(encoding="utf-8")
    analyzer = analyzer or build_analyzer_engine(patient_demographics=patient_demographics)
    anonymizer = AnonymizerEngine()
    redacted = redact_text(
        text,
        analyzer,
        anonymizer,
        accident_date=accident_date,
        eval_entities=eval_entities,
        eval_score_threshold=eval_score_threshold,
        eval_known_codes=eval_known_codes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(redacted, encoding="utf-8")
