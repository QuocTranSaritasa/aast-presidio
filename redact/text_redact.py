from pathlib import Path
from typing import List, Optional

from presidio_anonymizer import AnonymizerEngine

from .engine import AnalyzerEngine, build_analyzer_engine, redact_text


def redact_markdown_file(
    input_path: Path,
    output_path: Path,
    accident_date: Optional[str] = None,
    eval_entities: Optional[List[str]] = None,
    eval_score_threshold: Optional[float] = None,
    analyzer: Optional[AnalyzerEngine] = None,
) -> None:
    text = input_path.read_text(encoding="utf-8")
    analyzer = analyzer or build_analyzer_engine()
    anonymizer = AnonymizerEngine()
    redacted = redact_text(
        text,
        analyzer,
        anonymizer,
        accident_date=accident_date,
        eval_entities=eval_entities,
        eval_score_threshold=eval_score_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(redacted, encoding="utf-8")
