"""Image redaction for the clinical-transformer eval path. Uses pytesseract
directly (via redact.ocr_utils) instead of presidio_image_redactor, since
presidio-image-redactor pins an opencv-python version that needs numpy>=2,
which conflicts with the numpy<2 this platform's torch build needs. Only
usable from .venv-clinical (needs torch/transformers - see
build_clinical_analyzer_engine)."""
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw

from .date_logic import find_accident_anchor
from .engine import AnalyzerEngine, analyze_with_eval
from .image_draw import draw_boxes, merge_bboxes
from .ocr_utils import get_text_from_ocr_dict, map_results_to_bboxes, perform_ocr, remove_space_boxes


def redact_image_file_clinical(
    input_path: Path,
    output_path: Path,
    eval_entities: List[str],
    analyzer: AnalyzerEngine,
    eval_score_threshold: Optional[float] = None,
) -> None:
    image = Image.open(input_path).convert("RGB")

    ocr_result = remove_space_boxes(perform_ocr(image))
    ocr_text = get_text_from_ocr_dict(ocr_result)
    anchor = find_accident_anchor(ocr_text)

    custom_results, ner_only_results = analyze_with_eval(
        ocr_text, analyzer, eval_entities, score_threshold=eval_score_threshold
    )

    custom_bboxes = map_results_to_bboxes(custom_results, ocr_result, ocr_text)
    ner_bboxes = map_results_to_bboxes(ner_only_results, ocr_result, ocr_text)

    draw = ImageDraw.Draw(image)
    draw_boxes(image, draw, merge_bboxes(custom_bboxes), ocr_text, anchor, ner_hit=False)
    draw_boxes(image, draw, merge_bboxes(ner_bboxes), ocr_text, anchor, ner_hit=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
