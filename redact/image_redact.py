from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw
from presidio_image_redactor import ImageAnalyzerEngine

from .date_logic import find_accident_anchor, parse_date_text
from .engine import CUSTOM_ENTITIES, AnalyzerEngine, analyze_with_eval, build_analyzer_engine
from .image_draw import draw_boxes, merge_bboxes


def redact_image_file(
    input_path: Path,
    output_path: Path,
    accident_date: Optional[str] = None,
    eval_entities: Optional[List[str]] = None,
    analyzer: Optional[AnalyzerEngine] = None,
) -> None:
    image = Image.open(input_path).convert("RGB")

    analyzer = analyzer or build_analyzer_engine()
    image_analyzer = ImageAnalyzerEngine(analyzer_engine=analyzer)

    ocr_result = image_analyzer.ocr.perform_ocr(image)
    ocr_result = image_analyzer.remove_space_boxes(ocr_result)
    ocr_text = image_analyzer.ocr.get_text_from_ocr_dict(ocr_result)
    # Same override as the text path: use the supplied accident date as the
    # anchor when this document doesn't carry its own "Date of Accident"
    # field for OCR to find.
    anchor = parse_date_text(accident_date) if accident_date else find_accident_anchor(ocr_text)

    if eval_entities:
        custom_results, ner_only_results = analyze_with_eval(ocr_text, analyzer, eval_entities)
    else:
        custom_results = analyzer.analyze(text=ocr_text, language="en", entities=CUSTOM_ENTITIES)
        ner_only_results = []

    # Reuse the single OCR pass above to map both result sets to bounding
    # boxes, rather than letting analyze() redo OCR once per call.
    custom_bboxes = image_analyzer.map_analyzer_results_to_bounding_boxes(
        custom_results, ocr_result, ocr_text, allow_list=[]
    )
    ner_bboxes = image_analyzer.map_analyzer_results_to_bounding_boxes(
        ner_only_results, ocr_result, ocr_text, allow_list=[]
    )

    draw = ImageDraw.Draw(image)
    draw_boxes(image, draw, merge_bboxes(custom_bboxes), ocr_text, anchor, ner_hit=False)
    draw_boxes(image, draw, merge_bboxes(ner_bboxes), ocr_text, anchor, ner_hit=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
