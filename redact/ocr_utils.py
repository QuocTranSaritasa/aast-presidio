"""Minimal, dependency-light OCR + text/bbox mapping helpers.

Mirrors the equivalent logic in presidio_image_redactor.ImageAnalyzerEngine
(remove_space_boxes / get_text_from_ocr_dict / map_analyzer_results_to_
bounding_boxes) but only needs pytesseract - no opencv/numpy. presidio-
image-redactor pins opencv-python>=4.13 (which requires numpy>=2), while
this platform's torch build needs numpy<2 - so the clinical-eval image
path (which needs torch) uses this instead of presidio_image_redactor.
"""
from typing import Dict, List, Tuple

import pytesseract
from presidio_analyzer import RecognizerResult


def perform_ocr(image) -> dict:
    return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)


def remove_space_boxes(ocr_result: dict) -> dict:
    idx = [
        i
        for i, text in enumerate(ocr_result["text"])
        if text != "" and not text.isspace()
    ]
    return {key: [values[i] for i in idx] for key, values in ocr_result.items()}


def get_text_from_ocr_dict(ocr_result: dict, separator: str = " ") -> str:
    if not ocr_result:
        return ""
    return separator.join(ocr_result["text"])


class BBoxResult:
    __slots__ = ("entity_type", "start", "end", "left", "top", "width", "height")

    def __init__(self, entity_type, start, end, left, top, width, height):
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.left = left
        self.top = top
        self.width = width
        self.height = height


def map_results_to_bboxes(
    results: List[RecognizerResult], ocr_result: dict, text: str
) -> List[BBoxResult]:
    """Port of ImageAnalyzerEngine.map_analyzer_results_to_bounding_boxes,
    assuming ocr_result has already been through remove_space_boxes (i.e.
    every word is non-empty, matching how `text` was joined)."""
    if not ocr_result or not results:
        return []

    bboxes = []
    proc_indexes = 0
    total = len(results)

    pos = 0
    iter_ocr = enumerate(ocr_result["text"])
    for index, word in iter_ocr:
        if not word:
            pos += 1
            continue

        for element in results:
            text_element = text[element.start : element.end]
            if max(pos, element.start) < min(element.end, pos + len(word)) and (
                text_element in word or word in text_element
            ):
                bboxes.append(
                    BBoxResult(
                        element.entity_type,
                        element.start,
                        element.end,
                        ocr_result["left"][index],
                        ocr_result["top"][index],
                        ocr_result["width"][index],
                        ocr_result["height"][index],
                    )
                )

                while pos + len(word) < element.end:
                    prev_word = word
                    index, word = next(iter_ocr)
                    if word and not word.isspace():
                        bboxes.append(
                            BBoxResult(
                                element.entity_type,
                                element.start,
                                element.end,
                                ocr_result["left"][index],
                                ocr_result["top"][index],
                                ocr_result["width"][index],
                                ocr_result["height"][index],
                            )
                        )
                    pos += len(prev_word) + 1
                proc_indexes += 1

        if proc_indexes == total:
            break
        pos += len(word) + 1

    return bboxes
