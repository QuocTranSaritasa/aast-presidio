"""Pure-PIL drawing helpers for labeling image redaction boxes. No
presidio_image_redactor/opencv dependency, so both the main (spaCy/opencv)
and clinical (torch) venvs can share this."""
from PIL import ImageDraw, ImageFont

from .date_logic import redact_date_text
from .engine import LABELS
from .recognizers import AccidentDateRecognizer

BOX_FILL = (0, 0, 0)
TEXT_FILL = (255, 255, 255)
PADDING = 3


def label_for(entity_type: str, matched_text: str, anchor, ner_hit: bool = False) -> str:
    if ner_hit:
        return f"[[{entity_type}]]"
    if entity_type == AccidentDateRecognizer.ENTITY:
        return redact_date_text(matched_text, anchor)
    return LABELS.get(entity_type, f"[{entity_type}]")


def fit_font(draw: ImageDraw.ImageDraw, label: str, box_height: int, max_width: int):
    size = max(10, min(int(box_height * 0.8), 26))
    while size > 8:
        font = ImageFont.load_default(size=size)
        bbox = draw.textbbox((0, 0), label, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width or size <= 8:
            return font, width, bbox[3] - bbox[1]
        size -= 1
    font = ImageFont.load_default(size=8)
    bbox = draw.textbbox((0, 0), label, font=font)
    return font, bbox[2] - bbox[0], bbox[3] - bbox[1]


def merge_bboxes(bboxes) -> dict:
    """The image analyzer returns one bbox per OCR *word* that falls inside
    a matched entity span, so a multi-word match (e.g. "JANE, Dove") comes
    back as several separate boxes sharing the same (start, end). Merge
    those into a single bounding box so we draw/label each match once."""
    merged: dict = {}
    for result in bboxes:
        key = (result.entity_type, result.start, result.end)
        left, top = result.left, result.top
        right, bottom = left + result.width, top + result.height
        if key not in merged:
            merged[key] = [left, top, right, bottom]
        else:
            box = merged[key]
            box[0] = min(box[0], left)
            box[1] = min(box[1], top)
            box[2] = max(box[2], right)
            box[3] = max(box[3], bottom)
    return merged


def draw_boxes(image, draw: ImageDraw.ImageDraw, merged: dict, ocr_text: str, anchor, ner_hit: bool) -> None:
    image_width = image.size[0]
    for (entity_type, start, end), (left, top, right, bottom) in merged.items():
        matched_text = ocr_text[start:end]
        label = label_for(entity_type, matched_text, anchor, ner_hit=ner_hit)

        box_height = bottom - top
        font, text_width, text_height = fit_font(draw, label, box_height, image_width - left - PADDING * 2)
        box_right = max(right, min(image_width, left + text_width + PADDING * 2))
        box_bottom = max(bottom, top + text_height + PADDING * 2)

        draw.rectangle([left, top, box_right, box_bottom], fill=BOX_FILL)
        draw.text((left + PADDING, top + PADDING), label, font=font, fill=TEXT_FILL)
