"""OCR utilities for parsing planning board images."""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List

import cv2  # type: ignore
import numpy as np
import pytesseract
from pytesseract import Output


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess the image for OCR."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    deskewed = deskew_image(thresh)
    return deskewed


def deskew_image(binary_image: np.ndarray) -> np.ndarray:
    """Deskew the binary image using its minimum area rectangle angle."""
    inverted = 255 - binary_image
    coords = cv2.findNonZero(inverted)
    if coords is None:
        return binary_image

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    else:
        angle = angle

    (h, w) = binary_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _iter_ocr_lines(data: Dict[str, List]) -> Iterable[Tuple[str, List[int]]]:
    """Yield concatenated lines and their bounding boxes from Tesseract data."""
    keys = zip(
        data["page_num"],
        data["block_num"],
        data["par_num"],
        data["line_num"],
        data["text"],
        data["conf"],
        data["left"],
        data["top"],
        data["width"],
        data["height"],
    )
    lines: Dict[Tuple[int, int, int, int], List[Tuple[str, int, int, int, int]]]
    lines = {}
    for page, block, paragraph, line_num, text, conf, left, top, width, height in keys:
        if int(conf) < 0 or not text.strip():
            continue
        key = (page, block, paragraph, line_num)
        lines.setdefault(key, []).append((text, left, top, width, height))

    for parts in lines.values():
        parts.sort(key=lambda item: item[1])
        combined_text = " ".join(p[0] for p in parts).strip()
        bbox = [
            min(p[1] for p in parts),
            min(p[2] for p in parts),
            max(p[1] + p[3] for p in parts),
            max(p[2] + p[4] for p in parts),
        ]
        yield combined_text, bbox


def _parse_line_to_event(line: str) -> Dict[str, object]:
    """Parse a single OCR line into structured event fields."""
    cleaned = re.sub(r"\s+", " ", line).strip()
    tokens = [token.strip() for token in re.split(r"\s{2,}|\||\t", cleaned) if token.strip()]
    if not tokens:
        tokens = cleaned.split(" ")

    date_token = next((t for t in tokens if _looks_like_date(t)), "")
    technicians_raw = ""
    if tokens:
        technicians_raw = tokens[-1] if len(tokens) > 3 else ""

    project = ""
    location = ""
    notes = ""

    if date_token and date_token in tokens:
        date_index = tokens.index(date_token)
        remaining = tokens[date_index + 1 :]
    else:
        remaining = tokens

    if remaining:
        project = remaining[0]
    if len(remaining) > 1:
        location = remaining[1]
    if len(remaining) > 2:
        notes = " ".join(remaining[2:])

    technicians_list = _split_technicians(technicians_raw)

    return {
        "raw_text": cleaned,
        "date_raw": date_token,
        "project": project,
        "location": location,
        "notes": notes,
        "technicians_raw": technicians_raw,
        "technicians_list": technicians_list,
    }


def _looks_like_date(token: str) -> bool:
    """Return True if token resembles a date format."""
    date_patterns = [
        r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b",
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b",
    ]
    return any(re.search(pattern, token, re.IGNORECASE) for pattern in date_patterns)


def _split_technicians(raw: str) -> List[str]:
    """Split technicians string into individual names."""
    if not raw:
        return []
    return [part.strip() for part in re.split(r",|;|&|/", raw) if part.strip()]


def process_board_image(image_path: str) -> List[Dict[str, object]]:
    """Process a planning board image and return structured events."""
    processed = preprocess_image(image_path)
    ocr_data = pytesseract.image_to_data(processed, output_type=Output.DICT, config="--psm 6")

    lines = list(_iter_ocr_lines(ocr_data))
    events: List[Dict[str, object]] = []

    for line_text, _bbox in lines:
        if not line_text.strip():
            continue
        event = _parse_line_to_event(line_text)
        if any(event.values()):
            events.append(event)

    return events


def format_events_text(events: List[Dict[str, object]]) -> str:
    """Format events as a human-readable multi-line string."""
    formatted_lines = []
    for idx, event in enumerate(events, start=1):
        technicians = ", ".join(event.get("technicians_list", [])) or event.get("technicians_raw", "")
        formatted_lines.append(
            "\n".join(
                [
                    f"Event {idx}:",
                    f"  Date: {event.get('date_raw', '')}",
                    f"  Project: {event.get('project', '')}",
                    f"  Location: {event.get('location', '')}",
                    f"  Notes: {event.get('notes', '')}",
                    f"  Technicians: {technicians}",
                ]
            )
        )
    return "\n\n".join(formatted_lines)


def events_to_json(events: List[Dict[str, object]]) -> str:
    """Serialize events to formatted JSON."""
    return json.dumps(events, ensure_ascii=False, indent=2)

