"""OCR utilities for parsing planning board images using Google Vision API."""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List

from google.cloud import vision


def _extract_lines(annotation: vision.TextAnnotation) -> Iterable[str]:
    """Yield text lines from a Vision TextAnnotation preserving order."""
    if not annotation or not annotation.text:
        return []

    return (line.strip() for line in annotation.text.splitlines() if line.strip())


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
    """Process a planning board image with Google Vision and return structured events."""
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    response = client.document_text_detection(image=vision.Image(content=content))
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    annotation = response.full_text_annotation
    events: List[Dict[str, object]] = []

    for line in _extract_lines(annotation):
        event = _parse_line_to_event(line)
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
