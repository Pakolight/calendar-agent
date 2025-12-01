import base64
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Union

import requests

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI
import pdfplumber

# Load environment variables from .env file
load_dotenv()


@dataclass
class ParsedEvent:
    name: str
    start: Optional[str]
    end: Optional[str]
    notes: Optional[str]
    schedule: Optional[Union[str, List[dict]]]
    address: Optional[str]
    location_url: Optional[str]
    weather: Optional[dict]

    @classmethod
    def from_response(cls, payload: str) -> "ParsedEvent":
        data = json.loads(payload)
        return cls(
            name=data.get("name", ""),
            start=data.get("start_time"),
            end=data.get("end_time"),
            notes=data.get("notes"),
            schedule=data.get("schedule"),
            address=data.get("address"),
            location_url=data.get("location_url"),
            weather=data.get("weather"),
        )


def get_google_credentials() -> Credentials:
    scopes = [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/calendar",
    ]
    token_path = os.environ.get("GOOGLE_TOKEN_PATH", "token.json")
    credentials_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "credentials.json")

    if os.path.exists(token_path):
        # Загрузить существующие учетные данные
        creds = Credentials.from_authorized_user_file(token_path, scopes)
        return creds
    else:
        # Проверяем наличие файла credentials.json
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Файл учетных данных {credentials_path} не найден. "
                f"Загрузите его из Google Cloud Console."
            )

        # Создаем поток аутентификации и запускаем процесс авторизации
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)

            # Сохраняем полученные учетные данные
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

            print(f"Учетные данные успешно сохранены в {token_path}")
            return creds
        except ImportError:
            raise ImportError(
                "Для автоматической аутентификации требуется google-auth-oauthlib. "
                "Установите ее: pip install google-auth-oauthlib"
            )


def fetch_unread_messages(sender: str, service) -> Iterable[dict]:
    query = f"from:{sender} is:unread has:attachment filename:pdf"
    response = service.users().messages().list(userId="me", q=query).execute()
    messages = response.get("messages", [])
    for message in messages:
        full_message = service.users().messages().get(userId="me", id=message["id"]).execute()
        yield full_message


def extract_pdf_parts(message: dict, service) -> Iterable[bytes]:
    parts: List[dict] = []
    payload = message.get("payload", {})
    if "parts" in payload:
        parts.extend(payload["parts"])
    elif payload.get("body", {}).get("attachmentId"):
        parts.append(payload)

    for part in parts:
        filename = part.get("filename", "")
        if not filename.lower().endswith(".pdf"):
            continue

        body = part.get("body", {})
        attachment_id = body.get("attachmentId")
        if not attachment_id:
            continue

        attachment = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message["id"], id=attachment_id)
            .execute()
        )
        yield base64.urlsafe_b64decode(attachment.get("data", ""))


def mark_message_as_read(message_id: str, service) -> None:
    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={"removeLabelIds": ["UNREAD"]},
    ).execute()


def pdf_text_from_bytes(data: bytes) -> str:
    text: List[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def parse_with_llm(pdf_text: str) -> ParsedEvent:
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    prompt = (
        "You read text extracted from a PDF invitation or brief. "
        "Return a strict JSON object with keys: name, start_time, end_time, notes, schedule, address, location_url, weather. "
        "Translate all text to English. Provide ISO 8601 timestamps for start_time and end_time when possible. "
        "If a value is missing, use null. "
        "Extract any Google Maps link as location_url and derive the textual venue address for the address field. "
        "Compose notes as a human-readable, emoji-labeled set of sections using 24-hour times and bullet lists. "
        "Start notes with '{name} — {address}' (omit missing pieces), then add:\n"
        "🕒 Schedule with bullet items '• Activity: HH:MM–HH:MM';\n"
        "📍 Address with the full venue address on its own line;\n"
        "(if present) the Google Maps link on the next bullet;\n"
        "⸻ (em dash rule) separators between major sections;\n"
        "📝 Comments for general notes and staffing details;\n"
        "✅ Tasks for actionable to-dos. "
        "Keep weather null so it can be filled later."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": pdf_text}],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    return ParsedEvent.from_response(content)


def create_calendar_event(event: ParsedEvent, credentials: Credentials) -> dict:
    calendar = build("calendar", "v3", credentials=credentials)
    start = event.start or datetime.now(timezone.utc).isoformat()
    end = event.end or datetime.now(timezone.utc).isoformat()
    body = {
        "summary": event.name,
        "description": build_description(event),
        "start": {"dateTime": start},
        "end": {"dateTime": end},
        "location": event.address or event.location_url,
    }
    return calendar.events().insert(calendarId="primary", body=body).execute()


def build_description(event: ParsedEvent) -> str:
    lines = []
    note_body = event.notes or build_structured_note(event)
    if note_body:
        lines.append(note_body)

    weather_text: Optional[str] = None
    if event.weather:
        formatted_weather = (
            format_weather(event.weather)
            if isinstance(event.weather, dict)
            else str(event.weather)
        )
        weather_text = formatted_weather
    else:
        weather_text = build_weather_snippet(event)

    if weather_text:
        lines.append(f"Weather: {weather_text}")
    return "\n\n".join(lines)


def build_structured_note(event: ParsedEvent) -> str:
    parts: List[str] = []

    header_parts = [event.name] if event.name else []
    if event.address:
        header_parts.append(event.address)
    if header_parts:
        parts.append(" — ".join(header_parts))

    schedule_text = format_schedule(event.schedule) if event.schedule else None
    if schedule_text:
        parts.append("🕒 Schedule")
        parts.append(schedule_text)

    if schedule_text and (event.address or event.location_url):
        parts.append("⸻")

    if event.address or event.location_url:
        parts.append("📍 Address")
        if event.address:
            parts.append(event.address)
        if event.location_url:
            parts.append(f"• Map: {event.location_url}")

    return "\n\n".join(parts)


def format_schedule(schedule: Optional[object]) -> str:
    if isinstance(schedule, list):
        lines = []
        for item in schedule:
            activity = item.get("activity", "")
            start_time = format_time(item.get("start_time"))
            end_time = format_time(item.get("end_time"))
            if activity or start_time or end_time:
                lines.append(f"• {activity}: {start_time}–{end_time}")
        return "\n".join(lines)
    return str(schedule)


def format_time(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime("%H:%M")
    except ValueError:
        return value


def format_weather(weather: dict) -> str:
    temperature = weather.get("temperature_c")
    wind_speed = weather.get("wind_speed_kts")
    precipitation = weather.get("precipitation")
    parts = []
    if temperature is not None:
        parts.append(f"Temp: {temperature}°C")
    if wind_speed is not None:
        parts.append(f"Wind: {wind_speed} kts")
    if precipitation is not None:
        parts.append(f"Precipitation: {precipitation}")
    return ", ".join(parts)


def build_weather_snippet(event: ParsedEvent) -> Optional[str]:
    if not event.start or not (event.address or event.location_url):
        return None

    try:
        dt = datetime.fromisoformat(event.start)
    except ValueError:
        return None

    query = event.address or ""
    if event.location_url and "maps" in event.location_url:
        query = event.location_url

    coords = geocode_location(query)
    if not coords:
        return None

    forecast = fetch_weather(coords["lat"], coords["lon"], dt.date())
    if not forecast:
        return None

    parts = []
    if forecast.get("temperature") is not None:
        parts.append(f"temp {forecast['temperature']}°C")
    if forecast.get("wind_speed_kts") is not None:
        parts.append(f"wind {forecast['wind_speed_kts']} kts")
    if forecast.get("precipitation") is not None:
        parts.append(f"precip {forecast['precipitation']} mm")
    return ", ".join(parts)


def geocode_location(query: str) -> Optional[dict]:
    if not query:
        return None
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search", params={"name": query, "count": 1}
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    return {"lat": first.get("latitude"), "lon": first.get("longitude")}


def fetch_weather(lat: float, lon: float, date) -> Optional[dict]:
    if lat is None or lon is None:
        return None
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "UTC",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    if resp.status_code != 200:
        return None

    payload = resp.json()
    hourly = payload.get("hourly", {})
    timestamps = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precips = hourly.get("precipitation", [])
    winds = hourly.get("wind_speed_10m", [])
    if not timestamps:
        return None

    target_indices = [i for i, t in enumerate(timestamps) if t.startswith(date.isoformat())]
    if not target_indices:
        return None

    idx = target_indices[len(target_indices) // 2]
    celsius = temps[idx] if idx < len(temps) else None
    precip = precips[idx] if idx < len(precips) else None
    wind_m_s = winds[idx] if idx < len(winds) else None
    wind_kts = round(wind_m_s * 1.94384, 1) if wind_m_s is not None else None
    return {"temperature": celsius, "precipitation": precip, "wind_speed_kts": wind_kts}


def format_schedule(schedule: Optional[object]) -> str:
    if isinstance(schedule, list):
        lines = []
        for item in schedule:
            activity = item.get("activity", "")
            start_time = item.get("start_time", "")
            end_time = item.get("end_time", "")
            lines.append(f"- {activity}: {start_time} - {end_time}")
        return "\n".join(lines)
    return str(schedule)


def format_weather(weather: dict) -> str:
    temperature = weather.get("temperature_c")
    wind_speed = weather.get("wind_speed_kts")
    precipitation = weather.get("precipitation")
    parts = []
    if temperature is not None:
        parts.append(f"Temp: {temperature}°C")
    if wind_speed is not None:
        parts.append(f"Wind: {wind_speed} kts")
    if precipitation is not None:
        parts.append(f"Precipitation: {precipitation}")
    return ", ".join(parts)


def build_weather_snippet(event: ParsedEvent) -> Optional[str]:
    if not event.start or not (event.address or event.location_url):
        return None

    try:
        dt = datetime.fromisoformat(event.start)
    except ValueError:
        return None

    query = event.address or ""
    if event.location_url and "maps" in event.location_url:
        query = event.location_url

    coords = geocode_location(query)
    if not coords:
        return None

    forecast = fetch_weather(coords["lat"], coords["lon"], dt.date())
    if not forecast:
        return None

    parts = []
    if forecast.get("temperature") is not None:
        parts.append(f"temp {forecast['temperature']}°C")
    if forecast.get("wind_speed_kts") is not None:
        parts.append(f"wind {forecast['wind_speed_kts']} kts")
    if forecast.get("precipitation") is not None:
        parts.append(f"precip {forecast['precipitation']} mm")
    return ", ".join(parts)


def geocode_location(query: str) -> Optional[dict]:
    if not query:
        return None
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search", params={"name": query, "count": 1}
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    return {"lat": first.get("latitude"), "lon": first.get("longitude")}


def fetch_weather(lat: float, lon: float, date) -> Optional[dict]:
    if lat is None or lon is None:
        return None
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "UTC",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    if resp.status_code != 200:
        return None

    payload = resp.json()
    hourly = payload.get("hourly", {})
    timestamps = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precips = hourly.get("precipitation", [])
    winds = hourly.get("wind_speed_10m", [])
    if not timestamps:
        return None

    target_indices = [i for i, t in enumerate(timestamps) if t.startswith(date.isoformat())]
    if not target_indices:
        return None

    idx = target_indices[len(target_indices) // 2]
    celsius = temps[idx] if idx < len(temps) else None
    precip = precips[idx] if idx < len(precips) else None
    wind_m_s = winds[idx] if idx < len(winds) else None
    wind_kts = round(wind_m_s * 1.94384, 1) if wind_m_s is not None else None
    return {"temperature": celsius, "precipitation": precip, "wind_speed_kts": wind_kts}


def process_messages(sender_email: str, credentials: Credentials) -> int:
    gmail_service = build("gmail", "v1", credentials=credentials)
    created_events = 0
    for message in fetch_unread_messages(sender_email, gmail_service):
        for attachment in extract_pdf_parts(message, gmail_service):
            pdf_text = pdf_text_from_bytes(attachment)
            parsed = parse_with_llm(pdf_text)
            created = create_calendar_event(parsed, credentials)
            print(f"Created event {created.get('id')} for {parsed.name}")
            created_events += 1
        mark_message_as_read(message.get("id"), gmail_service)
    return created_events


def run_bot(sender_email: str, interval_seconds: int) -> None:
    credentials = get_google_credentials()
    while True:
        created_events = process_messages(sender_email, credentials)
        if created_events == 0:
            print("No new matching emails. Sleeping...")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    sender = os.environ.get("SENDER_EMAIL", "")
    if not sender:
        raise SystemExit("SENDER_EMAIL env variable is required")

    interval = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))
    run_bot(sender, interval)
