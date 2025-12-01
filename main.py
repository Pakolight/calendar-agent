import base64
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

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
    schedule: Optional[str]
    address: Optional[str]

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
        "Return a strict JSON object with keys: name, start_time, end_time, notes, schedule, address. "
        "Translate all text to English. Provide ISO 8601 timestamps for start_time and end_time when possible. "
        "If a value is missing, use null."
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
        "location": event.address,
    }
    return calendar.events().insert(calendarId="primary", body=body).execute()


def build_description(event: ParsedEvent) -> str:
    lines = []
    if event.notes:
        lines.append(f"Notes: {event.notes}")
    if event.schedule:
        lines.append(f"Schedule: {event.schedule}")
    if event.address:
        lines.append(f"Address: {event.address}")
    return "\n".join(lines)


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
