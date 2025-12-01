import base64
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Union
import time

import requests

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
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


class CalendarAgentService:
    """
    A service that processes PDF attachments from emails, extracts information using LLM,
    creates calendar events, and uploads the PDFs to Google Drive.
    """

    def __init__(self, sender_email: str, interval_seconds: int = 300):
        """
        Initialize the Calendar Agent Service.

        Args:
            sender_email: Email address to filter messages from
            interval_seconds: Interval between checks in continuous mode
        """
        self.sender_email = sender_email
        self.interval_seconds = interval_seconds
        self.credentials = self._get_google_credentials()

    def _get_google_credentials(self) -> Credentials:
        """Get Google API credentials."""
        scopes = [
            'https://www.googleapis.com/auth/gmail.modify',
            'https://www.googleapis.com/auth/calendar',
            'https://www.googleapis.com/auth/drive.file'
        ]

        token_path = os.environ.get("GOOGLE_TOKEN_PATH", "token.json")

        # Check if we're in production mode
        is_production = os.environ.get('DYNO') == 'True'

        # If in production mode and environment variables are set, use them
        if is_production and os.environ.get('ACCESS_TOKEN') and os.environ.get('REFRESH_TOKEN'):
            print("Using credentials from environment variables")
            access_token = os.environ.get('ACCESS_TOKEN')
            refresh_token = os.environ.get('REFRESH_TOKEN')
            token_uri = os.environ.get('TOKEN_URI', "https://oauth2.googleapis.com/token")
            client_id = os.environ.get('CLIENT_ID')
            client_secret = os.environ.get('CLIENT_SECRET')

            # Parse SCOPES from environment variable if available, otherwise use default
            try:
                env_scopes = os.environ.get('SCOPES')
                if env_scopes:
                    scopes = json.loads(env_scopes)
            except (json.JSONDecodeError, TypeError):
                # Keep using default scopes if parsing fails
                pass

            # Get expiry from environment variable
            expiry = os.environ.get('EXPIRY')
            # Convert expiry string to datetime object
            if expiry:
                # Remove Z suffix if present for strptime
                if expiry.endswith('Z'):
                    expiry = expiry[:-1]
                # Parse the datetime string
                try:
                    expiry = datetime.strptime(expiry.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    print(f"Warning: Could not parse expiry date '{expiry}'. Using default expiry.")
                    expiry = None

            # Create credentials from tokens
            creds = Credentials(
                token=access_token,
                refresh_token=refresh_token,
                token_uri=token_uri,
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes,
                expiry=expiry
            )

            # Hardcode universe_domain
            creds._universe_domain = "googleapis.com"
            return creds
        elif os.path.exists(token_path):
            # Load existing credentials
            creds = Credentials.from_authorized_user_file(token_path, scopes)
            return creds
        else:
            # Get credentials from environment variable
            credentials_json = os.environ.get("CREDENTIALS")
            if not credentials_json:
                raise ValueError(
                    "CREDENTIALS environment variable not found. "
                    "Set it in your .env file with the Google OAuth client credentials."
                )

            # Create authentication flow and start authorization process
            try:
                from google_auth_oauthlib.flow import InstalledAppFlow
                import tempfile

                # Parse the credentials JSON
                try:
                    credentials_data = json.loads(credentials_json)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON in CREDENTIALS environment variable")

                # Create a temporary file with the credentials
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                    json.dump(credentials_data, temp_file)
                    temp_credentials_path = temp_file.name

                try:
                    # Use the temporary file for authentication
                    flow = InstalledAppFlow.from_client_secrets_file(temp_credentials_path, scopes)

                    # Determine if we're in a production environment
                    if is_production:
                        # In production, use console-based auth flow
                        print("Running in production environment. Using console authentication flow.")
                        creds = flow.run_console()
                    else:
                        # In development, use local server auth flow
                        print("Running in development environment. Using local server authentication flow.")
                        creds = flow.run_local_server(port=0)

                        # Print tokens to console when not in production
                        creds_json = json.loads(creds.to_json())
                        print("\n===== OAUTH TOKENS =====")
                        print(f"ACCESS_TOKEN = {creds_json.get('token')}")
                        print(f"REFRESH_TOKEN = {creds_json.get('refresh_token')}")
                        print("=======================\n")

                    # Save the obtained credentials
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())

                    print(f"Credentials successfully saved to {token_path}")
                    return creds
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_credentials_path):
                        os.unlink(temp_credentials_path)
            except ImportError:
                raise ImportError(
                    "google-auth-oauthlib is required for automatic authentication. "
                    "Install it: pip install google-auth-oauthlib"
                )

    def _fetch_unread_messages(self, service) -> Iterable[dict]:
        """Fetch unread messages with PDF attachments from the specified sender."""
        query = f"from:{self.sender_email} is:unread has:attachment filename:pdf"
        response = service.users().messages().list(userId="me", q=query).execute()
        messages = response.get("messages", [])
        for message in messages:
            full_message = service.users().messages().get(userId="me", id=message["id"]).execute()
            yield full_message

    def _extract_pdf_parts(self, message: dict, service) -> Iterable[tuple[bytes, str]]:
        """Extract PDF attachments from a message."""
        parts: List[dict] = []
        payload = message.get("payload", {})
        if "parts" in payload:
            parts.extend(payload["parts"])
        elif payload.get("body", {}).get("attachmentId"):
            parts.append(payload)

        for idx, part in enumerate(parts):
            filename = part.get("filename") or f"attachment-{message.get('id', 'email')}-{idx}.pdf"
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
            yield base64.urlsafe_b64decode(attachment.get("data", "")), filename

    def _mark_message_as_read(self, message_id: str, service) -> None:
        """Mark a message as read."""
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()

    def _pdf_text_from_bytes(self, data: bytes) -> str:
        """Extract text from PDF bytes."""
        text: List[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    def _parse_with_llm(self, pdf_text: str) -> ParsedEvent:
        """Parse PDF text using LLM (OpenAI)."""
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
            "Always set weather to null as it will be fetched from an external API based on the location."
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": pdf_text}],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        return ParsedEvent.from_response(content)

    def _create_calendar_event(
        self, event: ParsedEvent, pdf_links: Optional[List[str]] = None
    ) -> dict:
        """Create a calendar event from parsed data."""
        calendar = build("calendar", "v3", credentials=self.credentials)
        start = event.start or datetime.now(timezone.utc).isoformat()
        end = event.end or datetime.now(timezone.utc).isoformat()
        body = {
            "summary": event.name,
            "description": self._build_description(event, pdf_links=pdf_links),
            "start": {"dateTime": start, "timeZone": "UTC"},
            "end": {"dateTime": end, "timeZone": "UTC"},
            "location": event.address or event.location_url,
        }
        return calendar.events().insert(calendarId="primary", body=body).execute()

    def _build_description(self, event: ParsedEvent, pdf_links: Optional[List[str]] = None) -> str:
        """Build event description with notes, PDF links, and weather."""
        lines = []
        note_body = event.notes or self._build_structured_note(event)
        if note_body:
            lines.append(note_body)

        if pdf_links:
            lines.append("📄 PDF attachments")
            for link in pdf_links:
                lines.append(f"• {link}")

        # Always get weather from API based on location
        weather_text = self._build_weather_snippet(event)

        if weather_text:
            lines.append(f"Weather: {weather_text}")
        return "\n\n".join(lines)

    def _build_structured_note(self, event: ParsedEvent) -> str:
        """Build structured note from event data."""
        parts: List[str] = []

        header_parts = [event.name] if event.name else []
        if event.address:
            header_parts.append(event.address)
        if header_parts:
            parts.append(" — ".join(header_parts))

        schedule_text = self._format_schedule(event.schedule) if event.schedule else None
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

    def _format_schedule(self, schedule: Optional[object]) -> str:
        """Format schedule data as text."""
        if isinstance(schedule, list):
            lines = []
            for item in schedule:
                activity = item.get("activity", "")
                start_time = self._format_time(item.get("start_time"))
                end_time = self._format_time(item.get("end_time"))
                if activity or start_time or end_time:
                    lines.append(f"• {activity}: {start_time}–{end_time}")
            return "\n".join(lines)
        return str(schedule)

    def _format_time(self, value: Optional[str]) -> str:
        """Format time value."""
        if not value:
            return ""
        try:
            return datetime.fromisoformat(value).strftime("%H:%M")
        except ValueError:
            return value

    def _format_weather(self, weather: dict) -> str:
        """Format weather data as text."""
        temperature = weather.get("temperature_c") or weather.get("temperature")
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

    def _build_weather_snippet(self, event: ParsedEvent) -> Optional[str]:
        """Build weather snippet for event location and time."""
        if not event.start or not (event.address or event.location_url):
            return None

        try:
            dt = datetime.fromisoformat(event.start)
        except ValueError:
            return None

        query = event.address or ""
        if event.location_url and "maps" in event.location_url:
            query = event.location_url

        coords = self._geocode_location(query)
        if not coords:
            return None

        forecast = self._fetch_weather(coords["lat"], coords["lon"], dt.date())
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

    def _geocode_location(self, query: str) -> Optional[dict]:
        """Geocode location query to coordinates."""
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

    def _fetch_weather(self, lat: float, lon: float, date) -> Optional[dict]:
        """Fetch weather forecast for coordinates and date."""
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

    def _upload_pdf_to_drive(self, data: bytes, filename: str) -> Optional[str]:
        """Upload PDF to Google Drive and return public link."""
        drive = build("drive", "v3", credentials=self.credentials)
        file_metadata = {"name": filename}
        folder_id = os.environ.get("DRIVE_FOLDER_ID")
        if folder_id:
            file_metadata["parents"] = [folder_id]

        media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/pdf", resumable=True)
        created = (
            drive.files()
            .create(body=file_metadata, media_body=media, fields="id, webViewLink")
            .execute()
        )
        file_id = created.get("id")
        if not file_id:
            return None

        drive.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            fields="id",
        ).execute()

        if created.get("webViewLink"):
            return created["webViewLink"]

        link_info = drive.files().get(fileId=file_id, fields="webViewLink, webContentLink").execute()
        return link_info.get("webViewLink") or link_info.get("webContentLink")

    def process_messages(self) -> int:
        """
        Process unread messages from the specified sender.

        Returns:
            Number of events created
        """
        gmail_service = build("gmail", "v1", credentials=self.credentials)
        created_events = 0
        for message in self._fetch_unread_messages(gmail_service):
            for attachment, filename in self._extract_pdf_parts(message, gmail_service):
                pdf_text = self._pdf_text_from_bytes(attachment)
                parsed = self._parse_with_llm(pdf_text)
                pdf_link = self._upload_pdf_to_drive(attachment, filename)
                links = [pdf_link] if pdf_link else None
                created = self._create_calendar_event(parsed, pdf_links=links)
                print(f"Created event {created.get('id')} for {parsed.name}")
                created_events += 1
            self._mark_message_as_read(message.get("id"), gmail_service)
        return created_events

    def run_once(self) -> int:
        """
        Run the service once to process all current unread messages.

        Returns:
            Number of events created
        """
        print(f"Processing messages from {self.sender_email}...")
        created_events = self.process_messages()
        if created_events == 0:
            print("No new matching emails found.")
        else:
            print(f"Created {created_events} events.")
        return created_events

    def run_continuously(self) -> None:
        """Run the service in a continuous loop with the specified interval."""
        print(f"Starting continuous mode. Checking for messages from {self.sender_email} every {self.interval_seconds} seconds.")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                created_events = self.process_messages()
                if created_events == 0:
                    print(f"No new matching emails. Sleeping for {self.interval_seconds} seconds...")
                else:
                    print(f"Created {created_events} events. Sleeping for {self.interval_seconds} seconds...")
                time.sleep(self.interval_seconds)
        except KeyboardInterrupt:
            print("\nService stopped by user.")
