# Calendar Agent

This service automates processing unread Gmail messages with PDF attachments from a specified sender, extracts structured event details using an LLM, uploads the PDF to Google Drive, and creates Google Calendar events that include a shareable file link. It can be run either as a long-lived service that polls Gmail on a schedule or as a one-time execution for manual processing.

## Features
- Queries Gmail for unread messages from a configured sender that include PDF attachments.
- Uploads the PDF attachments to Google Drive (optionally into a specific folder) and shares a view link.
- Extracts text from PDFs and sends it to an OpenAI model for structured parsing.
- Creates Calendar events populated with the parsed data plus a link to the uploaded PDF.
- Supports both continuous operation and one-time execution modes.

## Prerequisites
- Python 3.10+
- Google OAuth credentials with Gmail **modify**, Calendar, and Drive (file upload) scopes saved to `token.json` (overridable with `GOOGLE_TOKEN_PATH`).
- Environment variables:
  - `SENDER_EMAIL`: Email address to filter unread messages.
  - `OPENAI_API_KEY`: Key for the OpenAI API.
  - `OPENAI_MODEL` (optional): Model name to use (defaults to `gpt-4o-mini`).
  - `POLL_INTERVAL_SECONDS` (optional): How often the service checks for new messages in continuous mode (defaults to 300 seconds).
  - `DRIVE_FOLDER_ID` (optional): Google Drive folder ID where PDFs should be uploaded.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
2. Copy `.env_example` to `.env` and fill in your values:
   ```bash
   cp .env_example .env
   # edit .env with your sender email and OpenAI key
   ```
3. Run the service:

   ### Continuous Mode (default)
   Run the service as a long-running process that polls Gmail periodically:
   ```bash
   python main.py
   ```
   or explicitly:
   ```bash
   python main.py --mode continuous
   ```

   ### One-time Execution
   Process current unread messages once and exit:
   ```bash
   python main.py --mode once
   ```

   ### Command-line Options
   ```
   --mode {once,continuous}  Run mode: 'once' for single execution or 'continuous' for continuous operation (default: continuous)
   --sender SENDER           Email address to filter messages from (default: from SENDER_EMAIL env variable)
   --interval INTERVAL       Interval between checks in seconds for continuous mode (default: from POLL_INTERVAL_SECONDS env variable or 300)
   ```

   Note: Environment variables are loaded from the `.env` file using python-dotenv. You can also override them on the command line if needed:
   ```bash
   SENDER_EMAIL=organizer@example.com OPENAI_API_KEY=... POLL_INTERVAL_SECONDS=300 python main.py
   ```

The service will print the ID of each created Calendar event. Processed messages are marked as read to avoid duplication.
