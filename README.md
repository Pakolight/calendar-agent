# Calendar Agent

This script automates processing unread Gmail messages with PDF attachments from a specified sender, extracts structured event details using an LLM, and creates Google Calendar events. It can be run as a long-lived bot that polls Gmail on a schedule.

## Features
- Queries Gmail for unread messages from a configured sender that include PDF attachments.
- Extracts text from PDFs and sends it to an OpenAI model for structured parsing.
- Creates Calendar events populated with the parsed data.

## Prerequisites
- Python 3.10+
- Google OAuth credentials with Gmail **modify** and Calendar scopes saved to `token.json` (overridable with `GOOGLE_TOKEN_PATH`).
- Environment variables:
  - `SENDER_EMAIL`: Email address to filter unread messages.
  - `OPENAI_API_KEY`: Key for the OpenAI API.
  - `OPENAI_MODEL` (optional): Model name to use (defaults to `gpt-4o-mini`).
  - `POLL_INTERVAL_SECONDS` (optional): How often the bot checks for new messages (defaults to 300 seconds).

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
3. Run the bot (polling Gmail periodically):
   ```bash
   python main.py
   ```

   Note: Environment variables are loaded from the `.env` file using python-dotenv. You can also override them on the command line if needed:
   ```bash
   SENDER_EMAIL=organizer@example.com OPENAI_API_KEY=... POLL_INTERVAL_SECONDS=300 python main.py
   ```

The script will print the ID of each created Calendar event. Processed messages are marked as read to avoid duplication while the bot runs.
