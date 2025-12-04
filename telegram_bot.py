import os
import tempfile
import logging

import telebot
from telebot import types
from dotenv import load_dotenv

from board_ocr import events_to_json, format_events_text, process_board_image
from service_parce_pdf_gmail import CalendarAgentService

load_dotenv()


API_TOKEN = os.environ.get("TELEGRAM_BOT", None)
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
BOT_OWNER_ID = os.environ.get("BOT_OWNER_ID", "")
# Convert BOT_OWNER_ID to integer if it exists
try:
    BOT_OWNER_ID = int(BOT_OWNER_ID) if BOT_OWNER_ID else None
except ValueError:
    print(f"Warning: BOT_OWNER_ID '{BOT_OWNER_ID}' is not a valid integer. Owner verification will not work correctly.")

# This is a Telegram bot that can check for callsheets in Gmail
# and process them using the CalendarAgentService

bot = telebot.TeleBot(API_TOKEN)


@bot.message_handler(content_types=['photo'])
def handle_board_photo(message: telebot.types.Message) -> None:
    """Handle photos of planning boards, run OCR, and return parsed events."""
    bot.send_chat_action(message.chat.id, 'typing')

    try:
        photo = message.photo[-1]
        file_info = bot.get_file(photo.file_id)
        downloaded = bot.download_file(file_info.file_path)

        temp_dir = tempfile.mkdtemp(prefix="board_ocr_")
        image_path = os.path.join(temp_dir, 'board.jpg')
        with open(image_path, 'wb') as image_file:
            image_file.write(downloaded)

        events = process_board_image(image_path)
        if not events:
            bot.reply_to(message, "I couldn't recognize any rows. Please try another photo.")
            return

        summary = f"I found {len(events)} event{'s' if len(events) != 1 else ''} on the board."
        formatted = format_events_text(events)
        bot.send_message(message.chat.id, f"{summary}\n\n{formatted}")

        json_payload = events_to_json(events)
        json_path = os.path.join(temp_dir, 'events.json')
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_payload)

        with open(json_path, 'rb') as json_file:
            bot.send_document(
                message.chat.id,
                json_file,
                caption="Parsed events as JSON",
                visible_file_name='events.json',
            )
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Failed to process board photo")
        bot.reply_to(message, f"Failed to process the photo: {exc}")


# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    # Create keyboard with buttons
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    check_button = types.KeyboardButton('Check callsheets')
    markup.add(check_button)

    bot.reply_to(message, """\
Hi there, I am CalendarAgent Bot.
I can check your Gmail for callsheets and process them.
Press the 'Check callsheets' button to start processing.\
""", reply_markup=markup)


# Handle 'Check callsheets' button
@bot.message_handler(func=lambda message: message.text == 'Check callsheets')
def check_callsheets(message):
    # Only allow the bot owner to use this feature
    if BOT_OWNER_ID and message.from_user.id != BOT_OWNER_ID:
        bot.reply_to(message, "Sorry, you are not authorized to use this feature.")
        return

    bot.reply_to(message, "Starting to check for callsheets. This may take a moment...")

    try:
        # Create and run the service
        if not SENDER_EMAIL:
            bot.reply_to(message, "Error: Sender email is not configured. Please set the SENDER_EMAIL environment variable.")
            return

        service = CalendarAgentService(SENDER_EMAIL)
        events_created = service.run_once()

        if events_created == 0:
            bot.reply_to(message, "No new callsheets found.")
        else:
            bot.reply_to(message, f"Successfully processed {events_created} callsheets!")
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")


# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.reply_to(message, "I don't understand that command. Press 'Check callsheets' or type /help for assistance.")


bot.infinity_polling()
