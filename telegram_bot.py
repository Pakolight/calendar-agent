import telebot
import logging
from telebot import types
from dotenv import load_dotenv
import os
from service_parce_pdf_gmail import CalendarAgentService

load_dotenv()


API_TOKEN = os.environ.get("TELEGRAM_BOT", None)
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
BOT_OWNER_ID = os.environ.get("BOT_OWNER_ID", "")

# This is a Telegram bot that can check for callsheets in Gmail
# and process them using the CalendarAgentService

bot = telebot.TeleBot(API_TOKEN)


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
    if BOT_OWNER_ID and str(message.from_user.id) != BOT_OWNER_ID:
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
