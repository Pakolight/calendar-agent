import os
import sys
import argparse

from dotenv import load_dotenv
from service_parce_pdf_gmail import CalendarAgentService

# Load environment variables from .env file
load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calendar Agent Service")
    parser.add_argument(
        "--mode", 
        choices=["once", "continuous"], 
        default="continuous",
        help="Run mode: 'once' for single execution or 'continuous' for continuous operation (default: continuous)"
    )
    parser.add_argument(
        "--sender", 
        help="Email address to filter messages from (default: from SENDER_EMAIL env variable)"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        help="Interval between checks in seconds for continuous mode (default: from POLL_INTERVAL_SECONDS env variable or 300)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Get sender email from args or environment
    sender_email = args.sender or os.environ.get("SENDER_EMAIL", "")
    if not sender_email:
        print("Error: Sender email is required. Provide it with --sender or set SENDER_EMAIL environment variable.")
        sys.exit(1)

    # Get interval from args or environment
    interval_seconds = args.interval or int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))

    # Create and run the service
    service = CalendarAgentService(sender_email, interval_seconds)

    if args.mode == "once":
        service.run_once()
    else:
        service.run_continuously()