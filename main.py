"""
Calendar Agent - PDF Email Processor

This functionality has been split into separate files:
1. service_parce_pdf_gmail.py - Contains the service class for processing PDFs from Gmail
2. bot_raner.py - Contains the runner code for launching the service

Please use one of the following commands to run the service:
- python bot_raner.py --mode continuous  # Run continuously (default)
- python bot_raner.py --mode once        # Run once and exit

For more options, run:
- python bot_raner.py --help
"""

# This file is kept for backward compatibility
# The actual implementation has been moved to service_parce_pdf_gmail.py and bot_raner.py

import sys

if __name__ == "__main__":
    print("This file is deprecated. Please use bot_raner.py instead.")
    print("Example: python bot_raner.py --mode continuous")
    sys.exit(0)
