# iChain/src/config.py
import os
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, "Logs")

# Dynamic logging level (default to INFO)
LOGGING_LEVEL = logging.INFO 

def setup_logging():
    """Configures logging for the entire iChain project."""
    logging.basicConfig(
        filename=os.path.join(LOGS_DIR, 'iChain_log.txt'), 
        level=LOGGING_LEVEL, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# Call this at the start of your main script or application entry point
setup_logging() 