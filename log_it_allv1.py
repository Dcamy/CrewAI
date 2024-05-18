# logging_wrapper.py

import logging
import sys
import os
from datetime import datetime

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create 'Logs' directory relative to the script
log_directory = os.path.join(script_dir, "Logs")
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(
    log_directory, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ... (rest of the StreamToLogger class code is the same)

# Log the contents of the source files recursively
def log_source_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip specified directories
        dirnames[:] = [d for d in dirnames if d not in {".git", "..venv/", "__pycache__", "Logs"}]
        for filename in filenames:
            if filename.endswith((".pyc", ".pyo")): # Skip compiled Python files
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    logger.info(f"File contents: {file_path}\n{content}\n")
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")

# Log source files starting from the parent directory of the script
log_source_files(os.path.dirname(script_dir)) # One level up

# Run the main application (modified to use relative paths)
"""try:
    from .src.main import main # Assuming iChain is one level up 

    main()
except ModuleNotFoundError as e:
    logger.error(f"Module not found: {e}")
except Exception as e:
    logger.error(f"An error occurred while running the main application: {e}") """