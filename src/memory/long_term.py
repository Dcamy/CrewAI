# iChain/src/iChain/memory/long_term.py

import os
import logging
import datetime

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'long_term_memory_log.txt'), level=logging.INFO)

class LongTermMemory:
    """
    This class provides functionality for storing and retrieving long-term memory,
    which can include user preferences, historical interaction data, and other information
    that should be persisted across sessions.
    """

    def __init__(self, storage_path):
        """
        Initialize the LongTermMemory with a path to the storage file or database.

        Args:
            storage_path (str): The path to the file or database where long-term memories are stored.
        """
        self.storage_path = storage_path

    def save_memory(self, key, value):
        """
        Saves a piece of memory.

        Args:
            key (str): The key under which the memory should be stored.
            value (any): The data to store.
        """
        try:
            # Example: Save to a simple file or extend to use a database
            with open(self.storage_path, "a") as file:
                file.write(f"{key}:{value}\n")
            logging.info(f"[Memory: LongTerm] [Task: save_memory] Key: {key} Value: {value} Saved at: {datetime.datetime.now()}")
        except Exception as e:
            logging.error(f"[Memory: LongTerm] [Task: save_memory] Error saving memory: {e}")

    def retrieve_memory(self, key):
        """
        Retrieves a piece of memory.

        Args:
            key (str): The key for the memory to retrieve.

        Returns:
            any: The retrieved data, or None if the key is not found.
        """
        try:
            # Example: Read from a simple file or extend to use a database
            with open(self.storage_path, "r") as file:
                for line in file:
                    k, v = line.strip().split(":")
                    if k == key:
                        logging.info(f"[Memory: LongTerm] [Task: retrieve_memory] Key: {key} Retrieved at: {datetime.datetime.now()}")
                        return v
            logging.info(f"[Memory: LongTerm] [Task: retrieve_memory] Key: {key} Not found at: {datetime.datetime.now()}")
            return None
        except Exception as e:
            logging.error(f"[Memory: LongTerm] [Task: retrieve_memory] Error retrieving memory: {e}")
            return None