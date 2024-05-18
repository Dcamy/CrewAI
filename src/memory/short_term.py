# iChain/src/iChain/memory/short_term.py

import logging
import os
import datetime
from collections import deque

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'short_term_memory_log.txt'), level=logging.INFO)

class ShortTermMemory:
    """
    This class provides functionality for storing and retrieving short-term memory,
    which is used to maintain context during a user's interaction session.
    """

    def __init__(self, max_size=10):
        """
        Initialize the ShortTermMemory with a deque to hold the memory items, limited to max_size.

        Args:
            max_size (int): The maximum number of memory items to store.
        """
        self.memory_store = deque(maxlen=max_size)
        self.max_size = max_size

    def store_memory(self, key, value):
        """
        Stores a piece of memory in the short-term store.

        Args:
            key (str): The key under which the memory should be stored.
            value (any): The data to store.
        """
        try:
            self.memory_store.append((key, value))
            logging.info(f"[Memory: ShortTerm] [Task: store_memory] Key: {key} Value: {value} Stored at: {datetime.datetime.now()}")
        except Exception as e:
            logging.error(f"[Memory: ShortTerm] [Task: store_memory] Error storing memory: {e}")

    def retrieve_memory(self, key):
        """
        Retrieves a piece of memory from the short-term store.

        Args:
            key (str): The key for the memory to retrieve.

        Returns:
            any: The retrieved data if available, else None.
        """
        try:
            for k, v in self.memory_store:
                if k == key:
                    logging.info(f"[Memory: ShortTerm] [Task: retrieve_memory] Key: {key} Retrieved at: {datetime.datetime.now()}")
                    return v
            logging.info(f"[Memory: ShortTerm] [Task: retrieve_memory] Key: {key} Not found at: {datetime.datetime.now()}")
            return None
        except Exception as e:
            logging.error(f"[Memory: ShortTerm] [Task: retrieve_memory] Error retrieving memory: {e}")
            return None

    def clear_memory(self):
        """
        Clears all stored memory, typically used at the end of a session or when memory needs refreshing.
        """
        try:
            self.memory_store.clear()
            logging.info(f"[Memory: ShortTerm] [Task: clear_memory] Memory cleared at: {datetime.datetime.now()}")
        except Exception as e:
            logging.error(f"[Memory: ShortTerm] [Task: clear_memory] Error clearing memory: {e}")