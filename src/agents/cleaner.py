# iChain/src/iChain/agents/cleaner.py

from crewai import Agent
import re
import os
import datetime
from src.iChain.config.llm_config import get_models
import logging

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'cleaning_log.txt'), level=logging.INFO)

class DataCleaningAgent(Agent):
    """
    This class defines the Data Cleaning Agent that handles the anonymization of sensitive information
    from data collected during interactions. It uses regular expressions to identify and anonymize
    personal identifiers and other sensitive data.
    """

    def __init__(self, memory, llm):
        """
        Initializes the Data Cleaning Agent with specific roles and goals.

        Args:
            memory (object): A memory object capable of storing session data (imported from CrewAI).
            llm (object): The language model the agent will use for generating responses.
        """
        super().__init__(role="Data Cleaner", verbose=True, memory=memory, llm=llm)

    def anonymize_data(self, text):
        """
        Anonymizes sensitive information in the given text.

        Args:
            text (str): The text containing potentially sensitive information.

        Returns:
            str: The anonymized text.
        """
        try:
            # Regular expressions for detecting sensitive information
            patterns = {
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            }

            # Anonymizing data
            for key, pattern in patterns.items():
                text = re.sub(pattern, "[REDACTED]", text)

            return text
        except Exception as e:
            logging.error(f"Error anonymizing data: {e}")
            return None

    def process_data(self, data):
        """
        Processes the data through anonymization and logs the operation.

        Args:
            data (str): Data to be cleaned.

        Returns:
            str: Cleaned data.
        """
        try:
            anonymized_data = self.anonymize_data(data)
            self.log_cleaning(data, anonymized_data)
            return anonymized_data
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            return None

    def log_cleaning(self, original, anonymized):
        """
        Logs the before and after of data cleaning for audit and improvements.

        Args:
            original (str): The original data before cleaning.
            anonymized (str): The data after anonymization.
        """
        log_message = f"Original: {original}\nAnonymized: {anonymized}\nLogged at: {datetime.datetime.now()}\n---\n"
        logging.info(log_message)