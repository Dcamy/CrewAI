# iChain/src/iChain/tasks/anonymization.py

import logging
import os
import datetime
from crewai import Task

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'anonymization_task_log.txt'), level=logging.INFO)

class AnonymizationTask(Task):
    """
    This task handles the anonymization of sensitive information within data collected during interactions.
    It ensures that personal identifiers and other sensitive data are redacted to protect user privacy.
    """

    def __init__(self, data_cleaning_agent):
        """
        Initializes the Anonymization Task with a reference to a data cleaning agent.

        Args:
            data_cleaning_agent (object): The agent responsible for performing the data cleaning and anonymization.
        """
        super().__init__()
        self.data_cleaning_agent = data_cleaning_agent
        logging.info(f"[{datetime.datetime.now()}] [Task: AnonymizationTask] Initialized with data_cleaning_agent: {data_cleaning_agent}")

    def run(self, data):
        """
        Executes the anonymization process on the provided data.

        Args:
            data (str): The text data to be anonymized.

        Returns:
            str: The anonymized version of the input data.
        """
        if not data:
            logging.error(f"[{datetime.datetime.now()}] [Task: AnonymizationTask] No data provided for anonymization.")
            raise ValueError("No data provided for anonymization.")

        # Use the data cleaning agent to anonymize the data
        anonymized_data = self.data_cleaning_agent.anonymize_data(data)
        logging.info(f"[{datetime.datetime.now()}] [Task: AnonymizationTask] Anonymized data: {anonymized_data}")
        return anonymized_data