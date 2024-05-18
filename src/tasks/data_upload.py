# iChain/src/iChain/tasks/data_upload.py

import logging
import os
import datetime
from crewai import Task

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'data_upload_task_log.txt'), level=logging.INFO)

class DataUploadTask(Task):
    """
    This task handles uploading anonymized and processed data to a secure storage solution.
    It ensures that data is safely stored and retrievable for future use.
    """

    def __init__(self, storage_tool, data_valuator):
        """
        Initializes the Data Upload Task with references to necessary tools and agents.

        Args:
            storage_tool (object): Tool used to interact with the data storage solution (e.g., Google Drive API).
            data_valuator (object): The agent responsible for assessing data before uploading.
        """
        super().__init__()
        self.storage_tool = storage_tool
        self.data_valuator = data_valuator
        logging.info(f"[{datetime.datetime.now()}] [Task: DataUploadTask] Initialized with storage_tool: {storage_tool}, data_valuator: {data_valuator}")

    def run(self, data):
        """
        Executes the task of uploading data after assessing its quality and value.

        Args:
            data (dict): Containing 'data' to be uploaded and 'metadata' such as source, type, etc.

        Returns:
            str: Status message indicating success or failure of the upload.
        """
        if not data or "data" not in data:
            logging.error(f"[{datetime.datetime.now()}] [Task: DataUploadTask] Invalid data provided for upload.")
            raise ValueError("Invalid data provided for upload.")

        # Assess the data's value before uploading
        value, assessment = self.data_valuator.assess_data(data["data"])
        logging.info(f"[{datetime.datetime.now()}] [Task: DataUploadTask] Data assessment: {assessment}, Value: {value}")
        
        if value < 50:  # Assume a threshold value below which data is not worth uploading
            logging.warning(f"[{datetime.datetime.now()}] [Task: DataUploadTask] Data not uploaded due to low value.")
            return "Data not uploaded due to low value."

        # Upload the data using the storage tool
        upload_status = self.storage_tool.upload(data["data"], data.get("metadata", {}))
        logging.info(f"[{datetime.datetime.now()}] [Task: DataUploadTask] Data uploaded successfully: {upload_status}")
        return f"Data uploaded successfully on {datetime.datetime.now()}: {upload_status}"