# iChain/tests/test_data_upload_task.py
import unittest
from unittest.mock import MagicMock
from iChain.src.iChain.tasks.data_upload import DataUploadTask
from iChain.src.iChain.tools.google_drive import GoogleDriveTool
from iChain.src.iChain.agents.valuator import DataValuationAgent
from src.iChain.config.llm_config import get_models

class TestDataUploadTask(unittest.TestCase):
    """
    This class contains tests for the DataUploadTask.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        # Mock the GoogleDriveTool
        self.mock_storage_tool = MagicMock(spec=GoogleDriveTool)
        # Mock the DataValuationAgent
        self.mock_valuator = MagicMock(spec=DataValuationAgent)

        # Create the DataUploadTask with mocked dependencies
        self.task = DataUploadTask(self.mock_storage_tool, self.mock_valuator)

    def test_run_successful_upload(self):
        """
        Test that the task correctly processes and uploads data when the value is high.
        """
        sample_data = {
            "data": "This is sample data that has been anonymized.",
            "metadata": {"source": "user_interaction", "type": "text"},
        }

        # Mock the assess_data method to return high value
        self.mock_valuator.assess_data.return_value = (100, "High value - contains rich and relevant content")
        # Mock the upload method to simulate a successful upload
        self.mock_storage_tool.upload.return_value = "file_id_123"

        result = self.task.run(sample_data)
        self.assertIn("Data uploaded successfully", result)
        self.mock_valuator.assess_data.assert_called_once_with(sample_data["data"])
        self.mock_storage_tool.upload.assert_called_once_with(sample_data["data"], sample_data["metadata"])

    def test_run_low_value_data(self):
        """
        Test that the task does not upload data when the value is low.
        """
        sample_data = {
            "data": "This is low-value data.",
            "metadata": {"source": "user_interaction", "type": "text"},
        }

        # Mock the assess_data method to return low value
        self.mock_valuator.assess_data.return_value = (30, "Low value - not worth uploading")

        result = self.task.run(sample_data)
        self.assertEqual(result, "Data not uploaded due to low value.")
        self.mock_valuator.assess_data.assert_called_once_with(sample_data["data"])
        self.mock_storage_tool.upload.assert_not_called()

    def test_run_invalid_data(self):
        """
        Test that the task raises a ValueError when invalid data is provided.
        """
        with self.assertRaises(ValueError):
            self.task.run({"metadata": {"source": "user_interaction", "type": "text"}})

    def test_logging_initialization(self):
        """
        Test that the initialization logs the correct message.
        """
        with self.assertLogs('iChain.src.iChain.tasks.data_upload', level='INFO') as log:
            _ = DataUploadTask(self.mock_storage_tool, self.mock_valuator)
            self.assertIn("[Task: DataUploadTask] Initialized with storage_tool:", log.output[0])

    def test_logging_data_assessment(self):
        """
        Test that the data assessment is logged correctly.
        """
        sample_data = {
            "data": "This is sample data that has been anonymized.",
            "metadata": {"source": "user_interaction", "type": "text"},
        }
        self.mock_valuator.assess_data.return_value = (100, "High value - contains rich and relevant content")

        with self.assertLogs('iChain.src.iChain.tasks.data_upload', level='INFO') as log:
            self.task.run(sample_data)
            self.assertIn("[Task: DataUploadTask] Data assessment: High value - contains rich and relevant content, Value: 100", log.output[0])