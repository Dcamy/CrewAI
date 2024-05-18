# iChain/tests/test_anonymization_task.py
import unittest
from unittest.mock import MagicMock
from iChain.src.iChain.tasks.anonymization import AnonymizationTask
from iChain.src.iChain.agents.cleaner import DataCleaningAgent

class TestAnonymizationTask(unittest.TestCase):
    """
    This class contains tests for the AnonymizationTask.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        # Mock the DataCleaningAgent
        self.mock_cleaner = MagicMock(spec=DataCleaningAgent)
        self.task = AnonymizationTask(self.mock_cleaner)

    def test_run_anonymization(self):
        """
        Test that the task correctly anonymizes the data.
        """
        sample_data = "Here is a sample data containing sensitive information like email@example.com."
        expected_output = "Here is a sample data containing sensitive information like [REDACTED]."

        # Mock the anonymize_data method
        self.mock_cleaner.anonymize_data.return_value = expected_output
        
        result = self.task.run(sample_data)
        self.assertEqual(result, expected_output)
        self.mock_cleaner.anonymize_data.assert_called_once_with(sample_data)

    def test_run_no_data(self):
        """
        Test that the task raises a ValueError when no data is provided.
        """
        with self.assertRaises(ValueError):
            self.task.run("")

    def test_logging_initialization(self):
        """
        Test that the initialization logs the correct message.
        """
        with self.assertLogs('iChain.src.iChain.tasks.anonymization', level='INFO') as log:
            _ = AnonymizationTask(self.mock_cleaner)
            self.assertIn("[Task: AnonymizationTask] Initialized with data_cleaning_agent:", log.output[0])

    def test_logging_anonymized_data(self):
        """
        Test that the anonymized data is logged correctly.
        """
        sample_data = "Here is a sample data containing sensitive information like email@example.com."
        expected_output = "Here is a sample data containing sensitive information like [REDACTED]."
        self.mock_cleaner.anonymize_data.return_value = expected_output
        
        with self.assertLogs('iChain.src.iChain.tasks.anonymization', level='INFO') as log:
            self.task.run(sample_data)
            self.assertIn("[Task: AnonymizationTask] Anonymized data:", log.output[0])