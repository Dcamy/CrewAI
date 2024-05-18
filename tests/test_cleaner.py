# iChain/tests/test_cleaner.py

import unittest
from unittest.mock import MagicMock
from iChain.src.agents.cleaner import DataCleaningAgent
from iChain.src.config.llm_config import get_models
from iChain.src.memory.short_term import ShortTermMemory


class TestDataCleaningAgent(unittest.TestCase):
    """
    This class contains tests for the DataCleaningAgent.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        self.memory = ShortTermMemory()
        self.llm = get_models("groq", "llama3-8b-8192")
        self.agent = DataCleaningAgent(self.memory, self.llm)

    def test_anonymize_data_email(self):
        """
        Test that the agent correctly anonymizes email addresses.
        """
        sample_data = "Contact me at example@example.com."
        expected_output = "Contact me at [REDACTED]."
        result = self.agent.anonymize_data(sample_data)
        self.assertEqual(result, expected_output)

    def test_anonymize_data_phone_number(self):
        """
        Test that the agent correctly anonymizes phone numbers.
        """
        sample_data = "My phone number is 123-456-7890."
        expected_output = "My phone number is [REDACTED]."
        result = self.agent.anonymize_data(sample_data)
        self.assertEqual(result, expected_output)

    def test_anonymize_data_ssn(self):
        """
        Test that the agent correctly anonymizes social security numbers.
        """
        sample_data = "My SSN is 123-45-6789."
        expected_output = "My SSN is [REDACTED]."
        result = self.agent.anonymize_data(sample_data)
        self.assertEqual(result, expected_output)

    def test_process_data(self):
        """
        Test the overall data processing including logging.
        """
        sample_data = "Contact me at example@example.com or call 123-456-7890."
        expected_output = "Contact me at [REDACTED] or call [REDACTED]."
        result = self.agent.process_data(sample_data)
        self.assertEqual(result, expected_output)

    def test_log_cleaning(self):
        """
        Test that the logging function logs the correct data.
        """
        original_data = "Contact me at example@example.com."
        anonymized_data = "Contact me at [REDACTED]."
        
        # Mocking the logger
        with self.assertLogs('iChain.src.agents.cleaner', level='INFO') as log:
            self.agent.log_cleaning(original_data, anonymized_data)
            self.assertIn("Original: Contact me at example@example.com.", log.output[0])
            self.assertIn("Anonymized: Contact me at [REDACTED].", log.output[0])

if __name__ == "__main__":
    unittest.main()
