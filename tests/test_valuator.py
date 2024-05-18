# iChain/tests/test_valuator.py
# iChain/tests/test_valuation.py

import unittest
from unittest.mock import MagicMock
from iChain.src.iChain.agents.valuator import DataValuationAgent
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TestDataValuationAgent(unittest.TestCase):
    """
    This class contains tests for the DataValuationAgent.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        memory = True  # Mocked memory object
        self.llm = MagicMock()
        self.agent = DataValuationAgent(memory, self.llm)

    def test_assess_data_high_value(self):
        """
        Test that the agent correctly assesses high-value data.
        """
        self.llm.generate_response.return_value = "Relevant content"
        sample_data = "This data contains significant insights and detailed information."
        value, assessment = self.agent.assess_data(sample_data)
        self.assertEqual(value, 100)
        self.assertIn("high value", assessment.lower())

    def test_assess_data_low_value(self):
        """
        Test that the agent correctly assesses low-value data.
        """
        self.llm.generate_response.return_value = "REDACTED"
        sample_data = "Generic data with little unique content."
        value, assessment = self.agent.assess_data(sample_data)
        self.assertEqual(value, 50)
        self.assertIn("medium value", assessment.lower())

    def test_assess_data_with_exception(self):
        """
        Test that the agent handles exceptions gracefully.
        """
        self.agent.evaluate_data_quality = MagicMock(side_effect=Exception("Test error"))
        sample_data = "This is a sample data."
        value, assessment = self.agent.assess_data(sample_data)
        self.assertIsNone(value)
        self.assertIn("error assessing data", assessment.lower())

    def test_evaluate_data_quality(self):
        """
        Test the evaluation of data quality.
        """
        sample_data = "This is a sample data with useful content."
        tokens = word_tokenize(sample_data)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

        self.llm.generate_response.return_value = "Relevant content"
        value, assessment = self.agent.evaluate_data_quality(sample_data)
        self.assertEqual(value, 100)
        self.assertIn("high value", assessment.lower())

    def test_log_assessment(self):
        """
        Test the logging of assessment.
        """
        sample_data = "This is a sample data."
        value = 100
        assessment = "High value - contains rich and relevant content"
        self.agent.log_assessment(sample_data, value, assessment)
        # Check the log file for the correct log entry if needed