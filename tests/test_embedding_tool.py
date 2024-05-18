# iChain/tests/test_embedding_tool.py
import unittest
from unittest.mock import patch, MagicMock
from iChain.src.iChain.tools.embedding_tool import EmbeddingTool
import openai

class TestEmbeddingTool(unittest.TestCase):
    """
    This class contains tests for the EmbeddingTool.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        self.api_key = "dummy_api_key"
        self.embedding_tool = EmbeddingTool(self.api_key)

    @patch('openai.Embedding.create')
    def test_generate_embedding_success(self, mock_create):
        """
        Test that generate_embedding successfully generates embeddings.
        """
        # Mock response from OpenAI API
        mock_create.return_value = {
            "data": [0.1, 0.2, 0.3]
        }
        
        text = "Example text for embedding."
        embedding = self.embedding_tool.generate_embedding(text)
        
        self.assertIsNotNone(embedding)  # Check that we get a response
        self.assertIsInstance(embedding, list)  # Ensure the response is a list
        self.assertEqual(embedding, [0.1, 0.2, 0.3])  # Check the returned embedding

    @patch('openai.Embedding.create')
    def test_generate_embedding_failure(self, mock_create):
        """
        Test that generate_embedding handles failures correctly.
        """
        # Mock an exception raised by the OpenAI API
        mock_create.side_effect = Exception("Test exception")
        
        text = "Example text for embedding."
        embedding = self.embedding_tool.generate_embedding(text)
        
        self.assertIsNone(embedding)  # Check that the response is None in case of failure