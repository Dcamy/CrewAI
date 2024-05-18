# iChain/tests/test_assistant.py
import unittest
from unittest.mock import MagicMock
from ..src.agents.assistant import PersonalAssistantAgent
from .src.memory.short_term import ShortTermMemory
from .src.tools.embedding_tool import EmbeddingTool
from .src.config.llm_config import get_modelsp
import sys
print(sys.path)

class TestPersonalAssistantAgent(unittest.TestCase):
    """
    This class contains tests for the PersonalAssistantAgent.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        # Mock dependencies
        self.memory = ShortTermMemory()
        self.embedder = MagicMock(spec=EmbeddingTool)
        self.llm = get_models("groq", "llama3-8b-8192")
        self.agent = PersonalAssistantAgent(self.memory, self.embedder, self.llm)

    def test_receive_message(self):
        """
        Test that the agent correctly processes an input message.
        """
        self.embedder.embed.return_value = "embedded_message"
        response = self.agent.receive_message("Hello, how are you?")
        self.assertIsNotNone(response)  # Check that we get a response
        self.assertIsInstance(response, str)  # Ensure the response is a string
        self.embedder.embed.assert_called_once_with("Hello, how are you?")

    def test_receive_message_error(self):
        """
        Test that the agent handles errors during message processing.
        """
        self.embedder.embed.side_effect = Exception("Embedding error")
        response = self.agent.receive_message("Hello, how are you?")
        self.assertEqual(response, "Sorry, there was an error processing your message.")

    def test_process_message(self):
        """
        Test that the agent processes the embedded message correctly.
        """
        response = self.agent.process_message("embedded_message")
        self.assertEqual(response, "Thank you for your message. I'm currently learning to respond more effectively.")

    def test_log_interaction(self):
        """
        Test that the agent logs the interaction correctly.
        """
        self.agent.log_interaction("Hello, how are you?", "I am fine, thank you.")
        # Here we should check the log file or mock the logging to assert the log output