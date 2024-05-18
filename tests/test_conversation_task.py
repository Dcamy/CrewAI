# iChain/tests/test_conversation_task.py
import unittest
from unittest.mock import MagicMock
from iChain.src.iChain.tasks.conversation import ConversationTask
from iChain.src.iChain.agents.assistant import PersonalAssistantAgent

class TestConversationTask(unittest.TestCase):
    """
    This class contains tests for the ConversationTask.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        # Mock the PersonalAssistantAgent
        self.mock_assistant = MagicMock(spec=PersonalAssistantAgent)
        self.task = ConversationTask(self.mock_assistant)

    def test_run_conversation(self):
        """
        Test that the task correctly processes the input data through the assistant agent.
        """
        user_input = "Hello, how can you help me today?"
        expected_output = "I can assist you with various tasks."

        # Mock the receive_message method
        self.mock_assistant.receive_message.return_value = expected_output

        result = self.task.run(user_input)
        self.assertEqual(result, expected_output)
        self.mock_assistant.receive_message.assert_called_once_with(user_input)

    def test_run_no_input_data(self):
        """
        Test that the task raises a ValueError when no input data is provided.
        """
        with self.assertRaises(ValueError):
            self.task.run("")

    def test_logging_initialization(self):
        """
        Test that the initialization logs the correct message.
        """
        with self.assertLogs('iChain.src.iChain.tasks.conversation', level='INFO') as log:
            _ = ConversationTask(self.mock_assistant)
            self.assertIn("[Task: ConversationTask] Initialized with assistant_agent:", log.output[0])

    def test_logging_conversation(self):
        """
        Test that the conversation data is logged correctly.
        """
        user_input = "Hello, how can you help me today?"
        expected_output = "I can assist you with various tasks."
        self.mock_assistant.receive_message.return_value = expected_output

        with self.assertLogs('iChain.src.iChain.tasks.conversation', level='INFO') as log:
            self.task.run(user_input)
            self.assertIn("[Task: ConversationTask] User input: Hello, how can you help me today? Assistant response: I can assist you with various tasks.", log.output[0])