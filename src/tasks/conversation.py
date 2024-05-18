# iChain/src/iChain/tasks/conversation.py

import logging
import os
import datetime
from crewai import Task

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'conversation_task_log.txt'), level=logging.INFO)

class ConversationTask(Task):
    """
    This task manages interactions with the user, processing input and generating responses
    through the personal assistant agent.
    """

    def __init__(self, assistant_agent):
        """
        Initializes the Conversation Task with a reference to the personal assistant agent.

        Args:
            assistant_agent (object): The agent responsible for managing user interactions.
        """
        super().__init__()
        self.assistant_agent = assistant_agent
        logging.info(f"[{datetime.datetime.now()}] [Task: ConversationTask] Initialized with assistant_agent: {assistant_agent}")

    def run(self, input_data):
        """
        Executes the conversation task by processing the input data through the assistant agent.

        Args:
            input_data (str): The user input to process.

        Returns:
            str: The response generated by the assistant agent.
        """
        if not input_data:
            logging.error(f"[{datetime.datetime.now()}] [Task: ConversationTask] No input data provided for conversation.")
            raise ValueError("No input data provided for conversation.")

        # Use the assistant agent to handle the conversation and generate a response
        response = self.assistant_agent.receive_message(input_data)
        logging.info(f"[{datetime.datetime.now()}] [Task: ConversationTask] User input: {input_data} Assistant response: {response}")
        return response