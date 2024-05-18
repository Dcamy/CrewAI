# iChain/src/agents/assistant.py

from crewai import Agent
import datetime
import os
from ..config import LOGGING_LEVEL, setup_logging
from ..config.llm_config import get_models
import logging


class PersonalAssistantAgent(Agent):
    """
    This class defines the Personal Assistant Agent that interacts with users to collect data,
    manage conversations, and handle various user requests. It incorporates memory and embedding tools
    to enhance conversation context and data collection.
    """

    # Configure logging
    def set_logging_level(self, level):
        """Sets the logging level for the agent."""
        try:
            global LOGGING_LEVEL  # Access the global logging level
            LOGGING_LEVEL = getattr(logging, level.upper(), LOGGING_LEVEL)
            setup_logging() # Reconfigure logging with the new level
            logging.info(f"[{datetime.datetime.now()}] [Agent: PersonalAssistantAgent] Logging level set to: {LOGGING_LEVEL}")
        except Exception as e:
            logging.error(f"[{datetime.datetime.now()}] [Agent: PersonalAssistantAgent] Error setting logging level: {e}")


    def __init__(self, memory, embedder, llm):
        """
        Initializes the Personal Assistant Agent with memory and embedding capabilities.

        Args:
            memory (object): A memory object capable of storing session data (imported from CrewAI).
            embedder (object): An embedding tool for processing text input.
            llm (object): The language model the agent will use for generating responses.
        """
        super().__init__(role="Personal Assistant", verbose=True, memory=memory, llm=llm)
        self.embedder = embedder
        logging.info(f"[{datetime.datetime.now()}] [Agent: PersonalAssistantAgent] Initialized with memory: {memory}, embedder: {embedder}, llm: {llm}")

    def receive_message(self, message):
        """
        Handles incoming messages from users and processes them accordingly.

        Args:
            message (str): The message received from the user.

        Returns:
            str: The response generated based on the user's message.
        """
        try:
            # Process the message using the embedding tool
            embedded_message = self.embedder.embed(message)

            # Generate a response
            response = self.process_message(embedded_message)
            logging.info(f"[{datetime.datetime.now()}] [Agent: PersonalAssistantAgent] Received message: {message}, Generated response: {response}")
            return response
        except Exception as e:
            logging.error(f"[{datetime.datetime.now()}] [Agent: PersonalAssistantAgent] Error receiving message: {e}")
            return "Sorry, there was an error processing your message."

    def process_message(self, embedded_message):
        """
        Processes the embedded message and generates a response.

        Args:
            embedded_message (any): The embedded representation of the user's message.

        Returns:
            str: A response to the user's message.
        """
        # Example processing logic, to be expanded based on specific use cases
        # Here, you would typically add logic to analyze the embedded message and generate a response
        print(f"Processing message at {datetime.datetime.now()}")
        return "Thank you for your message. I'm currently learning to respond more effectively."

    def log_interaction(self, message, response):
        """
        Logs the interaction between the agent and the user for analysis and improvements.

        Args:
            message (str): The original message from the user.
            response (str): The agent's response to the user.
        """
        try:
            log_message = f"User: {message}\nAgent: {response}\nLogged at: {datetime.datetime.now()}\n---\n"
            logging.info(log_message)
        except Exception as e:
            logging.error(f"[{datetime.datetime.now()}] [Agent: PersonalAssistantAgent] Error logging interaction: {e}")