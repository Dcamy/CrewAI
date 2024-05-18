# iChain/src/iChain/agents/manager.py
from crewai import Agent
import os
from src.iChain.config.llm_config import get_models
import datetime
import logging

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_directory, 'manager_log.txt'),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ManagerAgent(Agent):
    """
    This class defines the Manager Agent responsible for overseeing and managing the tasks 
    and interactions of other agents within the crew.
    """

    def __init__(self, memory, llm):
        """
        Initializes the Manager Agent with memory and LLM capabilities.

        Args:
            memory (object): A memory object capable of storing session data (imported from CrewAI).
            llm (object): The language model the agent will use for generating responses.
        """
        super().__init__(role="Manager", verbose=True, memory=memory, llm=llm)

    def manage_tasks(self, tasks):
        """
        Manages the given tasks by delegating them to the appropriate agents and monitoring their execution.

        Args:
            tasks (list): A list of tasks to be managed.
        """
        for task in tasks:
            logger.info(f"Managing task: {task}")
            try:
                result = task.run()
                logger.info(f"Task {task} completed successfully with result: {result}")
            except Exception as e:
                logger.error(f"Error managing task {task}: {e}")

if __name__ == "__main__":
    # Initialize memory and LLM
    memory = True  # Memory imported from CrewAI
    llm = get_models("groq", "llama3-70b")  # Set the manager's LLM to Groq llama3-70b
    manager = ManagerAgent(memory, llm)

    # Example task list for demonstration purposes
    from iChain.tasks.conversation import ConversationTask
    from iChain.agents.assistant import PersonalAssistantAgent
    from iChain.config.llm_config import get_models

    assistant_llm = get_models("groq", "llama3-8b-8192")
    assistant = PersonalAssistantAgent(memory, get_models("ollama", "llama-base"), assistant_llm)
    conversation_task = ConversationTask(assistant)
    tasks = [conversation_task]

    manager.manage_tasks(tasks)