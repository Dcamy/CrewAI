# iChain/src/agents/__init__.py

# Importing the necessary agent modules so they can be accessible from other parts of the application
from .assistant import PersonalAssistantAgent
from .cleaner import DataCleaningAgent
from .valuator import DataValuationAgent
from .vision import VisionAgent
from .manager import ManagerAgent

# Optionally, you can define a list of all agents for easy iteration if needed elsewhere in your application
__all__ = [
    "PersonalAssistantAgent",
    "DataCleaningAgent",
    "DataValuationAgent",
    "VisionAgent",
    "ManagerAgent",
]