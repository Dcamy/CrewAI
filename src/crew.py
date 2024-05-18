# iChain/src/crew.py
import os
from .agents import (
    PersonalAssistantAgent,
    DataCleaningAgent,
    DataValuationAgent,
    VisionAgent,
    ManagerAgent,
)
from .tasks import (
    ConversationTask,
    AnonymizationTask,
    DataUploadTask,
    ImageProcessingTask,
)
from .tools import GoogleDriveTool, DiscordBotTool, EmbeddingTool
from .memory import LongTermMemory, ShortTermMemory
from .config.llm_config import get_models
from crewai.process import Process
from crewai import Crew

class CrewManager:
    """
    CrewManager class to manage different agents and assign tasks within the CrewAI environment.
    """

    def __init__(self):
        # Initialize memory components
        self.long_term_memory = LongTermMemory("path_to_long_term_storage.db")
        self.short_term_memory = ShortTermMemory()

        # Initialize tools
        self.google_drive_tool = GoogleDriveTool("path_to_token.json")
        self.discord_bot_tool = DiscordBotTool(os.getenv("DISCORD_BOT_TOKEN"))
        self.embedding_tool = EmbeddingTool(os.getenv("OPENAI_API_KEY"))

        # Initialize agents
        self.assistant = PersonalAssistantAgent(
            self.short_term_memory, self.embedding_tool, get_models("groq", "llama3-8b-8192")
        )
        self.cleaner = DataCleaningAgent(
            memory=self.short_term_memory, llm=get_models("groq", "llama3-8b-8192")
        )
        self.valuator = DataValuationAgent(
            memory=self.short_term_memory, llm=get_models("groq", "llama3-8b-8192")
        )
        self.vision_agent = VisionAgent(
            memory=self.short_term_memory, llm=get_models("ollama", "llava-v1")
        )
        self.manager_agent = ManagerAgent(
            memory=self.short_term_memory, llm=get_models("groq", "llama3-70b-8192")
        )

        # Setup tasks
        self.conversation_task = ConversationTask(self.assistant)
        self.anonymization_task = AnonymizationTask(self.cleaner)
        self.data_upload_task = DataUploadTask(self.google_drive_tool, self.valuator)
        self.image_processing_task = ImageProcessingTask(self.vision_agent)

        # Setup the crew
        self.crew = Crew(
            agents=[
                self.assistant,
                self.cleaner,
                self.valuator,
                self.vision_agent,
            ],
            tasks=[
                self.conversation_task,
                self.anonymization_task,
                self.data_upload_task,
                self.image_processing_task,
            ],
            process=Process.hierarchical,
            manager_llm=self.manager_agent,
            verbose=True,
            memory=True,
            embedder=self.embedding_tool,
            full_output=True,
            output_log_file="C:\\CrewAI\\iChain\\Logs\\crew_log.txt",
        )

    def run(self):
        """
        Start the Crew operation.
        """
        # Kickoff the crew's task execution
        result = self.crew.kickoff()
        print(result)
        print("Crew is operational and running...")


# Instantiation and running the crew
if __name__ == "__main__":
    crew_manager = CrewManager()
    crew_manager.run()

# Future Enhancements:
# - Implement dynamic task allocation based on agent availability and skill set.
# - Develop agent performance metrics to reward models and improve task efficiency.
# - Add intelligent features like NLP and ML for autonomous decision-making.
# - Introduce innovative ideas such as live model fine-tuning or adaptive learning mechanisms.
# - Enhance the system to allow the manager agent to adjust short-term memory size dynamically and implement selective transfer to long-term memory.
# - Incorporate user-friendly authentication methods for Google Drive and other services.
# - Develop a streamlined process for integrating new tools and models.