# iChain/tests/__init__.py

# Import test modules here if necessary to allow easier access from test runners or for package organization
from .test_assistant import TestPersonalAssistantAgent
from .test_cleaner import TestDataCleaningAgent
from .test_valuator import TestDataValuationAgent
from .test_vision import TestVisionAgent
from .test_memory_long_term import TestLongTermMemory
from .test_memory_short_term import TestShortTermMemory
from .test_file_handler import TestFileHandler
from .test_google_drive import TestGoogleDriveTool
from .test_discord_bot_tool import TestDiscordBotTool
from .test_embedding_tool import TestEmbeddingTool
from .test_anonymization_task import TestAnonymizationTask
from .test_conversation_task import TestConversationTask
from .test_data_upload_task import TestDataUploadTask
from .test_image_processing_task import TestImageProcessingTask

# You can define additional imports for other test classes as you expand your testing suite