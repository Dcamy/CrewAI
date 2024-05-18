# iChain/src/iChain/tools/embedding_tool.py

import openai
import os
import logging

# Setup logging configuration
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_directory, 'embedding_tool_log.txt'),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EmbeddingTool:
    """
    This tool is used to generate embeddings from text using the Llama3-8b model or any similar model.
    Embeddings are vector representations of text that can be used in various applications such as
    similarity measurement, data retrieval, and more.
    """

    def __init__(self, api_key):
        """
        Initializes the Embedding Tool with the necessary API key for the embedding service.

        Args:
            api_key (str): API key for accessing the model via OpenAI or a similar service.
        """
        self.api_key = api_key
        openai.api_key = api_key
        logger.info(f"Initialized EmbeddingTool with API key.")

    def generate_embedding(self, text):
        """
        Generates an embedding for the given text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            list: The embedding vector of the given text.
        """
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",  # Adjust model identifier based on actual availability and requirements
                input=text,
            )
            embedding_vector = response["data"]
            logger.info(f"Generated embedding for text: {text}")
            return embedding_vector
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None