# iChain/src/iChain/tasks/image_processing.py

from crewai import Task
import cv2
import pytesseract
import os
import datetime
from src.iChain.config.llm_config import get_models
import logging

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'image_processing_log.txt'), level=logging.INFO)

class ImageProcessingTask(Task):
    """
    This task handles the processing of images, utilizing computer vision techniques
    to extract information such as text and other relevant data from images provided by users.
    """

    def __init__(self, vision_agent):
        """
        Initializes the Image Processing Task with references to necessary tools and agents.

        Args:
            vision_agent (object): The agent responsible for performing image processing tasks.
        """
        super().__init__()
        self.vision_agent = vision_agent

    def run(self, image_path):
        """
        Executes the image processing task by applying OCR to extract text from the image.

        Args:
            image_path (str): The path to the image file to be processed.

        Returns:
            str: The text extracted from the image.
        """
        if not image_path:
            raise ValueError("No image path provided for processing.")
        
        logging.info(f"[ImageProcessingTask] Processing image: {image_path}")

        try:
            # Use the vision agent to process the image and extract text
            extracted_text = self.vision_agent.process_image(image_path)
            logging.info(f"[ImageProcessingTask] Extracted text: {extracted_text}")
            return extracted_text
        except Exception as e:
            logging.error(f"[ImageProcessingTask] Error processing image: {e}")
            return "Error processing image."