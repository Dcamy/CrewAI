# iChain/src/iChain/agents/vision.py

from crewai import Agent
import cv2
import pytesseract
import os
import datetime
from src.iChain.config.llm_config import get_models
import logging

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'vision_log.txt'), level=logging.INFO)

class VisionAgent(Agent):
    """
    This class defines the Vision Agent responsible for processing images. It uses computer vision
    techniques to extract text and other relevant information from images provided by users.
    """

    def __init__(self, memory, llm):
        """
        Initializes the Vision Agent with specific roles and goals.

        Args:
            memory (object): A memory object capable of storing session data (imported from CrewAI).
            llm (object): The language model the agent will use for generating responses.
        """
        super().__init__(role="Vision Specialist", verbose=True, memory=memory, llm=llm)

    def process_image(self, image_path):
        """
        Processes an image file to extract text using Optical Character Recognition (OCR).

        Args:
            image_path (str): The path to the image file to be processed.

        Returns:
            str: The extracted text from the image.
        """
        try:
            # Load the image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Image not found at path: {image_path}")

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply OCR to extract text
            extracted_text = pytesseract.image_to_string(gray)
            logging.info(f"[Agent: {self.role}] [Task: process_image] [Tool: pytesseract] Extracted Text: {extracted_text[:50]}...")  # Log with agent tag
            return extracted_text

        except Exception as e:
            logging.error(f"[Agent: {self.role}] [Task: process_image] Error processing image: {e}")
            return "Error processing image"

    def log_image_processing(self, image_path, extracted_text):
        """
        Logs the processing of the image and the extracted text for audit and improvements.

        Args:
            image_path (str): The path to the image processed.
            extracted_text (str): The text extracted from the image.
        """
        try:
            log_message = (
                f"[Agent: {self.role}] [Task: log_image_processing] Image processed: {image_path}\n"
                f"Extracted Text: {extracted_text}\nLogged at: {datetime.datetime.now()}\n---\n"
            )
            logging.info(log_message)
        except Exception as e:
            logging.error(f"[Agent: {self.role}] [Task: log_image_processing] Error logging image processing: {e}")