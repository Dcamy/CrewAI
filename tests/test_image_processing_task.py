# iChain/tests/test_image_processing_task.py
import unittest
from unittest.mock import MagicMock, patch
from iChain.src.iChain.tasks.image_processing import ImageProcessingTask
from iChain.src.iChain.agents.vision import VisionAgent
from src.iChain.config.llm_config import get_models

class TestImageProcessingTask(unittest.TestCase):
    """
    This class contains tests for the ImageProcessingTask.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        # Mock VisionAgent
        self.vision_agent = MagicMock(spec=VisionAgent)
        # Mock memory and llm
        self.memory = MagicMock()
        self.llm = MagicMock()

        # Initialize ImageProcessingTask with mocked VisionAgent, memory, and llm
        self.task = ImageProcessingTask(self.vision_agent)

    def test_run_with_valid_image_path(self):
        """
        Test the run method with a valid image path.
        """
        # Mock the process_image method
        self.vision_agent.process_image.return_value = "Extracted text from image"
        
        image_path = "valid_image_path.jpg"
        result = self.task.run(image_path)
        
        self.assertEqual(result, "Extracted text from image")
        self.vision_agent.process_image.assert_called_once_with(image_path)

    def test_run_with_no_image_path(self):
        """
        Test the run method with no image path provided.
        """
        with self.assertRaises(ValueError):
            self.task.run("")

    def test_run_with_error_processing_image(self):
        """
        Test the run method when an error occurs during image processing.
        """
        self.vision_agent.process_image.side_effect = Exception("Processing error")
        
        image_path = "error_image_path.jpg"
        result = self.task.run(image_path)
        
        self.assertEqual(result, "Error processing image.")
        self.vision_agent.process_image.assert_called_once_with(image_path)