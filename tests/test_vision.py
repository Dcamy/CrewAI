# iChain/tests/test_vision.py
# iChain/tests/test_vision.py

import unittest
from unittest.mock import patch, MagicMock
from ..src.iChain.agents.vision import VisionAgent
from ..src.iChain.config.llm_config import get_models
import cv2
import numpy as np

class TestVisionAgent(unittest.TestCase):
    """
    This class contains tests for the VisionAgent.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        memory = True  # Mocked memory object
        self.llm = get_models("ollama", "llava")
        self.agent = VisionAgent(memory, self.llm)

    @patch('cv2.imread')
    @patch('pytesseract.image_to_string')
    def test_process_image_success(self, mock_image_to_string, mock_imread):
        """
        Test that the agent correctly processes an image and extracts text.
        """
        # Mocking OpenCV and pytesseract functions
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)  # Mocked image data
        mock_image_to_string.return_value = "Extracted text from image."

        image_path = "dummy_path.jpg"
        extracted_text = self.agent.process_image(image_path)
        self.assertEqual(extracted_text, "Extracted text from image.")

    @patch('cv2.imread')
    def test_process_image_not_found(self, mock_imread):
        """
        Test that the agent handles image not found error.
        """
        mock_imread.return_value = None  # Mock image not found

        image_path = "dummy_path.jpg"
        extracted_text = self.agent.process_image(image_path)
        self.assertEqual(extracted_text, "Error processing image")

    @patch('cv2.imread')
    @patch('pytesseract.image_to_string')
    def test_process_image_exception(self, mock_image_to_string, mock_imread):
        """
        Test that the agent handles exceptions during image processing.
        """
        mock_imread.side_effect = Exception("Test error")  # Mock an exception

        image_path = "dummy_path.jpg"
        extracted_text = self.agent.process_image(image_path)
        self.assertEqual(extracted_text, "Error processing image")

    @patch('logging.info')
    def test_log_image_processing(self, mock_logging_info):
        """
        Test the logging of image processing.
        """
        image_path = "dummy_path.jpg"
        extracted_text = "Extracted text from image."

        self.agent.log_image_processing(image_path, extracted_text)
        mock_logging_info.assert_called()