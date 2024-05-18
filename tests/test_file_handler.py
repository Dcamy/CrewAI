# iChain/tests/test_file_handler.py
import unittest
from unittest.mock import patch, MagicMock
from iChain.src.local_storage.file_handler import FileHandler, FileEventHandler
import os

class TestFileHandler(unittest.TestCase):
    """
    This class contains tests for the FileHandler.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        self.directory = "C:/CrewAI/iChain/WatchedDirectory"
        self.file_handler = FileHandler(self.directory)

    @patch('iChain.src.local_storage.file_handler.Observer')
    def test_start(self, MockObserver):
        """
        Test the start method of the FileHandler.
        """
        mock_observer_instance = MockObserver.return_value
        self.file_handler.start()
        mock_observer_instance.schedule.assert_called_once()
        mock_observer_instance.start.assert_called_once()

    @patch('iChain.src.local_storage.file_handler.Observer')
    def test_stop(self, MockObserver):
        """
        Test the stop method of the FileHandler.
        """
        mock_observer_instance = MockObserver.return_value
        self.file_handler.start()
        self.file_handler.stop()
        mock_observer_instance.stop.assert_called_once()
        mock_observer_instance.join.assert_called_once()

    @patch('iChain.src.local_storage.file_handler.logger')
    def test_process_file(self, mock_logger):
        """
        Test the process_file method of the FileHandler.
        """
        test_file_path = "C:/CrewAI/iChain/WatchedDirectory/test_file.txt"
        self.file_handler.process_file(test_file_path)
        mock_logger.info.assert_called_with(f"Processing file: {test_file_path}")

    def test_file_event_handler_on_created(self):
        """
        Test the on_created method of the FileEventHandler.
        """
        process_file_callback = MagicMock()
        event_handler = FileEventHandler(process_file_callback)
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "C:/CrewAI/iChain/WatchedDirectory/test_file.txt"

        event_handler.on_created(mock_event)
        process_file_callback.assert_called_with(mock_event.src_path)