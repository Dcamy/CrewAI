# iChain/tests/test_google_drive.py
import unittest
from unittest.mock import patch, MagicMock
from iChain.src.iChain.tools.google_drive import GoogleDriveTool
import os

class TestGoogleDriveTool(unittest.TestCase):
    """
    This class contains tests for the GoogleDriveTool.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        self.token_file = "path_to_token.json"
        self.google_drive_tool = GoogleDriveTool(self.token_file)

    @patch('iChain.src.iChain.tools.google_drive.Credentials.from_authorized_user_file')
    @patch('iChain.src.iChain.tools.google_drive.build')
    def test_authenticate_google_drive(self, mock_build, mock_creds):
        """
        Test the authenticate_google_drive method of the GoogleDriveTool.
        """
        mock_creds.return_value = MagicMock()
        mock_build.return_value = MagicMock()
        service = self.google_drive_tool.authenticate_google_drive(self.token_file)
        self.assertIsNotNone(service)
        mock_creds.assert_called_once_with(self.token_file, scopes=["https://www.googleapis.com/auth/drive"])
        mock_build.assert_called_once_with("drive", "v3", credentials=mock_creds.return_value)

    @patch('iChain.src.iChain.tools.google_drive.MediaFileUpload')
    def test_upload_file(self, mock_media_file_upload):
        """
        Test the upload_file method of the GoogleDriveTool.
        """
        self.google_drive_tool.service = MagicMock()
        mock_media_file_upload.return_value = MagicMock()
        file_id = "file_id_123"
        self.google_drive_tool.service.files().create().execute.return_value = {"id": file_id}
        result = self.google_drive_tool.upload_file("sample_data.txt")
        self.assertEqual(result, file_id)
        self.google_drive_tool.service.files().create.assert_called_once()

    @patch('os.path.exists')
    @patch('iChain.src.iChain.tools.google_drive.Credentials.from_authorized_user_file')
    @patch('iChain.src.iChain.tools.google_drive.build')
    def test_token_file_not_found(self, mock_build, mock_creds, mock_path_exists):
        """
        Test the GoogleDriveTool initialization with a non-existing token file.
        """
        mock_path_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            GoogleDriveTool(self.token_file)