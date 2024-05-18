# iChain/src/iChain/tools/google_drive.py

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import logging

# Setup logging configuration
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_directory, 'google_drive_tool_log.txt'),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class GoogleDriveTool:
    """
    This tool handles interactions with Google Drive, such as uploading files,
    creating folders, and managing permissions.
    """

    def __init__(self, token_file):
        """
        Initializes the Google Drive Tool with credentials to authenticate Google Drive API requests.

        Args:
            token_file (str): Path to the file containing OAuth 2.0 credentials.
        """
        self.service = self.authenticate_google_drive(token_file)
        logger.info("Google Drive Tool initialized with token file.")

    def authenticate_google_drive(self, token_file):
        """
        Authenticates and returns a Google Drive service object to interact with the API.

        Args:
            token_file (str): Path to the token file.

        Returns:
            Resource: The authenticated Google Drive service object.
        """
        creds = None
        try:
            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(
                    token_file, scopes=["https://www.googleapis.com/auth/drive"]
                )
                logger.info("Google Drive credentials loaded successfully.")
            else:
                raise FileNotFoundError("Token file not found. Please check your token path and try again.")
        except Exception as e:
            logger.error(f"Failed to authenticate Google Drive: {e}")
            raise e

        # Build the service from the credentials
        try:
            service = build("drive", "v3", credentials=creds)
            logger.info("Google Drive service built successfully.")
            return service
        except Exception as e:
            logger.error(f"Failed to build Google Drive service: {e}")
            raise e

    def upload_file(self, file_path, folder_id=None):
        """
        Uploads a file to Google Drive.

        Args:
            file_path (str): The path to the file to upload.
            folder_id (str, optional): The ID of the folder where the file will be uploaded (None if root). Defaults to None.

        Returns:
            str: The ID of the uploaded file.
        """
        try:
            file_metadata = {
                "name": os.path.basename(file_path),
                "parents": [folder_id] if folder_id else [],
            }
            media = MediaFileUpload(file_path, mimetype="text/plain")
            file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )
            file_id = file.get("id")
            logger.info(f"File '{file_path}' uploaded successfully with ID: {file_id}")
            return file_id
        except Exception as e:
            logger.error(f"Failed to upload file '{file_path}': {e}")
            return None