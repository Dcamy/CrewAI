# iChain/src/iChain/models/data_models.py

import logging
import os
import datetime

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'data_models_log.txt'), level=logging.INFO)

class UserDataModel:
    """
    Data model for storing user-specific information.
    """

    def __init__(self, user_id, name, email):
        """
        Initialize the UserDataModel with user-specific details.

        Args:
            user_id (str): Unique identifier for the user.
            name (str): Name of the user.
            email (str): Email address of the user.
        """
        self.user_id = user_id
        self.name = name
        self.email = email
        logging.info(f"[Model: UserDataModel] [Task: __init__] User initialized: {self}")

    def __repr__(self):
        return f"<UserDataModel(user_id={self.user_id}, name={self.name}, email={self.email})>"


class InteractionDataModel:
    """
    Data model for storing details about interactions with users.
    """

    def __init__(self, interaction_id, user_id, interaction_type, details):
        """
        Initialize the InteractionDataModel with interaction details.

        Args:
            interaction_id (str): Unique identifier for the interaction.
            user_id (str): Unique identifier for the user.
            interaction_type (str): Type of the interaction.
            details (str): Details about the interaction.
        """
        self.interaction_id = interaction_id
        self.user_id = user_id
        self.interaction_type = interaction_type
        self.details = details
        logging.info(f"[Model: InteractionDataModel] [Task: __init__] Interaction initialized: {self}")

    def __repr__(self):
        return (
            f"<InteractionDataModel(interaction_id={self.interaction_id}, user_id={self.user_id}, "
            f"interaction_type={self.interaction_type}, details={self.details})>"
        )


class ImageDataModel:
    """
    Data model for storing information about processed images.
    """

    def __init__(self, image_id, user_id, file_path, extracted_text):
        """
        Initialize the ImageDataModel with image processing details.

        Args:
            image_id (str): Unique identifier for the image.
            user_id (str): Unique identifier for the user.
            file_path (str): File path of the processed image.
            extracted_text (str): Text extracted from the image.
        """
        self.image_id = image_id
        self.user_id = user_id
        self.file_path = file_path
        self.extracted_text = extracted_text
        logging.info(f"[Model: ImageDataModel] [Task: __init__] Image initialized: {self}")

    def __repr__(self):
        return (
            f"<ImageDataModel(image_id={self.image_id}, user_id={self.user_id}, "
            f"file_path={self.file_path}, extracted_text={self.extracted_text})>"
        )