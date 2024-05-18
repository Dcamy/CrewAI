# iChain/src/iChain/tasks/__init__.py

from .conversation import ConversationTask
from .anonymization import AnonymizationTask
from .data_upload import DataUploadTask
from .image_processing import ImageProcessingTask

# This allows other parts of the application to import task classes directly from the tasks package
__all__ = [
    "ConversationTask",
    "AnonymizationTask",
    "DataUploadTask",
    "ImageProcessingTask",
]