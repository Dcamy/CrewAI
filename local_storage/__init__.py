# iChain/src/local_storage/__init__.py

from .file_handler import FileHandler

# This allows other parts of the application to import the FileHandler class directly from the local_storage package
__all__ = ["FileHandler"]