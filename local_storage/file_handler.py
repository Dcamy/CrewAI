# iChain/src/local_storage/file_handler.py
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_directory, 'file_handler_log.txt'),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FileHandler:
    """
    Handles file operations in a specified directory. Monitors for new files and processes them
    as needed for further CrewAI tasks.
    """

    def __init__(self, directory):
        """
        Initializes the FileHandler with a directory to monitor.

        Args:
            directory (str): The path to the directory to monitor for new files.
        """
        self.directory = directory
        self.observer = Observer()

    def start(self):
        """
        Starts monitoring the directory for new files and handling them accordingly.
        """
        event_handler = FileEventHandler(self.process_file)
        self.observer.schedule(event_handler, self.directory, recursive=True)
        self.observer.start()
        logger.info(f"Monitoring {self.directory} for new files.")
        print(f"Monitoring {self.directory} for new files.")

    def stop(self):
        """
        Stops the directory monitoring.
        """
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped monitoring the directory.")

    def process_file(self, file_path):
        """
        Process the file that has been added to the directory.

        Args:
            file_path (str): Path to the newly added file.
        """
        logger.info(f"Processing file: {file_path}")
        # Implement the file processing logic here
        # This could include parsing, data extraction, or triggering other CrewAI tasks
        print(f"Processing file: {file_path}")


class FileEventHandler(FileSystemEventHandler):
    """
    Event handler for the file system events that triggers processing of new files.
    """

    def __init__(self, process_file_callback):
        self.process_file_callback = process_file_callback

    def on_created(self, event):
        """
        Called when a file or directory is created.

        Args:
            event (Event): The event object representing the file system event.
        """
        if not event.is_directory:
            self.process_file_callback(event.src_path)