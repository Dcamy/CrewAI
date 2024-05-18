# iChain/src/main.py 
import logging  # Import the standard Python logging module. Check logging_wrapper.py if this fails 🤯
import os  # Import the os module for operating system related operations. Check logging_wrapper.py if this fails 🤯
from datetime import datetime  # Import datetime for working with dates and times. Check logging_wrapper.py if this fails 🤯
from crewai import Agent, Crew 

# Correct the import paths to match the new file tree (removing the extra iChain directory)
from src.crew import Crew  # Import the Crew class from crew.py. Check crew.py for errors if this fails ❌
from src.config.llm_config import get_models  # Import the get_models function from llm_config.py. Check llm_config.py for errors if this fails ❌
from src.config import setup_logging # Import the setup_logging function from config.py. Check config.py for errors if this fails ❌

# --- HOW TO CODE WITH WOW FACTOR --- 
# 1. Descriptive Comments: EVERY line of code SHOULD have an inline comment, no matter how obvious. 
#    It's not just for YOU, it's for the AI and for the COMMUNITY. 
# 2. Import References:  When importing, tell us where to look if it breaks! 
# 3. Emojis: Use them wisely, like a sprinkle of magic. ✨  Don't overdo it! 
# 4. Revision Ideas: Always leave notes on how to make things BETTER.  Inspire the future! 

# Call this right at the beginning to configure logging 
setup_logging()  # Set up logging for the application 

# Set up logging to a file
log_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Logs") # Construct the path to the log directory relative to this script
os.makedirs(log_directory, exist_ok=True)  # Create the log directory if it doesn't exist
log_filename = os.path.join(log_directory, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")  # Construct the log filename with a timestamp
logging.basicConfig(  # Configure the basic logging settings
    level=logging.DEBUG,  # Set the logging level to DEBUG (capture all messages)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Define the format of log messages
    handlers=[
        logging.FileHandler(log_filename),  # Write log messages to the file 
        logging.StreamHandler()  # Print log messages to the console 
    ]
)

logger = logging.getLogger(__name__) # Create a logger for the current module

# Crew Initialization and Task Execution
def main():
    """
    The main function of the iChain application.
    It initializes the CrewAI crew, executes tasks, showcases error handling, 
    and is designed to produce extensive logging for AI training and analysis. 
    """
    logger.info("Initializing the Crew... 🚀") # Log the start of crew initialization

    # Initialize AI models 🤖🧠 
    logger.info("Loading AI models... ⏳")
    memory = True  # Memory imported from CrewAI (this is a placeholder, we'll replace it later)
    logger.debug("Memory object created: %s", memory) # Detailed debug log about memory 
    
    embedder = get_models("ollama", "llama-base")  # Initialize the embedding model 
    logger.info(f"Embedding model loaded: {embedder} ✅") # Log the specific model loaded
    logger.debug("Embedding model details: %s", embedder) # More detailed logging 

    llm_assistant = get_models("groq", "llama3-8b-8192")  # Initialize the assistant agent's LLM 
    logger.info(f"Assistant LLM loaded: {llm_assistant} ✅")
    logger.debug("Assistant LLM details: %s", llm_assistant)

    llm_cleaner = get_models("groq", "llama3-8b-8192")  # Initialize the cleaning agent's LLM 
    logger.info(f"Cleaner LLM loaded: {llm_cleaner} ✅")
    logger.debug("Cleaner LLM details: %s", llm_cleaner)

    llm_validator = get_models("groq", "llama3-8b-8192") # Initialize the validator agent's LLM 
    logger.info(f"Validator LLM loaded: {llm_validator} ✅")
    logger.debug("Validator LLM details: %s", llm_validator)
    
    logger.info("AI models loaded successfully. 🎉") 

    # Initialize the Crew with specific agent models
    logger.info("Initializing Crew with loaded models... 🏗️")
    crew = Crew(
        assistant_llm=llm_assistant,  # Assign the assistant agent's LLM
        cleaner_llm=llm_cleaner,  # Assign the cleaning agent's LLM
        validator_llm=llm_validator, # Assign the validator agent's LLM 
        embedder=embedder,  # Assign the embedding model
        memory=memory  # Assign the memory object 
    )
    logger.info("Crew initialized successfully.  Ready for action! 💪")

    # Example usage of Crew's capabilities (designed for logging and data collection)
    try:
        # Simulate a conversation task 💬
        user_input = "Hello, how can you help me today?"  
        logger.debug("User input: %s", user_input) # Log the user input
        conversation_response = crew.conversation_task.run(user_input)  
        logger.info(f"Conversation response: {conversation_response} 🗣️")  
        logger.debug("Conversation task details: [Add details here for training]") # Placeholder for more detailed task logging 

        # Simulate a data anonymization task 🕵️
        sample_data = "Contact me at example@example.com or call 123-456-7890." 
        logger.debug("Sample data before anonymization: %s", sample_data) 
        anonymized_data = crew.anonymization_task.run(sample_data)  
        logger.info(f"Anonymized Data: {anonymized_data} 🤫") 
        logger.debug("Anonymization task details: [Add details here for training]") 

        # Simulate data validation task  ✅❌
        logger.debug("Anonymized data to be validated: %s", anonymized_data)
        validation_result = crew.validation_task.run(anonymized_data)  
        logger.info(f"Data Validation Result: {validation_result} ") # Add emoji here based on result
        logger.debug("Validation task details: [Add details here for training]")

        # Simulate a data upload task 📤
        logger.debug("Data to be uploaded: %s", anonymized_data)
        upload_result = crew.data_upload_task.run({  
            "data": anonymized_data, 
            "metadata": {"source": "user_interaction", "type": "text"} 
        })
        logger.info(f"Upload Result: {upload_result} ☁️")  
        logger.debug("Upload task details: [Add details here for training]")

        # Simulate an image processing task 🖼️
        image_path = "path_to_your_image.jpg"  # Replace with a valid image path! 
        logger.debug("Image path for processing: %s", image_path) 
        extracted_text = crew.image_processing_task.run(image_path)  
        logger.info(f"Extracted Text from Image: {extracted_text} 🔍")  
        logger.debug("Image processing task details: [Add details here for training]") 

    except Exception as e:  # Catch any exceptions that occur during task execution 
        logger.error(f"An error occurred during task execution: {e} ❌🤯🤷‍♂️🤦‍♂️")  # Log the error with dramatic emojis 
        logger.exception("Exception details: ") # Log the full exception traceback 

    logger.info("Crew operations completed. Mission accomplished! 😎")


# --- REVISION IDEAS --- 🧠✨
# 1. Dynamic Emojis: Create a system to automatically add emojis to logs based on content (success, warnings, errors).
# 2. Custom Log Formatter: Build a formatter that injects emojis and rich formatting into log messages.
# 3. Visualizations:  Think about how to represent logged data visually (graphs, charts, etc.). 
# 4. Log Analysis Agent: Develop an agent that scans logs for patterns, errors, inefficiencies, and makes recommendations. 
# 5. Idea Agent: An agent that analyzes code and suggests improvements (automatically add to "Revision Ideas"). 

# Entry point for the script
if __name__ == "__main__": 
    main() # Call the main function 