# iChain/src/iChain/agents/valuator.py

from crewai import Agent
import os
import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from src.iChain.config.llm_config import get_models
import logging

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
log_directory = "C:\\CrewAI\\iChain\\Logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, 'valuation_log.txt'), level=logging.INFO)

class DataValuationAgent(Agent):
    """
    This class defines the Data Valuation Agent responsible for assessing the quality and value
    of anonymized data. It determines the potential market value based on content richness and relevance.
    """

    def __init__(self, memory, llm):
        """
        Initializes the Data Valuation Agent with specific roles and goals.

        Args:
            memory (object): A memory object capable of storing session data (part of the CrewAI library).
            llm (object): The language model the agent will use for generating responses.
        """
        super().__init__(role="Data Valuator", verbose=True, memory=memory, llm=llm)

    def assess_data(self, data):
        """
        Assesses the value of the anonymized data based on predefined criteria such as completeness,
        relevance, and potential demand in the market.

        Args:
            data (str): The anonymized data to be assessed.

        Returns:
            tuple: A tuple containing the assessed value and a descriptive assessment.
        """
        try:
            value, assessment = self.evaluate_data_quality(data)
            self.log_assessment(data, value, assessment)
            return value, assessment
        except Exception as e:
            logging.error(f"Error assessing data: {e}")
            return None, "Error assessing data"

    def evaluate_data_quality(self, data):
        """
        Evaluates the quality of the data by analyzing the content using NLP and the LLM.

        Args:
            data (str): The data to evaluate.

        Returns:
            tuple: The numeric value of the data and a string describing the assessment.
        """
        try:
            # Tokenize the data
            tokens = word_tokenize(data)

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]

            # Calculate the frequency of each token
            freq = nltk.FreqDist(tokens)

            # Calculate the average frequency of the top 10 most common tokens
            # This helps in determining the richness and diversity of the content.
            top_10_tokens = [token for token, freq in freq.most_common(10)]
            avg_freq = sum([freq[token] for token in top_10_tokens]) / len(top_10_tokens)

            # Use the LLM to generate a response
            response = self.llm.generate_response(data)

            # Analyze the response to determine the quality of the data
            if "REDACTED" not in response and avg_freq > 0.5:
                return 100, "High value - contains rich and relevant content"
            else:
                return 50, "Medium value - partially relevant content"
        except Exception as e:
            logging.error(f"Error evaluating data quality: {e}. Data: {data}")
            return None, "Error evaluating data quality"

    def log_assessment(self, data, value, assessment):
        """
        Logs the assessment of the data for audit and improvements.

        Args:
            data (str): The data assessed.
            value (int): The numerical value assigned to the data.
            assessment (str): The descriptive assessment of the data.
        """
        log_message = f"Data: {data}\nValue: {value}\nAssessment: {assessment}\nLogged at: {datetime.datetime.now()}\n---\n"
        logging.info(log_message)