# iChain/tests/test_memory_long_term.py
import unittest
import os
from iChain.src.iChain.memory.long_term import LongTermMemory

class TestLongTermMemory(unittest.TestCase):
    """
    This class contains tests for the LongTermMemory class.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        # Define the path for the test storage file
        self.test_storage_path = "C:\\CrewAI\\iChain\\tests\\ltm_test_storage.txt"
        # Initialize LongTermMemory with the test storage file
        self.ltm = LongTermMemory(self.test_storage_path)

        # Ensure the test storage file is empty before each test
        with open(self.test_storage_path, "w") as file:
            file.write("")

    def tearDown(self):
        """
        Clean up environment after each test.
        """
        # Remove the test storage file after each test
        if os.path.exists(self.test_storage_path):
            os.remove(self.test_storage_path)

    def test_save_memory(self):
        """
        Test that a piece of memory can be saved correctly.
        """
        key = "test_key"
        value = "test_value"
        self.ltm.save_memory(key, value)
        
        # Verify the content of the storage file
        with open(self.test_storage_path, "r") as file:
            lines = file.readlines()
        
        self.assertEqual(lines[0].strip(), f"{key}:{value}")

    def test_retrieve_memory(self):
        """
        Test that a piece of memory can be retrieved correctly.
        """
        key = "test_key"
        value = "test_value"
        self.ltm.save_memory(key, value)
        
        retrieved_value = self.ltm.retrieve_memory(key)
        self.assertEqual(retrieved_value, value)

    def test_retrieve_memory_not_found(self):
        """
        Test that retrieving a non-existent key returns None.
        """
        retrieved_value = self.ltm.retrieve_memory("non_existent_key")
        self.assertIsNone(retrieved_value)