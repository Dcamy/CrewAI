# iChain/tests/test_memory_short_term.py
# iChain/tests/test_short_term_memory.py

import unittest
from iChain.src.iChain.memory.short_term import ShortTermMemory


class TestShortTermMemory(unittest.TestCase):
    """
    This class contains tests for the ShortTermMemory class.
    """

    def setUp(self):
        """
        Set up environment before each test.
        """
        self.memory = ShortTermMemory(max_size=3)

    def test_store_memory(self):
        """
        Test that memory is correctly stored.
        """
        self.memory.store_memory("key1", "value1")
        self.memory.store_memory("key2", "value2")
        self.assertEqual(self.memory.retrieve_memory("key1"), "value1")
        self.assertEqual(self.memory.retrieve_memory("key2"), "value2")

    def test_retrieve_memory_not_found(self):
        """
        Test that retrieving a non-existent memory returns None.
        """
        self.assertIsNone(self.memory.retrieve_memory("non_existent_key"))

    def test_memory_overflow(self):
        """
        Test that memory overflow behaves correctly (oldest memory is discarded).
        """
        self.memory.store_memory("key1", "value1")
        self.memory.store_memory("key2", "value2")
        self.memory.store_memory("key3", "value3")
        self.memory.store_memory("key4", "value4")  # This should push out "key1"
        self.assertIsNone(self.memory.retrieve_memory("key1"))
        self.assertEqual(self.memory.retrieve_memory("key2"), "value2")
        self.assertEqual(self.memory.retrieve_memory("key3"), "value3")
        self.assertEqual(self.memory.retrieve_memory("key4"), "value4")

    def test_clear_memory(self):
        """
        Test that clearing the memory works correctly.
        """
        self.memory.store_memory("key1", "value1")
        self.memory.store_memory("key2", "value2")
        self.memory.clear_memory()
        self.assertIsNone(self.memory.retrieve_memory("key1"))
        self.assertIsNone(self.memory.retrieve_memory("key2"))