# iChain/src/iChain/memory/__init__.py

from .long_term import LongTermMemory
from .short_term import ShortTermMemory

# This allows other parts of the application to import memory classes directly from the memory package
__all__ = ["LongTermMemory", "ShortTermMemory"]