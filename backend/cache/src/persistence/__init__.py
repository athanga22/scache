"""
Persistence Layer for Cache System
Handles snapshot-based persistence and recovery.
"""

from .persistence import PersistenceManager

__all__ = ['PersistenceManager']
