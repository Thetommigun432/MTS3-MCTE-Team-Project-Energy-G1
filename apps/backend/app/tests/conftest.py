"""
Pytest configuration.
"""

import sys
from pathlib import Path

import pytest

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
