"""
Pytest configuration for backend tests.
Adds the backend root to sys.path so 'app' module can be imported.
"""
import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
