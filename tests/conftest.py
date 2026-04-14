"""
Pytest configuration file
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async (requires pytest-asyncio)"
    )


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
