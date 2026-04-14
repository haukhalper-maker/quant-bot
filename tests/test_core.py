"""
Unit tests for Core module
Run with: pytest tests/test_core.py -v
"""

import pytest
import asyncio
from datetime import datetime
from src.core import EventLoop, EventBus, Event, EventType


class TestEventBus:
    """Test event pub/sub system"""

    def test_subscribe_handler(self):
        """Test subscribing a handler"""
        bus = EventBus()
        called = []

        def handler(event):
            called.append(event)

        bus.subscribe(EventType.TICK, handler)
        assert EventType.TICK in bus._subscribers

    @pytest.mark.asyncio
    async def test_publish_event(self):
        """Test publishing an event"""
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(EventType.TICK, handler)
        event = Event(
            event_type=EventType.TICK,
            timestamp=datetime.utcnow(),
            data={"price": 100.0},
        )

        await bus.publish(event)
        await asyncio.sleep(0.1)
        assert len(received) == 1


class TestEventLoop:
    """Test main event loop"""

    @pytest.mark.asyncio
    async def test_loop_start_stop(self):
        """Test starting and stopping event loop"""
        loop = EventLoop("test_loop")
        
        # Start loop in background
        loop_task = asyncio.create_task(loop.start())
        await asyncio.sleep(0.1)
        
        # Should be running
        assert loop._running is True
        
        # Stop loop
        await loop.stop()
        await asyncio.sleep(0.1)
        assert loop._running is False


# ============================================================================
# MARKERS FOR INTEGRATION TESTS
# ============================================================================

pytestmark = pytest.mark.asyncio
