"""
Tests for Market Data module
Run with: pytest tests/test_market_data.py -v
"""

import pytest
from datetime import datetime
from src.market_data import (
    Tick,
    Candle,
    MockDataConnector,
    OptionsChain,
)


class TestTick:
    """Test Tick data structure"""

    def test_tick_creation(self):
        """Test creating a tick"""
        tick = Tick(
            symbol="SPY",
            timestamp=datetime.utcnow(),
            price=450.0,
            size=100,
            bid=449.95,
            ask=450.05,
        )
        assert tick.symbol == "SPY"
        assert tick.price == 450.0


class TestCandle:
    """Test Candle data structure"""

    def test_candle_creation(self):
        """Test creating a candle"""
        candle = Candle(
            symbol="SPY",
            timestamp=datetime.utcnow(),
            open=450.0,
            high=451.0,
            low=449.0,
            close=450.5,
            volume=1000000,
            timeframe="1m",
        )
        assert candle.symbol == "SPY"
        assert candle.timeframe == "1m"


class TestMockDataConnector:
    """Test mock data connector"""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connecting/disconnecting"""
        connector = MockDataConnector()
        connected = await connector.connect()
        assert connected is True
        assert connector.is_connected is True

        await connector.disconnect()
        assert connector.is_connected is False

    @pytest.mark.asyncio
    async def test_get_historical_ticks(self):
        """Test getting historical ticks"""
        connector = MockDataConnector()
        await connector.connect()

        ticks = await connector.get_historical_ticks(
            symbol="SPY",
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 31),
        )

        assert isinstance(ticks, list)


class TestOptionsChain:
    """Test options contract"""

    def test_options_chain(self):
        """Test creating an options chain"""
        chain = OptionsChain(
            symbol="SPY",
            expiry="2024-01-19",
            strike=450.0,
            option_type="CALL",
            bid=5.0,
            ask=5.10,
        )
        assert chain.symbol == "SPY"
        assert chain.option_type == "CALL"
