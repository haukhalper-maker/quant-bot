"""
Market Data Layer - Data ingestion, normalization, storage
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from loguru import logger


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class Tick:
    """Single market tick (quote or trade)"""

    symbol: str
    timestamp: datetime
    price: float
    size: int
    bid: float = 0.0
    bid_size: int = 0
    ask: float = 0.0
    ask_size: int = 0
    tick_type: str = "trade"  # 'trade', 'bid', 'ask', 'mid'
    open_interest: int = 0  # For options
    option_expiry: Optional[str] = None  # For options (YYYY-MM-DD)

    def __post_init__(self):
        if self.bid == 0 and self.ask == 0:
            self.bid = self.ask = self.price


@dataclass
class Candle:
    """OHLCV candle"""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "1m"  # '1m', '5m', '15m', '1h', '1d'
    tick_count: int = 0


@dataclass
class Trade:
    """Executed trade record"""

    trade_id: str
    symbol: str
    timestamp: datetime
    side: str  # 'BUY', 'SELL'
    quantity: int
    price: float
    filled_quantity: int
    status: str  # 'PENDING', 'FILLED', 'REJECTED'
    pnl: float = 0.0


@dataclass
class OptionsChain:
    """Options contract specification"""

    symbol: str  # e.g., SPY
    expiry: str  # YYYY-MM-DD
    strike: float
    option_type: str  # 'CALL', 'PUT'
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0
    open_interest: int = 0
    volume: int = 0
    impliedVol: float = 0.0


# ============================================================================
# Data Source Interface
# ============================================================================


class DataConnector(ABC):
    """Abstract base for data sources"""

    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        logger.info(f"DataConnector '{name}' initialized")

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source"""
        pass

    @abstractmethod
    async def subscribe_ticks(self, symbol: str, callback) -> None:
        """Subscribe to real-time ticks"""
        pass

    @abstractmethod
    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        """Get historical tick data"""
        pass

    @abstractmethod
    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        """Get available options contracts"""
        pass


# ============================================================================
# MOCK DATA CONNECTOR (for testing & development)
# ============================================================================


class MockDataConnector(DataConnector):
    """Mock data source for backtesting without API"""

    def __init__(self):
        super().__init__("MockConnector")
        self.ticks: Dict[str, List[Tick]] = {}

    async def connect(self) -> bool:
        """Mock connection"""
        self.is_connected = True
        logger.info("Mock connector connected")
        return True

    async def disconnect(self) -> None:
        """Mock disconnection"""
        self.is_connected = False
        logger.info("Mock connector disconnected")

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        """Mock tick subscription"""
        logger.debug(f"Mock subscribed to ticks for {symbol}")

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        """Return empty list (to be overridden in tests)"""
        return []

    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        """Return empty chain (to be overridden in tests)"""
        return []


# ============================================================================
# REAL DATA CONNECTORS (Stubs for integration)
# ============================================================================


class InteractiveBrokersConnector(DataConnector):
    """
    [API: Interactive Brokers TWS]
    To implement: Use ibapi library, connect to TWS running locally
    Docs: https://github.com/InteractiveBrokers/tws-api
    """

    async def connect(self) -> bool:
        raise NotImplementedError("[API: Interactive Brokers] not yet implemented")

    async def disconnect(self) -> None:
        pass

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        raise NotImplementedError()

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        raise NotImplementedError()

    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        raise NotImplementedError()


class PolygonConnector(DataConnector):
    """
    [API: Polygon.io Real-time & Historical Data]
    To implement: Use polygon-api-client library
    Docs: https://polygon.io/docs/options/getting-started
    """

    async def connect(self) -> bool:
        raise NotImplementedError("[API: Polygon.io] not yet implemented")

    async def disconnect(self) -> None:
        pass

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        raise NotImplementedError()

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        raise NotImplementedError()

    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        raise NotImplementedError()


class AlpacaConnector(DataConnector):
    """
    [API: Alpaca Trade API]
    To implement: Use alpaca-trade-api library
    Docs: https://alpaca.markets/docs/api-references/
    """

    async def connect(self) -> bool:
        raise NotImplementedError("[API: Alpaca] not yet implemented")

    async def disconnect(self) -> None:
        pass

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        raise NotImplementedError()

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        raise NotImplementedError()

    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        raise NotImplementedError()


class TastytradeConnector(DataConnector):
    """
    [API: Tastytrade API]
    To implement: Use tastytrade API for options data
    Docs: https://developer.tastytrade.com/
    """

    def __init__(self, username: str = None, password: str = None):
        super().__init__("TastytradeConnector")
        import os
        self.username = username or os.getenv("TASTYTRADE_USERNAME", "")
        self.password = password or os.getenv("TASTYTRADE_PASSWORD", "")
        self.base_url = "https://api.tastytrade.com"
        self.session_token = None

    async def connect(self) -> bool:
        """Authenticate with Tastytrade API"""
        try:
            # [API: Implement Tastytrade authentication]
            # POST /sessions
            # Body: {"login": api_key, "password": api_secret}
            logger.info("Tastytrade connector connected (stub)")
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"Tastytrade connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Tastytrade"""
        # [API: Implement logout if needed]
        self.is_connected = False
        logger.info("Tastytrade connector disconnected")

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        """Subscribe to real-time ticks (WebSocket)"""
        # [API: Implement WebSocket subscription for real-time data]
        logger.debug(f"Tastytrade subscribed to ticks for {symbol} (stub)")

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        """Get historical tick data"""
        # [API: Implement historical tick data retrieval]
        # GET /market-data/historical/{symbol}/ticks
        # Params: start_time, end_time, interval
        logger.debug(f"Fetching historical ticks for {symbol} from {start_time} to {end_time} (stub)")
        return []  # Return empty for now

    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        """Get options chain for symbol and expiry"""
        # [API: Implement options chain retrieval]
        # GET /instruments/equity-options
        # Params: underlying_symbol, expiration_date
        logger.debug(f"Fetching options chain for {symbol} expiry {expiry} (stub)")
        return []  # Return empty for now
