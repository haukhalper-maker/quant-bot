"""
Core event-driven architecture
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load .env from project root (two levels up from this file)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ============================================================================
# EVENT SYSTEM
# ============================================================================


class EventType(Enum):
    """All possible event types in the system"""

    # Market Data Events
    TICK = "tick"
    CANDLE = "candle"
    TRADE = "trade"
    QUOTE = "quote"

    # Analysis Events
    SIGNAL_GENERATED = "signal_generated"
    PATTERN_DETECTED = "pattern_detected"
    GREEK_UPDATE = "greek_update"

    # Strategy Events
    ORDER_SIGNAL = "order_signal"
    CANCEL_SIGNAL = "cancel_signal"

    # Execution Events
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    EXECUTION_ERROR = "execution_error"

    # Risk Events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    POSITION_UPDATE = "position_update"

    # System Events
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    ERROR = "error"


@dataclass
class Event:
    """Base event class"""

    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    priority: int = 0  # 0=normal, 1=high, 2=critical

    def __lt__(self, other):
        # For priority queue ordering
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.timestamp < other.timestamp


# ============================================================================
# EVENT BUS
# ============================================================================


class EventBus:
    """Pub/sub event bus for decoupled communication"""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        logger.info("EventBus initialized")

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe a handler to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe a handler from an event type"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed {handler.__name__} from {event_type.value}")

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers (async)"""
        if event.event_type not in self._subscribers:
            return

        logger.debug(
            f"Publishing {event.event_type.value} at {event.timestamp}",
            extra={"source": event.source},
        )

        handlers = self._subscribers[event.event_type]
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                handler(event)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# ============================================================================
# STATE MANAGEMENT
# ============================================================================


class State(Enum):
    """System state machine states"""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# ============================================================================
# EVENT LOOP (CORE ENGINE)
# ============================================================================


class EventLoop:
    """
    Core event-driven loop
    Processes market events, generates signals, manages execution
    """

    def __init__(self, name: str = "QuantBot"):
        self.name = name
        self.state = State.INITIALIZED
        self.event_bus = EventBus()
        self.event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._start_time: Optional[datetime] = None
        logger.info(f"EventLoop '{name}' initialized")

    async def start(self) -> None:
        """Start the event loop"""
        self.state = State.RUNNING
        self._running = True
        self._start_time = datetime.utcnow()
        logger.info(f"EventLoop '{self.name}' started at {self._start_time}")

        await self.event_bus.publish(
            Event(
                event_type=EventType.STARTUP,
                timestamp=datetime.utcnow(),
                source=self.name,
            )
        )

        # Main event processing loop
        while self._running:
            try:
                # Get next event (with timeout to check _running flag)
                try:
                    _, event = await asyncio.wait_for(
                        self.event_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process event
                await self._process_event(event)

            except Exception as e:
                logger.error(f"Error in event loop: {e}")
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.ERROR,
                        timestamp=datetime.utcnow(),
                        data={"error": str(e)},
                        source=self.name,
                    )
                )

    async def stop(self) -> None:
        """Stop the event loop gracefully"""
        self._running = False
        self.state = State.STOPPED
        logger.info(f"EventLoop '{self.name}' stopping")

        await self.event_bus.publish(
            Event(
                event_type=EventType.SHUTDOWN,
                timestamp=datetime.utcnow(),
                source=self.name,
            )
        )

    def pause(self) -> None:
        """Pause event processing"""
        self.state = State.PAUSED
        logger.info(f"EventLoop '{self.name}' paused")

    def resume(self) -> None:
        """Resume event processing"""
        self.state = State.RUNNING
        logger.info(f"EventLoop '{self.name}' resumed")

    async def post_event(self, event: Event) -> None:
        """Post an event to the queue"""
        await self.event_queue.put((event.priority, event))

    async def _process_event(self, event: Event) -> None:
        """Process a single event through the event bus"""
        logger.debug(
            f"Processing {event.event_type.value} from {event.source}",
            extra={"timestamp": event.timestamp},
        )
        await self.event_bus.publish(event)

    def get_uptime(self) -> Optional[float]:
        """Get uptime in seconds"""
        if self._start_time is None:
            return None
        return (datetime.utcnow() - self._start_time).total_seconds()


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class BotConfig:
    """Main bot configuration — secrets pulled from .env automatically"""

    # API credentials (from .env)
    tastytrade_username: str = field(default_factory=lambda: os.getenv("TASTYTRADE_USERNAME", ""))
    tastytrade_password: str = field(default_factory=lambda: os.getenv("TASTYTRADE_PASSWORD", ""))
    ib_account: str = field(default_factory=lambda: os.getenv("IB_ACCOUNT", ""))
    ib_port: int = field(default_factory=lambda: int(os.getenv("IB_PORT", "7497")))
    polygon_api_key: str = field(default_factory=lambda: os.getenv("POLYGON_API_KEY", ""))
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_api_secret: str = field(default_factory=lambda: os.getenv("ALPACA_API_SECRET", ""))

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "quant_trading"
    db_user: str = "postgres"
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))

    # Risk Management
    max_portfolio_delta: float = 5000.0  # max delta exposure
    max_portfolio_gamma: float = 1000.0
    max_position_size: int = 100  # contracts
    max_drawdown_pct: float = 0.05  # 5% max drawdown

    # Trading
    symbols: List[str] = field(default_factory=lambda: ["SPY", "SPX"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    backtesting_mode: bool = True
    paper_trading: bool = False

    # Local LLM (Ollama / LM Studio / LocalAI / vLLM)
    llm_enabled: bool = field(default_factory=lambda: os.getenv("LLM_ENABLED", "true").lower() == "true")
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.1:8b"))
    llm_timeout: int = 90
    llm_temperature: float = 0.1
    llm_min_confidence: float = 0.45  # Reject LLM decisions below this

    # ZeroDTE / intraday
    account_bp: float = field(default_factory=lambda: float(os.getenv("ACCOUNT_BP", "2500")))
    max_risk_pct: float = 0.08   # 8% of BP per trade
    defined_risk: bool = False   # True = iron condors (cash account / PDT restricted)

    # Engine
    event_queue_size: int = 10000
    max_workers: int = 4


# ============================================================================
# UTILITIES
# ============================================================================


def setup_logging(name: str = "quant_bot") -> None:
    """Configure loguru logging for the application"""
    logger.remove()  # Remove default handler
    logger.add(
        f"logs/{name}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="500 MB",
        retention="7 days",
    )
    import sys, io
    _stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    logger.add(lambda msg: (_stdout.write(msg), _stdout.flush()), level="INFO", colorize=True)
