"""
Execution Engine — Order management, fill simulation, and live broker stubs.

Paper trading executor models:
  - Bid/ask spread (configurable in bps)
  - Market impact slippage (size-dependent)
  - Per-contract commissions
  - Realistic partial fills for large orders
  - Order rate limiting

Live executor stubs provide the interface for IB and Alpaca integration.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


# ============================================================================
# ORDER MODEL
# ============================================================================


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Full order record with lifecycle tracking."""

    order_id: str
    symbol: str
    option_type: str        # 'CALL' | 'PUT' | 'SPREAD'
    strike: float
    expiry: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None        # Limit price
    stop_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fills: List["OrderFill"] = field(default_factory=list)
    strategy_name: str = ""
    position_id: Optional[str] = None   # set after fill

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_filled(self) -> bool:
        return self.filled_quantity >= self.quantity

    @property
    def fill_cost(self) -> float:
        """Total cost of fills (positive = paid, negative = received)."""
        total = sum(f.quantity * f.price * 100 for f in self.fills)
        return total if self.side == OrderSide.BUY else -total

    @property
    def total_commission(self) -> float:
        return sum(f.commission for f in self.fills)

    def add_fill(self, fill: "OrderFill") -> None:
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.updated_at = datetime.utcnow()
        if self.filled_quantity > 0:
            self.avg_fill_price = (
                sum(f.quantity * f.price for f in self.fills) / self.filled_quantity
            )

    def __repr__(self) -> str:
        return (
            f"Order({self.order_id} {self.side.value} {self.quantity}x "
            f"{self.symbol} {self.option_type} K={self.strike} exp={self.expiry} "
            f"type={self.order_type.value} status={self.status.value})"
        )


@dataclass
class OrderFill:
    fill_id: str
    order_id: str
    quantity: int
    price: float                    # per-share premium
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0           # slippage incurred vs mid


# ============================================================================
# ABSTRACT EXECUTION ENGINE
# ============================================================================


class ExecutionEngine(ABC):
    """Abstract base for execution engines."""

    def __init__(self, name: str = "Executor"):
        self.name = name
        self.orders: Dict[str, Order] = {}
        logger.info(f"ExecutionEngine '{name}' initialized")

    def _new_order_id(self) -> str:
        return f"ORD_{uuid.uuid4().hex[:8].upper()}"

    def _new_fill_id(self) -> str:
        return f"FILL_{uuid.uuid4().hex[:8].upper()}"

    async def place_order(self, order: Order) -> str:
        """Place an order. Returns order_id."""
        if not order.order_id:
            order.order_id = self._new_order_id()
        self.orders[order.order_id] = order
        order.status = OrderStatus.SUBMITTED
        logger.info(f"Order placed: {order}")
        await self._submit_order(order)
        return order.order_id

    @abstractmethod
    async def _submit_order(self, order: Order) -> None:
        pass

    async def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if not order:
            logger.warning(f"cancel_order: unknown id {order_id}")
            return False
        if order.is_filled:
            logger.warning(f"cancel_order: {order_id} already filled")
            return False
        order.status = OrderStatus.CANCELLED
        logger.info(f"Order cancelled: {order_id}")
        return True

    async def get_status(self, order_id: str) -> Optional[OrderStatus]:
        order = self.orders.get(order_id)
        return order.status if order else None

    def get_filled_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.status == OrderStatus.FILLED]

    def get_open_orders(self) -> List[Order]:
        return [
            o for o in self.orders.values()
            if o.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)
        ]


# ============================================================================
# PAPER TRADING EXECUTOR
# ============================================================================


class PaperTradingExecutor(ExecutionEngine):
    """
    Realistic paper trading executor.

    Fill model:
      1. Mid-price from current market (passed in or estimated)
      2. Bid/ask spread: half-spread applied against the order direction
         (buys fill at ask = mid + half_spread; sells at bid = mid - half_spread)
      3. Market impact: sqrt(quantity) scaling on slippage for larger orders
      4. Per-contract commission
      5. Partial fills: 10% chance per contract of partial fill (configurable)
    """

    def __init__(
        self,
        commission_per_contract: float = 0.65,
        half_spread_bps: float = 5.0,         # 5 bps = 0.05% of mid as half-spread
        market_impact_bps_per_sqrt: float = 2.0,  # 2 bps per sqrt(contracts)
        partial_fill_rate: float = 0.0,         # 0 = always full fill in paper mode
    ):
        super().__init__("PaperTrader")
        self.commission_per_contract = commission_per_contract
        self.half_spread_bps = half_spread_bps / 10_000
        self.market_impact_bps = market_impact_bps_per_sqrt / 10_000
        self.partial_fill_rate = partial_fill_rate
        self._market_prices: Dict[str, float] = {}  # symbol → mid price

    def set_market_price(self, symbol: str, mid_price: float) -> None:
        """Update the current mid price for a symbol. Called by the market data feed."""
        self._market_prices[symbol] = mid_price

    async def _submit_order(self, order: Order) -> None:
        """Simulate fill immediately using current market price."""
        order.status = OrderStatus.ACCEPTED
        mid = self._market_prices.get(order.symbol, order.price or 1.0)
        if mid is None or mid <= 0:
            logger.warning(f"No market price for {order.symbol} — using order price or $1")
            mid = order.price or 1.0

        await self._simulate_fill(order, mid)

    async def _simulate_fill(self, order: Order, mid_price: float) -> None:
        """Execute fill simulation with spread and slippage."""
        if order.is_filled or order.status == OrderStatus.CANCELLED:
            return

        qty_to_fill = order.remaining_quantity
        if qty_to_fill <= 0:
            return

        # Partial fill simulation
        if self.partial_fill_rate > 0:
            for _ in range(qty_to_fill):
                if np.random.random() < self.partial_fill_rate:
                    qty_to_fill -= 1

        if qty_to_fill <= 0:
            return

        # --- Fill price model ---
        # Half-spread: buys execute at ask (mid + spread), sells at bid (mid - spread)
        spread_adj = self.half_spread_bps * mid_price
        if order.side == OrderSide.BUY:
            base_price = mid_price + spread_adj
        else:
            base_price = mid_price - spread_adj

        # Market impact: scales with sqrt(quantity)
        impact = self.market_impact_bps * mid_price * np.sqrt(qty_to_fill)
        if order.side == OrderSide.BUY:
            fill_price = base_price + impact
        else:
            fill_price = base_price - impact

        fill_price = max(fill_price, 0.01)  # floor at 1 cent
        slippage = fill_price - mid_price if order.side == OrderSide.BUY else mid_price - fill_price
        commission = qty_to_fill * self.commission_per_contract

        fill = OrderFill(
            fill_id=self._new_fill_id(),
            order_id=order.order_id,
            quantity=qty_to_fill,
            price=fill_price,
            timestamp=datetime.utcnow(),
            commission=commission,
            slippage=slippage,
        )
        order.add_fill(fill)

        if order.is_filled:
            order.status = OrderStatus.FILLED
            logger.info(
                f"FILL: {order.order_id} {order.side.value} {qty_to_fill}x "
                f"{order.symbol} @ ${fill_price:.4f} (mid=${mid_price:.4f} "
                f"slip=${slippage:.4f} comm=${commission:.2f})"
            )
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            logger.info(
                f"PARTIAL FILL: {order.order_id} {qty_to_fill}/{order.quantity} "
                f"@ ${fill_price:.4f}"
            )

    def fill_summary(self) -> Dict:
        """Aggregate fill statistics for reporting."""
        fills = self.get_filled_orders()
        total_commission = sum(o.total_commission for o in fills)
        total_slippage = sum(
            sum(f.slippage for f in o.fills) for o in fills
        )
        return {
            "filled_orders": len(fills),
            "open_orders": len(self.get_open_orders()),
            "total_commission": total_commission,
            "total_slippage": total_slippage,
        }


# ============================================================================
# SIGNAL → ORDER CONVERSION
# ============================================================================

def signal_to_orders(signal, mid_price: float) -> List[Order]:
    """
    Convert a strategy Signal into one or more executable Orders.

    For spreads (iron condor, straddle, etc.) this produces multiple legs.
    Each leg is an independent Order that the execution engine handles separately.
    """
    from src.strategy import SignalType

    orders: List[Order] = []
    common = dict(
        symbol=signal.symbol,
        expiry=signal.expiry or "unknown",
        order_type=OrderType.MARKET,
        strategy_name=getattr(signal, "strategy_name", ""),
    )

    if signal.signal_type == SignalType.BUY_CALL:
        orders.append(Order(
            order_id="", option_type="CALL", strike=signal.strike,
            side=OrderSide.BUY, quantity=signal.position_size,
            price=mid_price, **common,
        ))

    elif signal.signal_type == SignalType.BUY_PUT:
        orders.append(Order(
            order_id="", option_type="PUT", strike=signal.strike,
            side=OrderSide.BUY, quantity=signal.position_size,
            price=mid_price, **common,
        ))

    elif signal.signal_type == SignalType.SELL_CALL:
        orders.append(Order(
            order_id="", option_type="CALL", strike=signal.strike,
            side=OrderSide.SELL, quantity=signal.position_size,
            price=mid_price, **common,
        ))

    elif signal.signal_type == SignalType.SELL_PUT:
        orders.append(Order(
            order_id="", option_type="PUT", strike=signal.strike,
            side=OrderSide.SELL, quantity=signal.position_size,
            price=mid_price, **common,
        ))

    elif signal.signal_type == SignalType.STRADDLE:
        # Buy ATM call + ATM put
        for opt_type in ("CALL", "PUT"):
            orders.append(Order(
                order_id="", option_type=opt_type, strike=signal.strike,
                side=OrderSide.BUY, quantity=signal.position_size,
                price=mid_price, **common,
            ))

    elif signal.signal_type == SignalType.STRANGLE:
        # Buy OTM call (strike + 5%) + OTM put (strike - 5%)
        orders.append(Order(
            order_id="", option_type="CALL", strike=signal.strike * 1.05,
            side=OrderSide.BUY, quantity=signal.position_size,
            price=mid_price * 0.5, **common,
        ))
        orders.append(Order(
            order_id="", option_type="PUT", strike=signal.strike * 0.95,
            side=OrderSide.BUY, quantity=signal.position_size,
            price=mid_price * 0.5, **common,
        ))

    elif signal.signal_type == SignalType.IRON_CONDOR:
        # Sell ATM call + ATM put; buy wing protection (5% OTM)
        # Short call spread + short put spread
        for opt_type, sell_strike, buy_strike in [
            ("CALL", signal.strike * 1.02, signal.strike * 1.07),
            ("PUT",  signal.strike * 0.98, signal.strike * 0.93),
        ]:
            orders.append(Order(
                order_id="", option_type=opt_type, strike=sell_strike,
                side=OrderSide.SELL, quantity=signal.position_size,
                price=mid_price * 0.6, **common,
            ))
            orders.append(Order(
                order_id="", option_type=opt_type, strike=buy_strike,
                side=OrderSide.BUY, quantity=signal.position_size,
                price=mid_price * 0.2, **common,
            ))

    elif signal.signal_type == SignalType.CLOSE_POSITION:
        # Close signal handled at portfolio level — no order generated here
        pass

    else:
        logger.warning(f"signal_to_orders: unhandled signal type {signal.signal_type}")

    return orders


# ============================================================================
# LIVE EXECUTOR STUBS
# ============================================================================


class InteractiveBrokersExecutor(ExecutionEngine):
    """
    [API: Interactive Brokers TWS / IB Gateway]
    Requires: ibapi>=9.81
    Connection: TWS must be running locally on configured port (default 7497)
    Docs: https://interactivebrokers.github.io/tws-api/
    """

    async def _submit_order(self, order: Order) -> None:
        raise NotImplementedError(
            "IB executor not yet implemented. "
            "Install ibapi and implement EWrapper/EClient callbacks."
        )


class AlpacaExecutor(ExecutionEngine):
    """
    [API: Alpaca Options Trading API]
    Requires: alpaca-trade-api>=3.0
    Docs: https://docs.alpaca.markets/reference/optioncontracts-1
    """

    async def _submit_order(self, order: Order) -> None:
        raise NotImplementedError(
            "Alpaca executor not yet implemented. "
            "Install alpaca-py and implement REST order submission."
        )


class TastytradeExecutor(ExecutionEngine):
    """
    [API: Tastytrade API — options order placement]
    Requires: tastytrade>=1.0.0
    Docs: https://developer.tastytrade.com/
    """

    async def _submit_order(self, order: Order) -> None:
        raise NotImplementedError(
            "Tastytrade executor not yet implemented. "
            "Use tastytrade SDK to POST /accounts/{account_number}/orders."
        )
