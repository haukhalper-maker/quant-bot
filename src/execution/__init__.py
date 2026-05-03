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

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from scipy.stats import norm as _bsm_norm


# ---------------------------------------------------------------------------
# BSM helpers — used to price individual option legs realistically
# ---------------------------------------------------------------------------

def _bsm_call(S: float, K: float, sigma: float, T: float) -> float:
    """Black-Scholes call price (r=0)."""
    if T <= 1e-6:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return float(S * _bsm_norm.cdf(d1) - K * _bsm_norm.cdf(d1 - sq))


def _bsm_put(S: float, K: float, sigma: float, T: float) -> float:
    """Black-Scholes put price via put-call parity (r=0)."""
    if T <= 1e-6:
        return max(K - S, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return float(K * _bsm_norm.cdf(sq - d1) - S * _bsm_norm.cdf(-d1))


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
        underlying = self._market_prices.get(order.symbol)

        # Options orders carry their Bachelier-estimated premium in order.price.
        # Use that as the fill mid when:
        #   (a) order.price is set, AND
        #   (b) it looks like an option premium — i.e., < 20% of the underlying
        #       (avoids accidentally using an underlying price stored in order.price)
        if (
            order.price is not None
            and order.price > 0
            and (underlying is None or order.price < 0.20 * underlying)
        ):
            mid = order.price
        elif underlying is not None and underlying > 0:
            mid = underlying
        else:
            logger.warning(f"No market price for {order.symbol} — using $1 fallback")
            mid = 1.0

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
        # Use BSM price from metadata when available (e.g. TailHedgeStrategy).
        # Fallback: compute from implied_vol in metadata; last resort: mid_price.
        price_override = signal.metadata.get("put_price")
        if price_override and price_override > 0:
            leg_price = float(price_override)
        else:
            iv = signal.metadata.get("implied_vol")
            if iv:
                try:
                    exp_dt = datetime.strptime(signal.expiry, "%Y-%m-%d")
                    T = max((exp_dt - signal.timestamp.replace(tzinfo=None)).days, 1) / 365.0
                except Exception:
                    T = 28 / 365.0
                leg_price = max(_bsm_put(mid_price, signal.strike, float(iv), T), 0.01)
            else:
                leg_price = mid_price
        orders.append(Order(
            order_id="", option_type="PUT", strike=signal.strike,
            side=OrderSide.BUY, quantity=signal.position_size,
            price=leg_price, **common,
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
        # Buy ATM call + ATM put — price each leg with BSM.
        iv = max(signal.metadata.get("implied_vol", 0.20), 0.01)
        try:
            exp_dt = datetime.strptime(signal.expiry, "%Y-%m-%d")
            T = max((exp_dt - signal.timestamp.replace(tzinfo=None)).days, 1) / 365.0
        except Exception:
            T = 28 / 365.0
        K = signal.strike
        call_price = max(_bsm_call(mid_price, K, iv, T), 0.01)
        put_price  = max(_bsm_put(mid_price, K, iv, T), 0.01)
        for opt_type, leg_price in (("CALL", call_price), ("PUT", put_price)):
            orders.append(Order(
                order_id="", option_type=opt_type, strike=K,
                side=OrderSide.BUY, quantity=signal.position_size,
                price=leg_price, **common,
            ))

    elif signal.signal_type == SignalType.STRANGLE:
        # Buy OTM call (strike+5%) + OTM put (strike-5%) — price with BSM.
        iv = max(signal.metadata.get("implied_vol", 0.20), 0.01)
        try:
            exp_dt = datetime.strptime(signal.expiry, "%Y-%m-%d")
            T = max((exp_dt - signal.timestamp.replace(tzinfo=None)).days, 1) / 365.0
        except Exception:
            T = 28 / 365.0
        K_c = signal.strike * 1.05
        K_p = signal.strike * 0.95
        orders.append(Order(
            order_id="", option_type="CALL", strike=K_c,
            side=OrderSide.BUY, quantity=signal.position_size,
            price=max(_bsm_call(mid_price, K_c, iv, T), 0.01), **common,
        ))
        orders.append(Order(
            order_id="", option_type="PUT", strike=K_p,
            side=OrderSide.BUY, quantity=signal.position_size,
            price=max(_bsm_put(mid_price, K_p, iv, T), 0.01), **common,
        ))

    elif signal.signal_type == SignalType.IRON_CONDOR:
        # Short call spread + short put spread priced with BSM.
        # Strikes come from signal metadata (GEX-derived), not % offsets.
        iv = max(signal.metadata.get("implied_vol", 0.20), 0.01)
        try:
            exp_dt = datetime.strptime(signal.expiry, "%Y-%m-%d")
            T = max((exp_dt - signal.timestamp.replace(tzinfo=None)).days, 1) / 365.0
        except Exception:
            T = 42 / 365.0
        S = mid_price
        K_sc = float(signal.metadata.get("call_strike",      signal.strike * 1.02))
        K_lc = float(signal.metadata.get("wing_call_strike", signal.strike * 1.07))
        K_sp = float(signal.metadata.get("put_strike",       signal.strike * 0.98))
        K_lp = float(signal.metadata.get("wing_put_strike",  signal.strike * 0.93))
        sc_price = max(_bsm_call(S, K_sc, iv, T), 0.01)
        lc_price = max(_bsm_call(S, K_lc, iv, T), 0.01)
        sp_price = max(_bsm_put(S, K_sp, iv, T), 0.01)
        lp_price = max(_bsm_put(S, K_lp, iv, T), 0.01)
        for opt_type, sell_strike, sell_price, buy_strike, buy_price in [
            ("CALL", K_sc, sc_price, K_lc, lc_price),
            ("PUT",  K_sp, sp_price, K_lp, lp_price),
        ]:
            orders.append(Order(
                order_id="", option_type=opt_type, strike=sell_strike,
                side=OrderSide.SELL, quantity=signal.position_size,
                price=sell_price, **common,
            ))
            orders.append(Order(
                order_id="", option_type=opt_type, strike=buy_strike,
                side=OrderSide.BUY, quantity=signal.position_size,
                price=buy_price, **common,
            ))

    elif signal.signal_type == SignalType.SELL_STRADDLE:
        # Sell ATM call + ATM put — IV crush / premium collection play
        iv = max(signal.metadata.get("implied_vol", 0.20), 0.01)
        dte = signal.metadata.get("dte", 1)
        T = max(dte, 0.5) / 365.0
        K = signal.strike
        call_price = max(_bsm_call(mid_price, K, iv, T), 0.01)
        put_price  = max(_bsm_put(mid_price, K, iv, T), 0.01)
        for opt_type, leg_price in (("CALL", call_price), ("PUT", put_price)):
            orders.append(Order(
                order_id="", option_type=opt_type, strike=K,
                side=OrderSide.SELL, quantity=signal.position_size,
                price=leg_price, **common,
            ))

    elif signal.signal_type == SignalType.SELL_STRANGLE:
        # Sell OTM call + OTM put around a positive-GEX gamma wall (PIN play)
        iv = max(signal.metadata.get("implied_vol", 0.20), 0.01)
        dte = signal.metadata.get("dte", 1)
        T = max(dte, 0.5) / 365.0
        K_c = signal.metadata.get("call_strike", signal.strike * 1.01)
        K_p = signal.metadata.get("put_strike", signal.strike * 0.99)
        orders.append(Order(
            order_id="", option_type="CALL", strike=K_c,
            side=OrderSide.SELL, quantity=signal.position_size,
            price=max(_bsm_call(mid_price, K_c, iv, T), 0.01), **common,
        ))
        orders.append(Order(
            order_id="", option_type="PUT", strike=K_p,
            side=OrderSide.SELL, quantity=signal.position_size,
            price=max(_bsm_put(mid_price, K_p, iv, T), 0.01), **common,
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
    Live order placement via the Tastytrade REST API.

    Requires an authenticated TastytradeConnector — pass the connector
    after calling connector.connect() so this class can reuse its
    httpx session and session token.

    Order format (Tastytrade v1):
      POST /accounts/{account_number}/orders
      {
        "order-type": "Market",
        "time-in-force": "Day",
        "legs": [
          {
            "instrument-type": "Equity Option",
            "symbol": "SPY   241015C00530000",   ← OCC symbol
            "quantity": 2,
            "action": "Buy to Open"              ← or "Sell to Open"
          }
        ]
      }

    OCC symbol format: underlying (6 chars padded) + YYMMDD + C/P + strike*1000 (8 digits)
    e.g. SPY at $530 call expiring 2024-10-15:  SPY   241015C00530000
    """

    _BASE = "https://api.tastytrade.com"

    def __init__(self, connector=None, account_number: str = None, paper: bool = True):
        """
        connector      : TastytradeConnector (authenticated)
        account_number : override account; defaults to first account on connector
        paper          : if True, log orders but don't actually submit (safety net)
        """
        super().__init__("TastytradeExecutor")
        self._connector   = connector
        self._account_num = account_number
        self.paper        = paper

    # ── helpers ──────────────────────────────────────────────────────────

    def _account(self) -> str:
        if self._account_num:
            return self._account_num
        if self._connector and self._connector._accounts:
            return self._connector._accounts[0].get("account-number", "")
        raise RuntimeError("No Tastytrade account number available.")

    @staticmethod
    def _occ_symbol(symbol: str, expiry: str, option_type: str, strike: float) -> str:
        """Build the OCC option symbol Tastytrade expects."""
        # Underlying padded to 6 chars
        und = symbol.upper().ljust(6)
        # Expiry YYMMDD
        try:
            from datetime import datetime as _dt
            exp = _dt.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
        except Exception:
            exp = expiry.replace("-", "")[2:]   # best-effort fallback
        cp   = "C" if option_type.upper() in ("CALL", "C") else "P"
        stk  = f"{int(strike * 1000):08d}"
        return f"{und}{exp}{cp}{stk}"

    @staticmethod
    def _action(side: "OrderSide", existing_position: bool = False) -> str:
        # For simplicity assume opening new positions always
        if side == OrderSide.BUY:
            return "Buy to Open"
        return "Sell to Open"

    # ── core ─────────────────────────────────────────────────────────────

    async def _submit_order(self, order: Order) -> None:
        occ = self._occ_symbol(order.symbol, order.expiry, order.option_type, order.strike)
        body = {
            "order-type": "Market",
            "time-in-force": "Day",
            "legs": [
                {
                    "instrument-type": "Equity Option",
                    "symbol": occ,
                    "quantity": order.quantity,
                    "action": self._action(order.side),
                }
            ],
        }

        logger.info(
            f"[TT] {'PAPER ' if self.paper else ''}ORDER  "
            f"{order.side.value} {order.quantity}x {occ}  "
            f"acct={self._account()}"
        )

        if self.paper:
            # Paper mode: simulate fill at order.price without hitting the API
            order.status = OrderStatus.ACCEPTED
            fill = OrderFill(
                fill_id=self._new_fill_id(),
                order_id=order.order_id,
                quantity=order.quantity,
                price=order.price or 1.0,
                timestamp=__import__("datetime").datetime.utcnow(),
                commission=order.quantity * 0.65,
            )
            order.add_fill(fill)
            order.status = OrderStatus.FILLED
            return

        # Live submission
        if not self._connector or not self._connector._client:
            raise RuntimeError("TastytradeConnector is not connected.")

        resp = await self._connector._client.post(
            f"{self._BASE}/accounts/{self._account()}/orders",
            json=body,
        )

        if resp.status_code in (200, 201):
            data   = resp.json().get("data", {})
            order_data = data.get("order", {})
            remote_id  = str(order_data.get("id", ""))
            order.status = OrderStatus.SUBMITTED
            logger.info(f"[TT] Order submitted  remote_id={remote_id}  occ={occ}")
            # Tastytrade fills async — mark accepted; real fill arrives via websocket
            order.status = OrderStatus.ACCEPTED
        else:
            err = resp.text[:300]
            logger.error(f"[TT] Order rejected  HTTP {resp.status_code}: {err}")
            order.status = OrderStatus.REJECTED
