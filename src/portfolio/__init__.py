"""
Portfolio Manager — Position tracking, live P&L, mark-to-market, performance attribution.

Responsibilities:
  - Maintain the canonical view of all open positions
  - Mark positions to market using latest Greeks / prices
  - Track realized and unrealized P&L with attribution
  - Provide performance metrics: Sharpe, Sortino, Calmar, max drawdown
  - Feed portfolio context to the LLM reasoning engine
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


# ============================================================================
# POSITION
# ============================================================================


@dataclass
class OptionPosition:
    """
    A single options position (one leg of a spread or standalone option).
    Greeks are refreshed each time mark_to_market() is called.
    """

    position_id: str
    symbol: str             # Underlying (e.g. SPY)
    option_type: str        # 'CALL' | 'PUT'
    strike: float
    expiry: str             # YYYY-MM-DD
    quantity: int           # positive = long, negative = short
    entry_price: float      # Premium paid/received per contract (per-share)
    entry_time: datetime

    # Greeks (refreshed on mark-to-market)
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    # Mark state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    is_closed: bool = False
    close_time: Optional[datetime] = None
    close_price: float = 0.0

    # Metadata
    strategy_name: str = ""
    signal_type: str = ""
    llm_reasoning: str = ""

    @property
    def notional_value(self) -> float:
        """Total notional exposure (100 shares per contract)."""
        return abs(self.quantity) * self.current_price * 100

    @property
    def cost_basis(self) -> float:
        """Original cost of the position (positive = debit paid, negative = credit received)."""
        return self.quantity * self.entry_price * 100

    @property
    def days_to_expiry(self) -> float:
        exp = datetime.strptime(self.expiry, "%Y-%m-%d").date()
        return max(0.0, (exp - date.today()).days)

    def mark(self, new_price: float, delta: float = 0.0, gamma: float = 0.0,
             vega: float = 0.0, theta: float = 0.0) -> None:
        """Update position with latest market data."""
        self.current_price = new_price
        self.delta = delta
        self.gamma = gamma
        self.vega = vega
        self.theta = theta
        # P&L: (current - entry) * quantity * 100 (contract multiplier)
        self.unrealized_pnl = (new_price - self.entry_price) * self.quantity * 100

    def close(self, close_price: float, close_time: datetime) -> float:
        """Close the position. Returns realized P&L."""
        self.close_price = close_price
        self.close_time = close_time
        self.is_closed = True
        self.realized_pnl = (close_price - self.entry_price) * self.quantity * 100
        self.unrealized_pnl = 0.0
        logger.info(
            f"Position closed: {self.position_id} {self.symbol} {self.option_type} "
            f"K={self.strike} exp={self.expiry} pnl=${self.realized_pnl:.2f}"
        )
        return self.realized_pnl

    def to_dict(self) -> Dict:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry": self.expiry,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "pnl": self.realized_pnl,     # alias used by compute_performance win-rate calc
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "days_to_expiry": self.days_to_expiry,
            "strategy_name": self.strategy_name,
            "is_closed": self.is_closed,
        }


# ============================================================================
# EQUITY CURVE & PERFORMANCE
# ============================================================================


@dataclass
class EquityPoint:
    timestamp: datetime
    portfolio_value: float
    daily_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    num_positions: int


@dataclass
class PerformanceMetrics:
    """Annualized performance statistics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_factor: float        # gross profits / gross losses
    avg_win: float
    avg_loss: float
    total_trades: int
    trading_days: int


def compute_performance(equity_curve: List[EquityPoint], trades: List[Dict]) -> PerformanceMetrics:
    """Compute full performance metrics from equity curve and trade list."""
    if len(equity_curve) < 2:
        return PerformanceMetrics(
            total_return=0, annualized_return=0, annualized_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_duration_days=0,
            win_rate=0, profit_factor=0, avg_win=0, avg_loss=0,
            total_trades=0, trading_days=0,
        )

    values = np.array([p.portfolio_value for p in equity_curve])
    initial, final = values[0], values[-1]
    total_return = (final - initial) / initial

    daily_returns = np.diff(values) / values[:-1]
    n = len(daily_returns)

    ann_return = (1 + total_return) ** (252 / max(n, 1)) - 1
    ann_vol = float(np.std(daily_returns) * np.sqrt(252)) if n > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Sortino: downside deviation only
    downside = daily_returns[daily_returns < 0]
    downside_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 1 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

    # Max drawdown + duration
    peak_idx = 0
    max_dd = 0.0
    max_dd_dur = 0
    current_dd_start = 0
    for i, v in enumerate(values):
        if v >= values[peak_idx]:
            peak_idx = i
            current_dd_start = i
        else:
            dd = (values[peak_idx] - v) / values[peak_idx]
            if dd > max_dd:
                max_dd = dd
                max_dd_dur = i - current_dd_start

    calmar = ann_return / max_dd if max_dd > 0 else 0.0

    # Trade-level stats
    pnls = [t.get("pnl", 0) for t in trades if not t.get("is_open", False)]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_dur,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_trades=len(pnls),
        trading_days=n,
    )


# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================


class PortfolioManager:
    """
    Single source of truth for portfolio state.

    Responsibilities:
      - Track open and closed positions
      - Compute aggregate Greeks across all positions
      - Mark-to-market on each new price update
      - Record equity curve for performance analysis
      - Provide snapshot dicts for LLM context

    Thread safety: designed for a single async event loop — no locks needed.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, OptionPosition] = {}  # position_id → position
        self.closed_positions: List[OptionPosition] = []
        self.equity_curve: List[EquityPoint] = []
        self._pos_counter = 0
        logger.info(f"PortfolioManager initialized — cash=${initial_cash:,.2f}")

    # ------------------------------------------------------------------
    # POSITION LIFECYCLE
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiry: str,
        quantity: int,
        entry_price: float,
        strategy_name: str = "",
        signal_type: str = "",
        llm_reasoning: str = "",
    ) -> OptionPosition:
        """Open a new position and debit/credit cash."""
        self._pos_counter += 1
        pos_id = f"POS_{self._pos_counter:05d}"
        cost = entry_price * quantity * 100   # quantity * multiplier

        pos = OptionPosition(
            position_id=pos_id,
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            current_price=entry_price,
            strategy_name=strategy_name,
            signal_type=signal_type,
            llm_reasoning=llm_reasoning,
        )
        self.positions[pos_id] = pos
        self.cash -= cost   # negative for buys, positive credit for sells
        logger.info(
            f"Opened {pos_id}: {'+' if quantity > 0 else ''}{quantity}x "
            f"{symbol} {option_type} K={strike} exp={expiry} @ ${entry_price:.2f} "
            f"| cash=${self.cash:,.2f}"
        )
        return pos

    def close_position(
        self, position_id: str, close_price: float
    ) -> Optional[float]:
        """
        Close a position by ID. Returns realized P&L or None if not found.

        Cash accounting
        ---------------
        Mirrors the open transaction but in reverse:
          open:  cash -= entry_price * quantity * 100
                 (long: pay premium; short: receive premium as negative debit)
          close: cash += close_price * quantity * 100
                 (long: receive proceeds from sale; short: pay to buy back)

        P&L = (close_price - entry_price) * quantity * 100
            = net cash inflow from open + close combined.

        Previous bug: sign was inverted AND realized P&L was added a second
        time (double counting).  Correct formula uses only the closing
        transaction cash flow.
        """
        pos = self.positions.get(position_id)
        if pos is None:
            logger.warning(f"close_position: unknown id {position_id}")
            return None
        if pos.is_closed:
            logger.warning(f"close_position: {position_id} already closed")
            return pos.realized_pnl

        pnl = pos.close(close_price, datetime.utcnow())
        # Closing transaction: reverse of open
        #   Long (qty>0): sell to close → receive close_price * qty * 100
        #   Short (qty<0): buy to close → pay close_price * |qty| * 100
        # Both cases: cash += close_price * quantity * 100
        self.cash += close_price * pos.quantity * 100
        self.closed_positions.append(pos)
        del self.positions[position_id]
        return pnl

    def close_all(self, prices: Dict[str, float]) -> float:
        """Close all open positions at given prices. Returns total realized P&L."""
        total_pnl = 0.0
        for pos_id in list(self.positions.keys()):
            pos = self.positions[pos_id]
            price = prices.get(pos.symbol, pos.current_price)
            pnl = self.close_position(pos_id, price)
            if pnl is not None:
                total_pnl += pnl
        return total_pnl

    # ------------------------------------------------------------------
    # MARK-TO-MARKET
    # ------------------------------------------------------------------

    def mark_position(
        self,
        position_id: str,
        new_price: float,
        delta: float = 0.0,
        gamma: float = 0.0,
        vega: float = 0.0,
        theta: float = 0.0,
    ) -> None:
        """Update a single position's market price and Greeks."""
        pos = self.positions.get(position_id)
        if pos:
            pos.mark(new_price, delta, gamma, vega, theta)

    def mark_all(self, prices: Dict[str, float]) -> None:
        """Bulk mark all positions. Prices keyed by symbol (uses current_price as fallback)."""
        for pos in self.positions.values():
            price = prices.get(pos.symbol, pos.current_price)
            pos.mark(price, pos.delta, pos.gamma, pos.vega, pos.theta)

    # ------------------------------------------------------------------
    # AGGREGATES
    # ------------------------------------------------------------------

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value: cash + unrealized P&L across all positions."""
        return self.cash + self.total_unrealized_pnl

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.closed_positions)

    @property
    def total_pnl(self) -> float:
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def net_delta(self) -> float:
        return sum(p.delta * p.quantity * 100 for p in self.positions.values())

    @property
    def net_gamma(self) -> float:
        return sum(p.gamma * p.quantity * 100 for p in self.positions.values())

    @property
    def net_vega(self) -> float:
        return sum(p.vega * p.quantity * 100 for p in self.positions.values())

    @property
    def net_theta(self) -> float:
        return sum(p.theta * p.quantity * 100 for p in self.positions.values())

    def positions_for_symbol(self, symbol: str) -> List[OptionPosition]:
        return [p for p in self.positions.values() if p.symbol == symbol]

    # ------------------------------------------------------------------
    # EQUITY CURVE
    # ------------------------------------------------------------------

    def snapshot(self, daily_pnl: float = 0.0) -> EquityPoint:
        """Record current portfolio state to equity curve. Returns the point."""
        pt = EquityPoint(
            timestamp=datetime.utcnow(),
            portfolio_value=self.portfolio_value,
            daily_pnl=daily_pnl,
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=self.total_unrealized_pnl,
            num_positions=len(self.positions),
        )
        self.equity_curve.append(pt)
        return pt

    # ------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------

    def get_performance(self) -> Optional[PerformanceMetrics]:
        if len(self.equity_curve) < 2:
            return None
        trades = [p.to_dict() for p in self.closed_positions]
        return compute_performance(self.equity_curve, trades)

    def get_risk_snapshot(self) -> Dict:
        """Compact risk snapshot for LLM context / risk monitor."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "unrealized_pnl": self.total_unrealized_pnl,
            "realized_pnl": self.total_realized_pnl,
            "net_delta": self.net_delta,
            "net_gamma": self.net_gamma,
            "net_vega": self.net_vega,
            "net_theta": self.net_theta,
            "open_positions": len(self.positions),
            "return_pct": (self.portfolio_value - self.initial_cash) / self.initial_cash,
        }

    def open_positions_summary(self, symbol: Optional[str] = None) -> List[Dict]:
        """Positions dict list suitable for LLM context (trimmed to key fields)."""
        positions = self.positions_for_symbol(symbol) if symbol else list(self.positions.values())
        return [p.to_dict() for p in positions]

    def print_report(self) -> None:
        """Print a formatted portfolio report to stdout."""
        snap = self.get_risk_snapshot()
        metrics = self.get_performance()

        print("\n" + "=" * 70)
        print("PORTFOLIO REPORT")
        print("=" * 70)
        print(f"Portfolio Value:  ${snap['portfolio_value']:>12,.2f}")
        print(f"Cash:             ${snap['cash']:>12,.2f}")
        print(f"Unrealized P&L:   ${snap['unrealized_pnl']:>12,.2f}")
        print(f"Realized P&L:     ${snap['realized_pnl']:>12,.2f}")
        print(f"Return:           {snap['return_pct']:>11.2%}")
        print()
        print(f"Net Delta:  {snap['net_delta']:>8.2f}")
        print(f"Net Gamma:  {snap['net_gamma']:>8.4f}")
        print(f"Net Vega:   {snap['net_vega']:>8.2f}")
        print(f"Net Theta:  {snap['net_theta']:>8.2f}")
        print()
        print(f"Open Positions: {snap['open_positions']}")

        if metrics:
            print()
            print(f"Sharpe Ratio:    {metrics.sharpe_ratio:>7.3f}")
            print(f"Sortino Ratio:   {metrics.sortino_ratio:>7.3f}")
            print(f"Max Drawdown:    {metrics.max_drawdown:>7.2%}")
            print(f"Win Rate:        {metrics.win_rate:>7.2%}")
            print(f"Profit Factor:   {metrics.profit_factor:>7.2f}")
            print(f"Avg Win:         ${metrics.avg_win:>7.2f}")
            print(f"Avg Loss:        ${metrics.avg_loss:>7.2f}")
            print(f"Total Trades:    {metrics.total_trades}")

        if self.positions:
            print()
            print(f"{'ID':<12} {'Symbol':<6} {'Type':<5} {'K':>7} {'Qty':>5} "
                  f"{'Entry':>7} {'Curr':>7} {'UPnL':>9} {'DTE':>5}")
            print("-" * 70)
            for pos in self.positions.values():
                print(
                    f"{pos.position_id:<12} {pos.symbol:<6} {pos.option_type:<5} "
                    f"{pos.strike:>7.1f} {pos.quantity:>5} "
                    f"${pos.entry_price:>6.2f} ${pos.current_price:>6.2f} "
                    f"${pos.unrealized_pnl:>8.2f} {pos.days_to_expiry:>5.0f}"
                )
        print("=" * 70)
