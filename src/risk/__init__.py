"""
Risk Management - Position Limits, Greeks Exposure, Drawdown Controls
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger


# ============================================================================
# PORTFOLIO STATE
# ============================================================================


@dataclass
class PositionGreeks:
    """Greeks aggregated for a position"""

    symbol: str
    expiry: str
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0

    def aggregate(self, other: "PositionGreeks") -> None:
        """Add another position's Greeks"""
        self.delta += other.delta
        self.gamma += other.gamma
        self.vega += other.vega
        self.theta += other.theta
        self.rho += other.rho


@dataclass
class Portfolio:
    """Full portfolio state"""

    positions: Dict[str, PositionGreeks] = field(default_factory=dict)
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0
    total_rho: float = 0.0
    cash: float = 100000.0  # Starting cash
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def update_position(self, pos_greek: PositionGreeks) -> None:
        """Update or add a position"""
        key = f"{pos_greek.symbol}_{pos_greek.expiry}"
        self.positions[key] = pos_greek
        self._recalculate_greeks()

    def _recalculate_greeks(self) -> None:
        """Recalculate total Greeks across all positions"""
        self.total_delta = sum(p.delta for p in self.positions.values())
        self.total_gamma = sum(p.gamma for p in self.positions.values())
        self.total_vega = sum(p.vega for p in self.positions.values())
        self.total_theta = sum(p.theta for p in self.positions.values())
        self.total_rho = sum(p.rho for p in self.positions.values())

    def get_Greeks(self) -> Dict[str, float]:
        """Get total Greeks"""
        return {
            "delta": self.total_delta,
            "gamma": self.total_gamma,
            "vega": self.total_vega,
            "theta": self.total_theta,
            "rho": self.total_rho,
        }


# ============================================================================
# RISK LIMITS & CONSTRAINTS
# ============================================================================


@dataclass
class RiskLimits:
    """Risk management constraints"""

    # Position Greeks limits
    max_portfolio_delta: float = 5000.0
    max_portfolio_gamma: float = 1000.0
    max_position_delta: float = 2000.0
    max_vega: float = 500.0

    # Portfolio limits
    max_position_count: int = 20
    max_single_symbol_positions: int = 5
    max_notional_exposure: float = 500000.0

    # Drawdown limits
    max_daily_loss_pct: float = 0.02  # 2%
    max_monthly_loss_pct: float = 0.05  # 5%
    max_drawdown_pct: float = 0.10  # 10%

    # Execution limits
    max_orders_per_minute: int = 30
    min_order_size: int = 1
    max_order_size: int = 100

    def __repr__(self) -> str:
        return f"RiskLimits(delta={self.max_portfolio_delta}, gamma={self.max_portfolio_gamma})"


# ============================================================================
# RISK MONITOR
# ============================================================================


class RiskMonitor:
    """Monitors portfolio against risk limits"""

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.portfolio = Portfolio()
        self.violations: List[str] = []
        logger.info(f"RiskMonitor initialized with limits: {self.limits}")

    def reset_violations(self) -> None:
        """Clear violation list"""
        self.violations = []

    def check_portfolio_limits(self) -> bool:
        """
        Check if portfolio is within limits
        Returns: True if all limits met, False if any violated
        """
        self.reset_violations()

        # Delta limits
        if abs(self.portfolio.total_delta) > self.limits.max_portfolio_delta:
            self.violations.append(
                f"Delta limit exceeded: {self.portfolio.total_delta:.0f} > {self.limits.max_portfolio_delta}"
            )

        # Gamma limits
        if self.portfolio.total_gamma > self.limits.max_portfolio_gamma:
            self.violations.append(
                f"Gamma limit exceeded: {self.portfolio.total_gamma:.0f} > {self.limits.max_portfolio_gamma}"
            )

        # Vega limits
        if abs(self.portfolio.total_vega) > self.limits.max_vega:
            self.violations.append(
                f"Vega limit exceeded: {self.portfolio.total_vega:.0f} > {self.limits.max_vega}"
            )

        # Position count
        if len(self.portfolio.positions) > self.limits.max_position_count:
            self.violations.append(
                f"Position count limit exceeded: {len(self.portfolio.positions)} > {self.limits.max_position_count}"
            )

        if self.violations:
            logger.warning(f"Risk violations: {self.violations}")
            return False

        return True

    def get_risk_report(self) -> Dict:
        """Generate risk report"""
        return {
            "timestamp": self.portfolio.timestamp,
            "greeks": self.portfolio.get_Greeks(),
            "positions": len(self.portfolio.positions),
            "cash": self.portfolio.cash,
            "pnl": {
                "realized": self.portfolio.realized_pnl,
                "unrealized": self.portfolio.unrealized_pnl,
                "total": self.portfolio.realized_pnl + self.portfolio.unrealized_pnl,
            },
            "violations": self.violations,
            "within_limits": len(self.violations) == 0,
        }


# ============================================================================
# CIRCUIT BREAKERS (Emergency Stops)
# ============================================================================


class CircuitBreaker:
    """Emergency risk circuit breakers"""

    def __init__(self, monitor: RiskMonitor):
        self.monitor = monitor
        self.is_tripped = False
        self.trip_reason = None
        logger.info("CircuitBreaker initialized")

    def check(self) -> bool:
        """Check if circuit should trip"""
        if not self.monitor.check_portfolio_limits():
            self.trip(f"Risk limit violation: {self.monitor.violations}")
            return True

        # Additional checks:
        # - Max daily loss
        # - Extreme drawdown
        # - Liquidity crisis

        return False

    def trip(self, reason: str) -> None:
        """Trip the circuit breaker"""
        self.is_tripped = True
        self.trip_reason = reason
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")

    def reset(self) -> None:
        """Reset circuit breaker (manual only)"""
        self.is_tripped = False
        self.trip_reason = None
        logger.info("Circuit breaker reset")
