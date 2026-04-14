"""
Tests for Risk Management module
Run with: pytest tests/test_risk.py -v
"""

import pytest
from src.risk import (
    PositionGreeks,
    Portfolio,
    RiskLimits,
    RiskMonitor,
    CircuitBreaker,
)


class TestPortfolioGreeks:
    """Test Greeks aggregation"""

    def test_position_greeks_creation(self):
        """Test creating position Greeks"""
        greeks = PositionGreeks(symbol="SPY", expiry="2024-01-19", delta=100.0)
        assert greeks.delta == 100.0

    def test_aggregate_greeks(self):
        """Test aggregating Greeks"""
        pos1 = PositionGreeks(symbol="SPY", expiry="2024-01-19", delta=100.0, gamma=0.5)
        pos2 = PositionGreeks(symbol="SPY", expiry="2024-02-16", delta=50.0, gamma=0.2)

        pos1.aggregate(pos2)
        assert pos1.delta == 150.0
        assert pos1.gamma == 0.7


class TestPortfolio:
    """Test portfolio state management"""

    def test_portfolio_creation(self):
        """Test creating portfolio"""
        portfolio = Portfolio(cash=100000.0)
        assert portfolio.cash == 100000.0
        assert len(portfolio.positions) == 0

    def test_update_position(self):
        """Test updating portfolio position"""
        portfolio = Portfolio()
        greeks = PositionGreeks(symbol="SPY", expiry="2024-01-19", delta=100.0)

        portfolio.update_position(greeks)
        assert len(portfolio.positions) == 1
        assert portfolio.total_delta == 100.0


class TestRiskLimits:
    """Test risk limit configuration"""

    def test_default_limits(self):
        """Test default risk limits"""
        limits = RiskLimits()
        assert limits.max_portfolio_delta == 5000.0
        assert limits.max_portfolio_gamma == 1000.0


class TestRiskMonitor:
    """Test risk monitoring"""

    def test_monitor_creation(self):
        """Test creating monitor"""
        limits = RiskLimits(max_portfolio_delta=1000.0)
        monitor = RiskMonitor(limits)
        assert monitor.limits.max_portfolio_delta == 1000.0

    def test_check_limits_pass(self):
        """Test limits check (passing)"""
        limits = RiskLimits(max_portfolio_delta=5000.0)
        monitor = RiskMonitor(limits)
        
        # Create a position within limits
        greeks = PositionGreeks(symbol="SPY", expiry="2024-01-19", delta=1000.0)
        monitor.portfolio.update_position(greeks)

        # Should pass
        assert monitor.check_portfolio_limits() is True

    def test_check_limits_fail(self):
        """Test limits check (failing)"""
        limits = RiskLimits(max_portfolio_delta=1000.0)
        monitor = RiskMonitor(limits)

        # Create position that violates delta limit
        greeks = PositionGreeks(symbol="SPY", expiry="2024-01-19", delta=2000.0)
        monitor.portfolio.update_position(greeks)

        # Should fail
        assert monitor.check_portfolio_limits() is False
        assert len(monitor.violations) > 0


class TestCircuitBreaker:
    """Test emergency circuit breaker"""

    def test_circuit_breaker_normal(self):
        """Test circuit breaker in normal state"""
        limits = RiskLimits()
        monitor = RiskMonitor(limits)
        breaker = CircuitBreaker(monitor)

        assert breaker.is_tripped is False

    def test_circuit_breaker_trip(self):
        """Test tripping circuit breaker"""
        limits = RiskLimits(max_portfolio_delta=1000.0)
        monitor = RiskMonitor(limits)
        breaker = CircuitBreaker(monitor)

        # Create violating position
        greeks = PositionGreeks(symbol="SPY", expiry="2024-01-19", delta=2000.0)
        monitor.portfolio.update_position(greeks)

        # Check should trip breaker
        result = breaker.check()
        assert result is True
        assert breaker.is_tripped is True
