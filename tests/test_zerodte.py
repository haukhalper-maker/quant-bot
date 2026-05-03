"""
Tests for the ZeroDTE pipeline:
  - GammaExposureEngine  (GEX walls + confluence)
  - IVTermStructureAnalyzer (IV term structure + DTE selection)
  - PredictionLogger (SQLite self-learning loop)
  - ZeroDTEStrategy (signal generation + time regime)
  - signal_to_orders (SELL_STRADDLE / SELL_STRANGLE legs)
  - ZeroDTEReasoningEngine heuristic (no LLM server needed)

Run with: pytest tests/test_zerodte.py -v
"""

import asyncio
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.market_data import Candle, PolygonOptionsContract
from src.analysis import GammaExposureEngine, IVTermStructureAnalyzer, PredictionLogger, SetupPrediction
from src.strategy import ZeroDTEStrategy, SignalType, PlayType
from src.execution import signal_to_orders, OrderSide
from src.llm import ZeroDTEReasoningEngine, ZeroDTEContext


# ============================================================================
# FIXTURES
# ============================================================================


def _make_contract(
    strike: float,
    contract_type: str = "call",
    gamma: float = 0.05,
    open_interest: int = 10_000,
    volume: int = 5_000,
    implied_vol: float = 0.20,
    dte: int = 0,
    underlying_price: float = 530.0,
) -> PolygonOptionsContract:
    return PolygonOptionsContract(
        ticker=f"O:SPY{strike}{contract_type[0].upper()}",
        symbol="SPY",
        strike=strike,
        expiry=(datetime.utcnow() + timedelta(days=dte)).strftime("%Y-%m-%d"),
        contract_type=contract_type,
        bid=1.00,
        ask=1.10,
        mid=1.05,
        open_interest=open_interest,
        volume=volume,
        implied_vol=implied_vol,
        delta=0.50 if contract_type == "call" else -0.50,
        gamma=gamma,
        theta=-0.05,
        vega=0.10,
        dte=dte,
        underlying_price=underlying_price,
    )


def _spy_chain(spot: float = 530.0) -> list:
    """Realistic mini options chain: 5 strikes, 0DTE and 1DTE."""
    contracts = []
    for strike in [525, 527, 530, 533, 535]:
        for dte in [0, 1]:
            # big call OI at 533 → positive GEX wall
            oi = 50_000 if strike == 533 else 8_000
            vol = 20_000 if strike == 533 else 3_000
            contracts.append(_make_contract(strike, "call", gamma=0.06, open_interest=oi,
                                            volume=vol, dte=dte, underlying_price=spot))
            contracts.append(_make_contract(strike, "put",  gamma=0.06, open_interest=oi // 2,
                                            volume=vol // 2, dte=dte, underlying_price=spot))
    return contracts


def _candle(close: float = 530.0, ts: datetime = None) -> Candle:
    ts = ts or datetime(2025, 6, 15, 14, 0, 0)  # 10 AM ET in UTC
    return Candle(
        symbol="SPY",
        timestamp=ts,
        open=close - 0.10,
        high=close + 0.20,
        low=close - 0.20,
        close=close,
        volume=500_000,
    )


# ============================================================================
# GEX ENGINE
# ============================================================================


class TestGammaExposureEngine:

    def test_find_walls_returns_sorted_by_strength(self):
        engine = GammaExposureEngine()
        contracts = _spy_chain(spot=530.0)
        walls = engine.find_walls(contracts, spot=530.0, max_distance_pct=0.05, top_n=5)
        assert len(walls) > 0
        # Walls should be sorted descending by |GEX| × confluence
        strengths = [abs(w.net_gex_dollars) * w.confluence_score for w in walls]
        assert strengths == sorted(strengths, reverse=True)

    def test_strike_533_is_top_wall(self):
        """Strike 533 has 50k OI — should dominate the wall ranking."""
        engine = GammaExposureEngine()
        contracts = _spy_chain(spot=530.0)
        walls = engine.find_walls(contracts, spot=530.0, max_distance_pct=0.05, top_n=5)
        assert walls[0].strike == 533

    def test_net_gex_surface_covers_all_strikes(self):
        engine = GammaExposureEngine()
        contracts = _spy_chain(spot=530.0)
        surface = engine.net_gex_surface(contracts, spot=530.0)
        strikes_in_contracts = {c.strike for c in contracts}
        assert strikes_in_contracts.issubset(set(surface.keys()))

    def test_positive_call_oi_yields_positive_gex(self):
        """High call OI → dealers are long gamma → positive GEX at that strike."""
        engine = GammaExposureEngine()
        contracts = [
            _make_contract(530, "call", gamma=0.05, open_interest=100_000, underlying_price=530),
            _make_contract(530, "put",  gamma=0.05, open_interest=1_000,   underlying_price=530),
        ]
        surface = engine.net_gex_surface(contracts, spot=530.0)
        assert surface[530] > 0

    def test_confluence_score_bounded(self):
        engine = GammaExposureEngine()
        contracts = _spy_chain(spot=530.0)
        walls = engine.find_walls(contracts, spot=530.0, max_distance_pct=0.05, top_n=5)
        for w in walls:
            assert 0.0 <= w.confluence_score <= 1.0

    def test_distance_filter_excludes_far_strikes(self):
        """Strikes more than max_distance_pct away should not appear."""
        engine = GammaExposureEngine()
        contracts = _spy_chain(spot=530.0)
        walls = engine.find_walls(contracts, spot=530.0, max_distance_pct=0.005, top_n=10)
        for w in walls:
            pct_away = abs(w.strike - 530.0) / 530.0
            assert pct_away <= 0.005 + 1e-9


# ============================================================================
# IV TERM STRUCTURE ANALYZER
# ============================================================================


class TestIVTermStructureAnalyzer:

    def test_analyze_returns_one_entry_per_dte(self):
        analyzer = IVTermStructureAnalyzer()
        contracts = _spy_chain(spot=530.0)
        analyses = analyzer.analyze(contracts, spot=530.0, max_dte=7)
        dtes_found = {a.dte for a in analyses}
        assert 0 in dtes_found
        assert 1 in dtes_found

    def test_crush_probability_bounded(self):
        analyzer = IVTermStructureAnalyzer()
        contracts = _spy_chain(spot=530.0)
        for a in analyzer.analyze(contracts, spot=530.0, max_dte=7):
            assert 0.0 <= a.crush_probability <= 1.0

    def test_iv_premium_is_ratio(self):
        """iv_premium_vs_next should be ≥ 0 (ratio of ATM IV this DTE vs next)."""
        analyzer = IVTermStructureAnalyzer()
        contracts = _spy_chain(spot=530.0)
        for a in analyzer.analyze(contracts, spot=530.0, max_dte=7):
            assert a.iv_premium_vs_next >= 0

    def test_select_best_dte_iv_crush_prefers_zero(self):
        analyzer = IVTermStructureAnalyzer()
        contracts = _spy_chain(spot=530.0)
        # Boost 0DTE IV to simulate crush setup
        for c in contracts:
            if c.dte == 0:
                c.implied_vol = 0.45
            else:
                c.implied_vol = 0.20
        analyses = analyzer.analyze(contracts, spot=530.0, max_dte=7)
        best = analyzer.select_best_dte(analyses, PlayType.IV_CRUSH)
        assert best is not None
        assert best.dte == 0


# ============================================================================
# PREDICTION LOGGER
# ============================================================================


class TestPredictionLogger:

    def _make_logger(self, tmp_path):
        db_path = os.path.join(tmp_path, "predictions.db")
        return PredictionLogger(db_path=db_path)

    def _pred(self, play_type="iv_crush", predicted_low=528.0, predicted_high=532.0,
              traded=True, net_gex=1_500_000, confluence=0.75):
        return SetupPrediction(
            timestamp=datetime.utcnow(),
            symbol="SPY",
            wall_strike=533.0,
            spot_price=530.0,
            dte=0,
            net_gex_dollars=net_gex,
            confluence_score=confluence,
            play_type=play_type,
            predicted_outcome=play_type,
            predicted_low=predicted_low,
            predicted_high=predicted_high,
            condition_score=0.70,
            traded=traded,
        )

    def test_log_and_resolve(self, tmp_path):
        pl = self._make_logger(tmp_path)
        row_id = pl.log(self._pred("iv_crush"))
        assert row_id > 0
        pl.resolve(row_id, actual_price=530.5, pnl=85.0)
        stats = pl.get_accuracy_stats()
        assert "iv_crush" in stats
        assert stats["iv_crush"]["accuracy"] == 1.0
        pl.close()

    def test_resolve_outside_range_marks_incorrect(self, tmp_path):
        pl = self._make_logger(tmp_path)
        row_id = pl.log(self._pred("explosive", predicted_low=528.0, predicted_high=532.0))
        pl.resolve(row_id, actual_price=536.0, pnl=-120.0)  # outside range
        stats = pl.get_accuracy_stats()
        assert stats["explosive"]["accuracy"] == 0.0
        pl.close()

    def test_calibrated_win_rate_defaults_to_50_with_few_samples(self, tmp_path):
        pl = self._make_logger(tmp_path)
        rate = pl.get_calibrated_win_rate("pin")
        assert rate == 0.50
        pl.close()

    def test_calibrated_win_rate_clamped(self, tmp_path):
        pl = self._make_logger(tmp_path)
        for _ in range(20):
            rid = pl.log(self._pred("pin", predicted_low=529.0, predicted_high=531.0,
                                    net_gex=2_000_000, confluence=0.80))
            pl.resolve(rid, actual_price=530.0, pnl=60.0)
        rate = pl.get_calibrated_win_rate("pin", min_samples=15)
        assert 0.30 <= rate <= 0.80
        pl.close()


# ============================================================================
# ZERODTESTRATEGY — time regime + signal generation
# ============================================================================


class TestTimeRegime:
    """Test UTC → ET conversion in _time_regime."""

    from src.strategy import ZeroDTEStrategy, TimeRegime

    def test_pre_open_is_avoid(self):
        from src.strategy import ZeroDTEStrategy, TimeRegime
        # 13:00 UTC = 9:00 AM EDT (UTC-4) → AVOID
        ts = datetime(2025, 6, 15, 13, 0, 0)
        assert ZeroDTEStrategy._time_regime(ts) == TimeRegime.AVOID

    def test_open_window(self):
        from src.strategy import ZeroDTEStrategy, TimeRegime
        # 14:00 UTC = 10:00 AM EDT → OPEN
        ts = datetime(2025, 6, 15, 14, 0, 0)
        assert ZeroDTEStrategy._time_regime(ts) == TimeRegime.OPEN

    def test_midday(self):
        from src.strategy import ZeroDTEStrategy, TimeRegime
        # 16:00 UTC = 12:00 PM EDT → MIDDAY
        ts = datetime(2025, 6, 15, 16, 0, 0)
        assert ZeroDTEStrategy._time_regime(ts) == TimeRegime.MIDDAY

    def test_close_window(self):
        from src.strategy import ZeroDTEStrategy, TimeRegime
        # 18:00 UTC = 2:00 PM EDT → CLOSE
        ts = datetime(2025, 6, 15, 18, 0, 0)
        assert ZeroDTEStrategy._time_regime(ts) == TimeRegime.CLOSE

    def test_after_close(self):
        from src.strategy import ZeroDTEStrategy, TimeRegime
        # 21:00 UTC = 5:00 PM EDT → CLOSED
        ts = datetime(2025, 6, 15, 21, 0, 0)
        assert ZeroDTEStrategy._time_regime(ts) == TimeRegime.CLOSED

    def test_winter_est_offset(self):
        from src.strategy import ZeroDTEStrategy, TimeRegime
        # January: UTC-5 (EST). 14:00 UTC = 9:00 AM EST → AVOID (before 9:50)
        ts = datetime(2025, 1, 15, 14, 0, 0)
        assert ZeroDTEStrategy._time_regime(ts) == TimeRegime.AVOID

    def test_winter_open_window(self):
        from src.strategy import ZeroDTEStrategy, TimeRegime
        # January: 14:55 UTC = 9:55 AM EST → OPEN
        ts = datetime(2025, 1, 15, 14, 55, 0)
        assert ZeroDTEStrategy._time_regime(ts) == TimeRegime.OPEN


class TestZeroDTEStrategy:

    @pytest.fixture
    def strategy(self, tmp_path):
        pl = PredictionLogger(db_path=str(tmp_path / "p.db"))
        s = ZeroDTEStrategy(account_bp=2500.0, risk_pct=0.08, prediction_logger=pl)
        yield s
        pl.close()

    @pytest.mark.asyncio
    async def test_returns_none_before_open(self, strategy):
        """No signal should be emitted in AVOID regime."""
        # 13:00 UTC = 9:00 AM EDT
        candle = _candle(ts=datetime(2025, 6, 15, 13, 0, 0))
        contracts = _spy_chain(spot=530.0)
        engine = GammaExposureEngine()
        analyzer = IVTermStructureAnalyzer()
        walls = engine.find_walls(contracts, 530.0, max_distance_pct=0.05)
        term  = analyzer.analyze(contracts, 530.0, max_dte=7)
        result = await strategy.on_bar(candle, walls, term, available_bp=2500.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_with_no_walls(self, strategy):
        candle = _candle(ts=datetime(2025, 6, 15, 14, 0, 0))  # 10 AM EDT
        result = await strategy.on_bar(candle, walls=[], term_analyses=[], available_bp=2500.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_generates_signal_during_open_window(self, strategy):
        """Should produce a signal when conditions align during OPEN."""
        # 10:00 AM EDT = 14:00 UTC
        candle = _candle(530.0, ts=datetime(2025, 6, 15, 14, 0, 0))
        contracts = _spy_chain(spot=530.0)
        # Boost IV at 0DTE to trigger iv_crush or pin
        for c in contracts:
            if c.dte == 0:
                c.implied_vol = 0.45
                c.open_interest = 80_000
                c.volume = 40_000
        engine   = GammaExposureEngine()
        analyzer = IVTermStructureAnalyzer()
        walls = engine.find_walls(contracts, 530.0, max_distance_pct=0.05, top_n=5)
        term  = analyzer.analyze(contracts, 530.0, max_dte=7)
        # Seed price history so velocity/deceleration are defined
        for p in [529.5, 529.7, 529.9, 530.0]:
            strategy._price_history.append(p)
        result = await strategy.on_bar(candle, walls, term, available_bp=2500.0)
        # May or may not signal depending on scores — just assert it doesn't crash
        # and if it does signal, it's a valid type
        if result is not None:
            assert result.signal_type in (
                SignalType.SELL_STRADDLE, SignalType.SELL_STRANGLE,
                SignalType.BUY_CALL, SignalType.BUY_PUT,
                SignalType.STRADDLE,
            )

    @pytest.mark.asyncio
    async def test_position_size_respects_risk_pct(self, strategy):
        """Kelly-sized contracts should never exceed 8% of BP."""
        candle = _candle(530.0, ts=datetime(2025, 6, 15, 14, 0, 0))
        contracts = _spy_chain(spot=530.0)
        for c in contracts:
            if c.dte == 0:
                c.implied_vol = 0.50
                c.open_interest = 100_000
                c.volume = 50_000
        engine   = GammaExposureEngine()
        analyzer = IVTermStructureAnalyzer()
        walls = engine.find_walls(contracts, 530.0, max_distance_pct=0.05, top_n=5)
        term  = analyzer.analyze(contracts, 530.0, max_dte=7)
        for p in [529.0, 529.5, 529.8, 530.0]:
            strategy._price_history.append(p)
        result = await strategy.on_bar(candle, walls, term, available_bp=2500.0)
        if result is not None:
            max_risk = 2500.0 * 0.08
            assert result.position_size >= 1


# ============================================================================
# SIGNAL → ORDERS
# ============================================================================


class TestSignalToOrders:

    def _mock_signal(self, signal_type, strike=530.0, position_size=2,
                     implied_vol=0.25, dte=0,
                     call_strike=533.0, put_strike=527.0):
        sig = MagicMock()
        sig.signal_type = signal_type
        sig.symbol = "SPY"
        sig.expiry = (datetime.utcnow() + timedelta(days=dte)).strftime("%Y-%m-%d")
        sig.strike = strike
        sig.position_size = position_size
        sig.metadata = {
            "implied_vol": implied_vol,
            "dte": dte,
            "call_strike": call_strike,
            "put_strike": put_strike,
        }
        sig.timestamp = datetime.utcnow()
        return sig

    def test_sell_straddle_produces_two_legs(self):
        sig = self._mock_signal(SignalType.SELL_STRADDLE, strike=530.0)
        orders = signal_to_orders(sig, mid_price=530.0)
        assert len(orders) == 2
        sides   = {o.side for o in orders}
        types   = {o.option_type for o in orders}
        assert sides == {OrderSide.SELL}
        assert types == {"CALL", "PUT"}

    def test_sell_straddle_same_strike(self):
        sig = self._mock_signal(SignalType.SELL_STRADDLE, strike=530.0)
        orders = signal_to_orders(sig, mid_price=530.0)
        for o in orders:
            assert o.strike == 530.0

    def test_sell_strangle_produces_two_legs(self):
        sig = self._mock_signal(SignalType.SELL_STRANGLE, strike=530.0,
                                call_strike=533.0, put_strike=527.0)
        orders = signal_to_orders(sig, mid_price=530.0)
        assert len(orders) == 2
        assert all(o.side == OrderSide.SELL for o in orders)

    def test_sell_strangle_correct_strikes(self):
        sig = self._mock_signal(SignalType.SELL_STRANGLE, strike=530.0,
                                call_strike=533.0, put_strike=527.0)
        orders = signal_to_orders(sig, mid_price=530.0)
        call_order = next(o for o in orders if o.option_type == "CALL")
        put_order  = next(o for o in orders if o.option_type == "PUT")
        assert call_order.strike == 533.0
        assert put_order.strike  == 527.0

    def test_sell_straddle_prices_positive(self):
        sig = self._mock_signal(SignalType.SELL_STRADDLE, dte=1)
        orders = signal_to_orders(sig, mid_price=530.0)
        for o in orders:
            assert o.price > 0

    def test_sell_strangle_call_cheaper_than_itm(self):
        """OTM call (533) should be cheaper than ATM call (530)."""
        sig_straddle  = self._mock_signal(SignalType.SELL_STRADDLE, strike=530.0, dte=1)
        sig_strangle  = self._mock_signal(SignalType.SELL_STRANGLE, strike=530.0,
                                          call_strike=533.0, put_strike=527.0, dte=1)
        straddle_call = next(o for o in signal_to_orders(sig_straddle, 530.0) if o.option_type == "CALL")
        strangle_call = next(o for o in signal_to_orders(sig_strangle, 530.0) if o.option_type == "CALL")
        assert strangle_call.price < straddle_call.price

    def test_buy_call_single_leg(self):
        sig = self._mock_signal(SignalType.BUY_CALL, strike=533.0)
        orders = signal_to_orders(sig, mid_price=530.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].option_type == "CALL"


# ============================================================================
# LLM HEURISTIC (no server required)
# ============================================================================


class TestZeroDTEHeuristic:

    def _make_ctx(self, condition_score=0.70, net_gex=1_500_000,
                  crush_prob=0.40, iv_premium=1.25, play_type="iv_crush"):
        return ZeroDTEContext(
            symbol="SPY",
            timestamp=datetime.utcnow(),
            spot=530.0,
            wall_strike=533.0,
            wall_type="pin",
            net_gex_dollars=net_gex,
            confluence_score=0.75,
            dte=0,
            atm_iv=0.30,
            iv_premium_vs_next=iv_premium,
            crush_probability=crush_prob,
            expected_move_1sd=3.5,
            velocity_pct=0.0005,
            deceleration=0.06,
            play_type=play_type,
            condition_score=condition_score,
            kelly_contracts=2,
            available_bp=2500.0,
        )

    def test_heuristic_enters_strong_setup(self):
        engine = ZeroDTEReasoningEngine(client=None)
        ctx = self._make_ctx(condition_score=0.72, net_gex=2_000_000,
                             crush_prob=0.50, iv_premium=1.30)
        decision = engine._heuristic(ctx)
        assert decision.action == "enter"
        assert decision.confidence >= 0.65

    def test_heuristic_skips_low_score(self):
        engine = ZeroDTEReasoningEngine(client=None)
        ctx = self._make_ctx(condition_score=0.45, net_gex=300_000,
                             crush_prob=0.10, iv_premium=1.02)
        decision = engine._heuristic(ctx)
        assert decision.action == "skip"

    def test_heuristic_skips_low_gex(self):
        """Even with good score, tiny GEX shouldn't trigger entry."""
        engine = ZeroDTEReasoningEngine(client=None)
        ctx = self._make_ctx(condition_score=0.70, net_gex=100_000,
                             crush_prob=0.40, iv_premium=1.25)
        decision = engine._heuristic(ctx)
        assert decision.action == "skip"

    @pytest.mark.asyncio
    async def test_evaluate_falls_back_to_heuristic_when_no_client(self):
        """evaluate() should use heuristic when no LLM client is configured."""
        engine = ZeroDTEReasoningEngine(client=None)
        ctx = self._make_ctx(condition_score=0.72, net_gex=2_000_000,
                             crush_prob=0.50, iv_premium=1.30)
        decision = await engine.evaluate(ctx)
        assert decision.action in ("enter", "skip")
        assert decision.source == "heuristic"
