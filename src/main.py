"""
QuantBot — Institutional Options Trading Engine
Main orchestration layer: wires event loop, data, analysis, strategy, LLM, execution, risk.

Usage:
    python -m src backtest --start 2024-01-01 --end 2024-12-31
    python -m src live --paper
    python -m src live               # real money — requires confirmation
    python -m src status
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
import pandas as pd
from loguru import logger

from src.core import EventLoop, BotConfig, setup_logging, Event, EventType
from src.market_data import MockDataConnector, TastytradeConnector, Candle, Tick, DeviceChallengeRequired
from src.analysis import AnalysisEngine
from src.strategy import (
    VolatilityMeanReversionStrategy,
    GammaScalpingStrategy,
    GammaImbalanceTrader,
    TailHedgeStrategy,
    RuleEngine,
    Signal,
    SignalType,
)
from src.execution import PaperTradingExecutor, signal_to_orders, OrderSide
from src.risk import RiskMonitor, RiskLimits, CircuitBreaker
from src.portfolio import PortfolioManager
from src.llm import LocalLLMClient, TradeReasoningEngine


# ============================================================================
# QUANT BOT (LIVE / PAPER)
# ============================================================================


class QuantBot:
    """
    Main trading bot orchestrator.

    Data flow:
      MarketData → AnalysisEngine → Strategy → RuleEngine (LLM) → Execution → Portfolio → Risk
    """

    def __init__(self, config: BotConfig):
        self.config = config

        # Core
        self.event_loop = EventLoop(name="QuantBot")

        # Data
        if not config.backtesting_mode and config.tastytrade_username:
            self.data_connector = TastytradeConnector(
                username=config.tastytrade_username,
                password=config.tastytrade_password,
            )
        else:
            self.data_connector = MockDataConnector()

        # Analysis
        self.analysis = AnalysisEngine()

        # LLM reasoning (optional but enabled by default)
        self.reasoning_engine: Optional[TradeReasoningEngine] = None
        if config.llm_enabled:
            llm_client = LocalLLMClient(
                base_url=config.llm_base_url,
                model=config.llm_model,
                timeout=config.llm_timeout,
                temperature=config.llm_temperature,
            )
            self.reasoning_engine = TradeReasoningEngine(llm_client)

        # Strategy
        self.rule_engine = RuleEngine(reasoning_engine=self.reasoning_engine)
        self.rule_engine.register_strategy(VolatilityMeanReversionStrategy())
        self.rule_engine.register_strategy(GammaScalpingStrategy())

        # Portfolio
        self.portfolio = PortfolioManager(initial_cash=100_000.0)

        # Execution
        self.executor = PaperTradingExecutor(
            commission_per_contract=0.65,
            half_spread_bps=5.0,
        )

        # Risk
        self.risk_monitor = RiskMonitor(
            RiskLimits(
                max_portfolio_delta=config.max_portfolio_delta,
                max_portfolio_gamma=config.max_portfolio_gamma,
                max_position_count=10,
            )
        )
        self.circuit_breaker = CircuitBreaker(self.risk_monitor)

        # State
        self._pending_signals: List[Signal] = []
        self._candle_buffer: Dict[str, List[Candle]] = {}

        logger.info(
            f"QuantBot initialized — mode={'BACKTEST' if config.backtesting_mode else 'LIVE'} "
            f"paper={config.paper_trading} LLM={config.llm_enabled}"
        )

    async def setup(self) -> None:
        await self.data_connector.connect()
        self._setup_event_subscriptions()
        logger.info(f"Data connector: {self.data_connector.name}")
        await self._seed_iv_histories()

    async def _seed_iv_histories(self) -> None:
        """
        Pre-populate 30-day IV history in all strategies from Tastytrade
        market metrics.  Called once at startup.

        Without this, VolatilityMeanReversionStrategy uses an AR(1) process
        seeded from a random Gaussian — the IV percentile reading is garbage
        for the first 30+ candles.  With real data the percentile is correct
        from candle 1.
        """
        if not isinstance(self.data_connector, TastytradeConnector):
            return
        for sym in self.config.symbols:
            try:
                history = await self.data_connector.seed_iv_history(sym, days=30)
                for strategy in self.rule_engine.strategies:
                    if hasattr(strategy, "seed_iv_history"):
                        strategy.seed_iv_history(sym, history)
            except Exception as exc:
                logger.warning(f"IV history seeding failed for {sym}: {exc}")

    async def run(self) -> None:
        try:
            await self.setup()
            await self.event_loop.start()

            # Subscribe to live data
            for symbol in self.config.symbols:
                await self.data_connector.subscribe_ticks(
                    symbol, lambda tick: asyncio.create_task(self._on_tick(tick))
                )

            logger.info("Bot running — press Ctrl+C to stop")
            while True:
                await asyncio.sleep(1)
                if self.circuit_breaker.check():
                    logger.critical("Circuit breaker tripped — halting trading")
                    break

        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        except Exception as e:
            logger.exception(f"Fatal error: {e}")
        finally:
            self.portfolio.print_report()
            await self.shutdown()

    async def shutdown(self) -> None:
        logger.info("Shutting down...")
        await self.event_loop.stop()
        await self.data_connector.disconnect()
        if self.reasoning_engine:
            await self.reasoning_engine.client.close()
        logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # EVENT HANDLERS
    # ------------------------------------------------------------------

    def _setup_event_subscriptions(self) -> None:
        bus = self.event_loop.event_bus
        bus.subscribe(EventType.ORDER_SIGNAL, self._handle_signal_event)
        bus.subscribe(EventType.ORDER_FILLED, self._handle_fill_event)
        bus.subscribe(EventType.RISK_LIMIT_BREACH, self._handle_risk_breach)

    async def _handle_signal_event(self, event: Event) -> None:
        signal = event.data.get("signal")
        if signal:
            self._pending_signals.append(signal)

    async def _handle_fill_event(self, event: Event) -> None:
        fill_data = event.data
        logger.info(f"Fill event: {fill_data}")

    async def _handle_risk_breach(self, event: Event) -> None:
        logger.warning(f"Risk breach: {event.data}")

    async def _on_tick(self, tick: Tick) -> None:
        """Process a live tick through the full pipeline."""
        self.analysis.on_price(tick.symbol, tick.price)
        self.executor.set_market_price(tick.symbol, tick.price)

        # Refresh dealer gamma skew for GammaImbalanceTrader (throttled inside connector)
        if isinstance(self.data_connector, TastytradeConnector):
            try:
                skew = await self.data_connector.get_dealer_gamma_skew(
                    tick.symbol, tick.price
                )
                for strategy in self.rule_engine.strategies:
                    if hasattr(strategy, "update_gamma_skew"):
                        strategy.update_gamma_skew(tick.symbol, skew)
            except Exception as exc:
                logger.debug(f"Gamma skew refresh failed for {tick.symbol}: {exc}")

        # Collect signals from all strategies
        signals = []
        for strategy in self.rule_engine.strategies:
            sig = await strategy.on_tick(tick)
            if sig:
                signals.append(sig)

        if signals:
            await self._process_signals(signals, current_price=tick.price)

    async def _on_candle(self, candle: Candle) -> None:
        """
        Process a candle through the full pipeline.

        If connected to Tastytrade, inject real IV into the candle before
        passing it to strategies.  This replaces the AR(1) simulation in
        VolatilityMeanReversionStrategy._get_iv() with real market data.
        """
        self.analysis.on_price(candle.symbol, candle.close)
        self.executor.set_market_price(candle.symbol, candle.close)

        # Inject live IV from Tastytrade into the candle
        if isinstance(self.data_connector, TastytradeConnector):
            try:
                metrics = await self.data_connector.get_market_metrics(candle.symbol)
                m = metrics.get(candle.symbol, {})
                if m:
                    candle.implied_vol = m.get("iv")
                    candle.iv_rank = m.get("iv_rank")
                    candle.hv30 = m.get("hv30")
                    self.analysis.on_iv(candle.symbol, candle.implied_vol)
            except Exception as exc:
                logger.debug(f"IV injection failed for {candle.symbol}: {exc}")

        signals = []
        for strategy in self.rule_engine.strategies:
            sig = await strategy.on_candle(candle)
            if sig:
                signals.append(sig)

        if signals:
            await self._process_signals(signals, current_price=candle.close)

    async def _process_signals(self, signals: List[Signal], current_price: float) -> None:
        """Run signals through LLM rule engine, then execute approved ones."""
        if self.circuit_breaker.is_tripped:
            logger.warning("Circuit breaker active — all signals suppressed")
            return

        approved = await self.rule_engine.evaluate_signals(
            signals,
            portfolio=self.portfolio,
            analysis_engine=self.analysis,
        )

        for signal in approved:
            await self._execute_signal(signal, current_price)

    async def _execute_signal(self, signal: Signal, mid_price: float) -> None:
        """Convert signal to orders, execute, update portfolio."""
        if signal.signal_type == SignalType.CLOSE_POSITION:
            await self._close_symbol_positions(signal.symbol, mid_price)
            return

        orders = signal_to_orders(signal, mid_price)
        for order in orders:
            order_id = await self.executor.place_order(order)
            order = self.executor.orders[order_id]

            if order.status.value == "filled":
                qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
                pos = self.portfolio.open_position(
                    symbol=order.symbol,
                    option_type=order.option_type,
                    strike=order.strike,
                    expiry=order.expiry,
                    quantity=qty,
                    entry_price=order.avg_fill_price,
                    strategy_name=signal.strategy_name,
                    signal_type=signal.signal_type.value,
                    llm_reasoning=signal.metadata.get("llm_reasoning", ""),
                )
                # Update position ID on order for cross-referencing
                order.position_id = pos.position_id

                # Push Greeks update to risk monitor
                self._sync_risk_monitor()

    async def _close_symbol_positions(self, symbol: str, mid_price: float) -> None:
        for pos in self.portfolio.positions_for_symbol(symbol):
            pnl = self.portfolio.close_position(pos.position_id, mid_price)
            logger.info(f"Closed {pos.position_id}: realized P&L = ${pnl:.2f}")

    def _sync_risk_monitor(self) -> None:
        """Sync portfolio Greeks into the risk monitor."""
        from src.risk import PositionGreeks
        snap = self.portfolio.get_risk_snapshot()
        pg = PositionGreeks(
            symbol="PORTFOLIO",
            expiry="",
            delta=snap["net_delta"],
            gamma=snap["net_gamma"],
            vega=snap["net_vega"],
            theta=snap["net_theta"],
        )
        self.risk_monitor.portfolio.update_position(pg)


# ============================================================================
# BACKTEST RUNNER
# ============================================================================


# ============================================================================
# HISTORICAL VOL REGIMES — calibrated to actual VIX/SPY data
# ============================================================================

_VOL_REGIMES: Dict[str, dict] = {
    # ── 2024 ─────────────────────────────────────────────────────────────────
    # SPY: ~475 → ~590  (+24%)
    # VIX: ranged 12-20 most of the year
    # Aug 5 Yen-carry unwind: VIX intraday 65, daily close ~38 (≈ trading day 145)
    "2024": {
        "label": "2024 Calm Bull (VIX 12-20, Aug spike to 65)",
        "initial_price": 475.0,
        "annual_drift": 0.24,
        "base_vol": 0.135,
        "long_run_vol": 0.155,
        "vol_of_vol": 0.0035,
        "vol_mean_rev": 0.06,
        "vol_jump_prob": 0.003,
        "vol_jump_mean": 0.04,
        "iv_premium": 1.10,
        "iv_seed_mean": 0.155,
        "iv_seed_std": 0.025,
        # Events: list of {tday_start, tday_end, vol_override, drift_override, vol_floor, iv_premium}
        "events": [
            # Aug 5 spike: VIX 65 intraday; 5-day chaos window (~trading days 145-152)
            {"tday_start": 145, "tday_end": 152,
             "vol_override": 0.55, "drift_override": -5.0, "iv_premium": 1.70},
            # Cool-down rest of August — vol above normal but falling
            {"tday_start": 152, "tday_end": 175,
             "vol_floor": 0.20, "iv_premium": 1.25},
        ],
    },

    # ── 2022 ─────────────────────────────────────────────────────────────────
    # SPY: ~474 → ~380  (-19%)
    # VIX: sustained 25-35 all year, peaked ~39 in Jun/Oct
    "2022": {
        "label": "2022 Bear Market (VIX sustained 25-35)",
        "initial_price": 474.0,
        "annual_drift": -0.19,
        "base_vol": 0.26,
        "long_run_vol": 0.29,
        "vol_of_vol": 0.007,
        "vol_mean_rev": 0.025,       # slow — vol stays elevated all year
        "vol_jump_prob": 0.012,
        "vol_jump_mean": 0.055,
        "iv_premium": 1.18,
        "iv_seed_mean": 0.28,
        "iv_seed_std": 0.04,
        "events": [],
    },

    # ── 2020 COVID ───────────────────────────────────────────────────────────
    # SPY: 329 → 218 (crash) → 313 (partial recovery by Jun 30)
    # VIX: 14 calm → 85 peak → gradual decline to ~30 by Jun
    # Three-phase regime — processed sequentially by trading-day count
    "covid_2020": {
        "label": "COVID Crash 2020 (VIX 14 to 85 to 30)",
        "initial_price": 329.0,
        "iv_seed_mean": 0.155,
        "iv_seed_std": 0.020,
        "phases": [
            # Phase 1 — calm pre-crash, Feb 1 – Feb 19 (~13 trading days)
            {"max_tdays": 13,
             "annual_drift": 0.10, "base_vol": 0.15,
             "vol_start": 0.15, "vol_end": 0.18, "iv_premium": 1.05},
            # Phase 2 — crash, Feb 20 – Mar 23 (~23 trading days)
            # SPY fell from 337 → 218 = -35%; VIX spiked 15 → 85
            {"max_tdays": 23,
             "annual_drift": -2.80, "base_vol": 0.80,
             "vol_start": 0.18, "vol_end": 0.85, "iv_premium": 1.45},
            # Phase 3 — recovery, Mar 24 – Jun 30 (~69 trading days)
            # SPY recovered 218 → 313 (+44%); VIX slowly back to ~30
            {"max_tdays": 999,
             "annual_drift": 1.70, "base_vol": 0.40,
             "vol_start": 0.82, "vol_end": 0.30, "iv_premium": 1.25},
        ],
    },

    # ── Generic ──────────────────────────────────────────────────────────────
    "generic": {
        "label": "Generic (no regime)",
        "initial_price": 450.0,
        "annual_drift": 0.08,
        "base_vol": 0.18,
        "long_run_vol": 0.18,
        "vol_of_vol": 0.005,
        "vol_mean_rev": 0.03,
        "vol_jump_prob": 0.02,
        "vol_jump_mean": 0.05,
        "iv_premium": 1.08,
        "iv_seed_mean": 0.18,
        "iv_seed_std": 0.03,
        "events": [],
    },
}


class BacktestRunner:
    """
    Full event-driven backtest with strategy signals, LLM validation,
    realistic execution, and portfolio P&L tracking.
    """

    def __init__(self, config: BotConfig, data_connector=None, enable_hedge: bool = True):
        self.config = config
        self.data_connector = data_connector or MockDataConnector()
        self.analysis = AnalysisEngine()
        self.enable_hedge = enable_hedge

        # LLM (optional in backtest — useful for validating LLM logic offline)
        self.reasoning_engine: Optional[TradeReasoningEngine] = None
        if config.llm_enabled:
            llm_client = LocalLLMClient(
                base_url=config.llm_base_url,
                model=config.llm_model,
                timeout=config.llm_timeout,
                temperature=config.llm_temperature,
            )
            self.reasoning_engine = TradeReasoningEngine(llm_client)

        self.rule_engine = RuleEngine(reasoning_engine=self.reasoning_engine)
        self.rule_engine.register_strategy(VolatilityMeanReversionStrategy())
        if enable_hedge:
            self.rule_engine.register_strategy(TailHedgeStrategy())

        self.executor = PaperTradingExecutor(
            commission_per_contract=0.65,
            half_spread_bps=5.0,
        )
        self.portfolio = PortfolioManager(initial_cash=100_000.0)
        self.risk_monitor = RiskMonitor(
            RiskLimits(
                max_portfolio_delta=config.max_portfolio_delta,
                max_portfolio_gamma=config.max_portfolio_gamma,
                max_position_count=10,
            )
        )
        self.circuit_breaker = CircuitBreaker(self.risk_monitor)

        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self._prev_portfolio_value: float = 100_000.0

    @staticmethod
    def _detect_regime(start: datetime, end: datetime) -> str:
        """Map a date range to the closest historical vol regime."""
        sy, sm = start.year, start.month
        if sy == 2024:
            return "2024"
        if sy == 2022:
            return "2022"
        if sy == 2020 and sm <= 6:
            return "covid_2020"
        return "generic"

    def _seed_regime_iv_history(self, regime: dict) -> None:
        """
        Pre-populate strategy IV history from regime parameters.
        Without this, _iv_percentile() returns 50% for all first-day readings
        and the strategy enters on wrong signals.
        """
        seed_mean = regime.get("iv_seed_mean", 0.20)
        seed_std  = regime.get("iv_seed_std",  0.03)
        rng = np.random.default_rng(seed=42)

        for sym in self.config.symbols:
            # Build a 30-day AR(1) seed anchored at the regime's typical IV
            hist: List[float] = []
            val = seed_mean
            for _ in range(30):
                val = val + 0.08 * (seed_mean - val) + rng.normal(0, seed_std * 0.3)
                hist.append(float(np.clip(val, 0.04, 0.90)))
            hist[-1] = seed_mean   # pin the last value to the regime anchor

            for strategy in self.rule_engine.strategies:
                if hasattr(strategy, "seed_iv_history"):
                    strategy.seed_iv_history(sym, hist)

    async def run(self, start_date: str, end_date: str, regime_key: str = "auto") -> dict:
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date   = datetime.strptime(end_date,   "%Y-%m-%d")

        if regime_key == "auto":
            regime_key = self._detect_regime(self.start_date, self.end_date)
        regime = _VOL_REGIMES.get(regime_key, _VOL_REGIMES["generic"])

        logger.info(
            f"Backtest: {start_date} → {end_date} | "
            f"regime={regime['label']} | LLM={'on' if self.reasoning_engine else 'off'}"
        )

        try:
            await self.data_connector.connect()
            self._seed_regime_iv_history(regime)
            market_data = self._generate_market_data(regime)
            await self._run_simulation(market_data)
            result = self._build_results()
            result["regime"] = regime["label"]
            return result
        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            return {"status": "FAILED", "error": str(e)}
        finally:
            await self.data_connector.disconnect()
            if self.reasoning_engine:
                await self.reasoning_engine.client.close()

    def _bsm_mark_portfolio(self, spot: float, iv: float, current_date: datetime) -> None:
        """
        Mark every open option position at its Black-Scholes theoretical value.

        Replaces the broken mark_all(underlying_prices) which marked options
        at the underlying stock price (e.g. $590 for SPY) instead of the
        option premium (~$5-$15), producing huge phantom unrealized P&L.

        Uses each position's actual strike and option_type so the equity
        curve correctly reflects option decay, vol expansion, and spot moves.
        """
        from scipy.stats import norm as _n
        sigma = max(iv, 0.01)
        for pos in self.portfolio.positions.values():
            try:
                exp = datetime.strptime(pos.expiry, "%Y-%m-%d")
                T = max((exp - current_date).days, 0) / 365.0
                K = pos.strike
                if T <= 1e-6:
                    mark = max(spot - K, 0.0) if pos.option_type == "CALL" else max(K - spot, 0.0)
                else:
                    sq = sigma * T ** 0.5
                    d1 = (np.log(spot / K) + 0.5 * sigma ** 2 * T) / sq
                    d2 = d1 - sq
                    if pos.option_type == "CALL":
                        mark = float(spot * _n.cdf(d1) - K * _n.cdf(d2))
                    else:
                        mark = float(K * _n.cdf(-d2) - spot * _n.cdf(-d1))
                pos.mark(max(mark, 0.0))
            except Exception:
                pass   # leave current_price unchanged on error

    async def _run_simulation(self, market_data: Dict[datetime, List[Candle]]) -> None:
        current = self.start_date
        peak_value: float = self.portfolio.portfolio_value   # high-water mark for hedge

        while current <= self.end_date:
            if current.weekday() >= 5:  # Skip weekends
                current += timedelta(days=1)
                continue

            day_candles = market_data.get(current, [])
            daily_pnl = 0.0

            for candle in day_candles:
                self.analysis.on_price(candle.symbol, candle.close)
                self.executor.set_market_price(candle.symbol, candle.close)

                # Mark open positions at BSM theoretical value (not underlying price)
                self._bsm_mark_portfolio(candle.close, candle.implied_vol or 0.20, current)

                # Feed current drawdown state into TailHedgeStrategy before signals
                current_portfolio_value = self.portfolio.portfolio_value
                for strategy in self.rule_engine.strategies:
                    if hasattr(strategy, "update_portfolio_state"):
                        strategy.update_portfolio_state(
                            current_value=current_portfolio_value,
                            peak_value=peak_value,
                            capital=100_000.0,
                        )

                # Collect strategy signals; sync IV estimates into analysis engine
                signals = []
                for strategy in self.rule_engine.strategies:
                    sig = await strategy.on_candle(candle)
                    if sig:
                        signals.append(sig)
                        iv = sig.metadata.get("implied_vol")
                        if iv:
                            self.analysis.on_iv(sig.symbol, iv)

                if signals and not self.circuit_breaker.is_tripped:
                    approved = await self.rule_engine.evaluate_signals(
                        signals,
                        portfolio=self.portfolio,
                        analysis_engine=self.analysis,
                    )
                    approved_syms = {s.symbol for s in approved}
                    for sig in signals:
                        if sig.signal_type != SignalType.CLOSE_POSITION and sig.symbol not in approved_syms:
                            for strategy in self.rule_engine.strategies:
                                if hasattr(strategy, "on_signal_rejected"):
                                    strategy.on_signal_rejected(sig.symbol)
                    for signal in approved:
                        await self._execute_signal(signal, candle.close)

            # EOD: snapshot portfolio and advance high-water mark
            current_value = self.portfolio.portfolio_value
            peak_value = max(peak_value, current_value)
            daily_pnl = current_value - self._prev_portfolio_value
            self._prev_portfolio_value = current_value
            self.portfolio.snapshot(daily_pnl=daily_pnl)

            current += timedelta(days=1)

        logger.info(f"Simulation complete — {len(self.portfolio.equity_curve)} trading days")

    async def _execute_signal(self, signal: Signal, mid_price: float) -> None:
        if signal.signal_type == SignalType.CLOSE_POSITION:
            # Close each leg at its actual BSM market value using:
            #   - current spot   = signal.strike (set to candle.close by strategy)
            #   - current IV     = signal.metadata["current_iv"]
            #   - DTE            = signal.metadata["dte"]
            #   - per-leg strike = pos.strike  (exact strike from the original fill)
            #   - option type    = pos.option_type
            from scipy.stats import norm as _n
            import math as _math

            reason      = signal.metadata.get("reason", "unknown")
            current_iv  = max(signal.metadata.get("current_iv", signal.metadata.get("entry_iv", 0.20)), 0.01)
            dte         = signal.metadata.get("dte", 21)
            T_close     = max(dte, 0) / 365.0
            spot        = signal.strike    # strategy sets signal.strike = candle.close

            positions = self.portfolio.positions_for_symbol(signal.symbol)
            # If the signal targets a specific strategy's positions (e.g. TailHedge
            # closing only its own puts, not the condor legs), filter by strategy_name.
            close_strategy = signal.metadata.get("close_strategy_name")
            if close_strategy:
                positions = [p for p in positions if p.strategy_name == close_strategy]
            total_pnl   = 0.0

            for pos in positions:
                # BSM price for this specific leg
                K = pos.strike
                sigma = current_iv
                if T_close <= 1e-6:
                    close_price = max(spot - K, 0.0) if pos.option_type == "CALL" else max(K - spot, 0.0)
                else:
                    sq = sigma * T_close ** 0.5
                    d1 = (_math.log(spot / K) + 0.5 * sigma**2 * T_close) / sq
                    d2 = d1 - sq
                    if pos.option_type == "CALL":
                        close_price = max(float(spot * _n.cdf(d1) - K * _n.cdf(d2)), 0.01)
                    else:
                        close_price = max(float(K * _n.cdf(-d2) - spot * _n.cdf(-d1)), 0.01)

                pnl = self.portfolio.close_position(pos.position_id, close_price)
                if pnl is not None:
                    total_pnl += pnl
                    direction = "SHORT" if pos.quantity < 0 else "LONG"
                    sign = "+" if pnl >= 0 else ""
                    logger.info(
                        f"CLOSED {pos.position_id} [{reason}]: "
                        f"{direction} {pos.symbol} {pos.option_type} "
                        f"K={pos.strike:.0f} exp={pos.expiry}  "
                        f"entry=${pos.entry_price:.3f}  close=${close_price:.3f}  "
                        f"pnl={sign}${pnl:.2f}"
                    )

            if positions:
                net_sign = "+" if total_pnl >= 0 else ""
                logger.info(
                    f"  NET {signal.symbol}: reason={reason}  IV={current_iv:.1%}  "
                    f"DTE={dte:.0f}  total_pnl={net_sign}${total_pnl:.2f}"
                )
            return

        orders = signal_to_orders(signal, mid_price)
        filled = False
        for order in orders:
            order_id = await self.executor.place_order(order)
            order = self.executor.orders[order_id]
            if order.status.value == "filled":
                qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
                self.portfolio.open_position(
                    symbol=order.symbol,
                    option_type=order.option_type,
                    strike=order.strike,
                    expiry=order.expiry,
                    quantity=qty,
                    entry_price=order.avg_fill_price,
                    strategy_name=signal.strategy_name,
                    signal_type=signal.signal_type.value,
                    llm_reasoning=signal.metadata.get("llm_reasoning", ""),
                )
                filled = True

        # Notify strategy that the entry was actually filled so exit monitoring activates
        if filled:
            for strategy in self.rule_engine.strategies:
                if hasattr(strategy, "confirm_entry"):
                    strategy.confirm_entry(signal.symbol)

    def _generate_market_data(self, regime: dict) -> Dict[datetime, List[Candle]]:
        """
        Generate historically-calibrated synthetic market data.

        Each regime specifies a GBM process with:
          - Historically accurate drift + starting vol
          - Mean-reverting vol process (GARCH-like) or phase-based (COVID)
          - Event overlays for discrete shocks (Aug 2024 VIX spike)
          - implied_vol and hv30 injected into each candle so strategy
            entry/exit decisions use regime-appropriate IV

        NOTE: The Bachelier ATM formula tracks P&L only via spot movement
        and theta decay — it does NOT reprice individual legs by moneyness.
        This means deep-ITM losses (COVID crash) are underestimated.  The
        results below show directionally correct behaviour (more losses in
        stress regimes) but absolute drawdowns are understated for extreme
        spot moves.  Full per-strike BSM pricing is the next milestone.
        """
        data: Dict[datetime, List[Candle]] = {}
        current = self.start_date

        is_phased = "phases" in regime
        rng = np.random.default_rng(seed=0)   # reproducible runs

        # ── Shared state ────────────────────────────────────────────────────
        initial_price = regime.get("initial_price", 450.0)
        symbol_state = {
            sym: {
                "price": initial_price if sym in ("SPY", "SPX") else initial_price / 4.5,
                "vol": regime.get("base_vol", 0.18) if not is_phased
                       else regime["phases"][0]["vol_start"],
                "trading_day": 0,
                "phase_idx": 0,
                "phase_tday": 0,  # trading days elapsed within current phase
            }
            for sym in self.config.symbols
        }

        while current <= self.end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            for sym in self.config.symbols:
                state = symbol_state[sym]
                base_price: float = state["price"]
                tday: int         = state["trading_day"]

                # ── Determine today's vol + drift ────────────────────────────
                if is_phased:
                    phases = regime["phases"]
                    pidx = state["phase_idx"]
                    ptday = state["phase_tday"]
                    phase = phases[min(pidx, len(phases) - 1)]

                    # Interpolate vol linearly across the phase
                    progress = min(ptday / max(phase["max_tdays"] - 1, 1), 1.0)
                    day_vol = float(np.interp(
                        progress,
                        [0.0, 1.0],
                        [phase["vol_start"], phase["vol_end"]],
                    ))
                    annual_drift = phase["annual_drift"]
                    iv_premium   = phase["iv_premium"]

                    # Advance phase if needed
                    if ptday >= phase["max_tdays"] and pidx < len(phases) - 1:
                        state["phase_idx"] += 1
                        state["phase_tday"] = 0
                    else:
                        state["phase_tday"] += 1

                else:
                    # Continuous GARCH-like process with event overlays
                    day_vol       = state["vol"]
                    annual_drift  = regime.get("annual_drift", 0.08)
                    iv_premium    = regime.get("iv_premium", 1.10)

                    # Check event overlays (by trading day index)
                    for ev in regime.get("events", []):
                        if ev["tday_start"] <= tday < ev["tday_end"]:
                            if "vol_override" in ev and ev["vol_override"] is not None:
                                day_vol = ev["vol_override"]
                            elif "vol_floor" in ev:
                                day_vol = max(day_vol, ev["vol_floor"])
                            if "drift_override" in ev:
                                annual_drift = ev["drift_override"]
                            iv_premium = ev.get("iv_premium", iv_premium)
                            break

                    # Mean-revert vol back toward long-run level (between days)
                    long_run   = regime.get("long_run_vol", day_vol)
                    rev_speed  = regime.get("vol_mean_rev", 0.05)
                    vov        = regime.get("vol_of_vol", 0.004)
                    vol_noise  = float(rng.normal(0, vov))
                    next_vol   = day_vol + rev_speed * (long_run - day_vol) + vol_noise

                    # Occasional vol jumps (macro shock)
                    jprob = regime.get("vol_jump_prob", 0.005)
                    if rng.random() < jprob:
                        jump = float(rng.exponential(regime.get("vol_jump_mean", 0.05)))
                        next_vol += jump
                        logger.debug(f"[{sym}] Vol jump on day {tday}: +{jump:.3f}")

                    state["vol"] = float(np.clip(next_vol, 0.04, 0.90))

                # Clamp vol to physical range
                day_vol = float(np.clip(day_vol, 0.04, 0.90))

                # ── Today's implied vol for candles ───────────────────────────
                # Add a small daily IV noise (±2% of the premium ratio) so
                # IV percentile readings aren't perfectly deterministic.
                iv_noise  = float(rng.normal(1.0, 0.02))
                today_iv  = float(np.clip(day_vol * iv_premium * iv_noise, 0.04, 0.90))
                today_hv30 = day_vol   # realized vol ≈ GBM process vol

                # ── Generate one daily bar ────────────────────────────────────
                # One candle per trading day so that the strategy's 30-day
                # lookback_days (candles) maps correctly to 30 calendar days.
                # Options strategies are daily decisions — intraday granularity
                # adds noise without accuracy gain.
                dt           = 1.0 / 252
                daily_drift  = annual_drift / 252
                rand         = float(rng.standard_normal())
                price_change = base_price * (
                    daily_drift * dt + day_vol * np.sqrt(dt) * rand
                )
                base_price = max(base_price + price_change, 1.0)

                spread = base_price * 0.0002 * (1 + day_vol)
                volume = int(float(rng.lognormal(12, 0.4)))
                candles = [Candle(
                    symbol=sym,
                    timestamp=current.replace(hour=16, minute=0),  # EOD bar
                    open=base_price  * (1 - day_vol * 0.005 * abs(float(rng.standard_normal()))),
                    high=base_price  * (1 + day_vol * 0.008 * abs(float(rng.standard_normal()))),
                    low=base_price   * (1 - day_vol * 0.008 * abs(float(rng.standard_normal()))),
                    close=base_price,
                    volume=volume,
                    timeframe="1d",
                    implied_vol=today_iv,
                    hv30=today_hv30,
                )]

                state["price"]        = base_price
                state["trading_day"] += 1
                data[current] = candles

            current += timedelta(days=1)

        return data

    def _build_results(self) -> dict:
        """Compile comprehensive backtest results."""
        metrics = self.portfolio.get_performance()
        fill_stats = self.executor.fill_summary()
        closed = self.portfolio.closed_positions

        result = {
            "status": "COMPLETED",
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "initial_capital": self.portfolio.initial_cash,
            "final_value": self.portfolio.portfolio_value,
            "total_pnl": self.portfolio.total_pnl,
            "realized_pnl": self.portfolio.total_realized_pnl,
            "unrealized_pnl": self.portfolio.total_unrealized_pnl,
            "open_positions": len(self.portfolio.positions),
            **fill_stats,
        }

        if metrics:
            result.update({
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "annualized_volatility": metrics.annualized_volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "max_drawdown": metrics.max_drawdown,
                "max_drawdown_duration_days": metrics.max_drawdown_duration_days,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "total_trades": metrics.total_trades,
                "trading_days": metrics.trading_days,
            })

        return result


# ============================================================================
# CLI
# ============================================================================


@click.group()
def cli():
    """QuantBot — Institutional Options Trading Engine"""
    setup_logging()


@cli.command()
@click.option("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
@click.option("--symbols", default="SPY", help="Comma-separated symbols")
@click.option("--capital", default=100_000.0, type=float, help="Starting capital")
@click.option("--no-llm", is_flag=True, default=False, help="Disable LLM validation")
@click.option("--data-source", default="mock", type=click.Choice(["mock", "tastytrade"]))
@click.option(
    "--regime",
    default="auto",
    type=click.Choice(["auto", "2024", "2022", "covid_2020", "generic"]),
    help="Vol regime (default: auto-detect from date range)",
)
@click.option("--no-hedge", is_flag=True, default=False,
              help="Disable the dynamic drawdown tail hedge")
def backtest(start: str, end: str, symbols: str, capital: float, no_llm: bool,
             data_source: str, regime: str, no_hedge: bool):
    """Run historical backtest with historically-calibrated vol regimes."""
    config = BotConfig(
        backtesting_mode=True,
        paper_trading=True,
        symbols=symbols.split(","),
        llm_enabled=not no_llm,
    )
    connector = TastytradeConnector() if data_source == "tastytrade" else MockDataConnector()
    runner = BacktestRunner(config, connector, enable_hedge=not no_hedge)
    results = asyncio.run(runner.run(start, end, regime_key=regime))

    hedge_label = "hedged" if not no_hedge else "unhedged"
    click.echo("\n" + "=" * 72)
    click.echo(f"  BACKTEST RESULTS  [{results.get('regime', 'unknown regime')}]  [{hedge_label}]")
    click.echo("=" * 72)

    fmt_fields = {
        "total_return", "annualized_return", "annualized_volatility",
        "max_drawdown", "win_rate",
    }
    ratio_fields = {"sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"}
    dollar_fields = {
        "initial_capital", "final_value", "total_pnl",
        "realized_pnl", "unrealized_pnl", "avg_win", "avg_loss",
        "total_commission", "total_slippage",
    }

    for key, value in results.items():
        if key in ("status", "regime"):
            continue
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            if key in fmt_fields:
                click.echo(f"  {label:<35} {value:.2%}")
            elif key in ratio_fields:
                click.echo(f"  {label:<35} {value:.4f}")
            elif key in dollar_fields:
                click.echo(f"  {label:<35} ${value:,.2f}")
            else:
                click.echo(f"  {label:<35} {value:.2f}")
        else:
            click.echo(f"  {label:<35} {value}")

    click.echo("=" * 72)


@cli.command()
@click.option("--paper", is_flag=True, default=True, help="Paper trading mode")
@click.option("--symbols", default="SPY,SPX", help="Comma-separated symbols")
@click.option("--no-llm", is_flag=True, default=False, help="Disable LLM validation")
def live(paper: bool, symbols: str, no_llm: bool):
    """Run bot in live (or paper) trading mode."""
    if not paper:
        click.confirm(
            "\n⚠️  LIVE MONEY MODE: Real capital will be at risk.\n"
            "   Ensure risk limits are configured in .env.\n"
            "   Continue?",
            abort=True,
        )

    config = BotConfig(
        backtesting_mode=False,
        paper_trading=paper,
        symbols=symbols.split(","),
        llm_enabled=not no_llm,
    )

    mode_str = "PAPER" if paper else "LIVE ⚠️"
    logger.info(f"Starting {mode_str} bot on {symbols}")
    bot = QuantBot(config)
    asyncio.run(bot.run())


@cli.command()
def status():
    """Show bot status (requires running bot process)."""
    click.echo("Status endpoint not yet wired to a running process.")
    click.echo("Check logs/quant_bot.log for live status.")


@cli.command()
@click.option("--lines", default=30, type=int, help="Number of log lines to show")
def logs(lines: int):
    """Tail the bot log file."""
    log_file = Path("logs/quant_bot.log")
    if not log_file.exists():
        click.echo("No log file found at logs/quant_bot.log")
        return
    with open(log_file) as f:
        tail = f.readlines()[-lines:]
    for line in tail:
        click.echo(line, nl=False)


@cli.command()
def llm_check():
    """Test connectivity to your local LLM."""
    config = BotConfig()
    click.echo(f"Checking LLM at {config.llm_base_url} (model: {config.llm_model})...")

    async def _check():
        client = LocalLLMClient(
            base_url=config.llm_base_url,
            model=config.llm_model,
            timeout=10,
        )
        ok = await client.is_available()
        if ok:
            click.echo("[OK] LLM server is reachable")
            # Send a quick smoke test
            resp = await client.chat(
                [{"role": "user", "content": 'Reply with exactly this JSON: {"status": "ok"}'}],
                json_mode=True,
            )
            click.echo(f"  Test response: {resp}")
        else:
            click.echo(f"[FAIL] LLM server not reachable at {config.llm_base_url}")
            click.echo("  Start Ollama:    ollama run llama3.1:8b")
            click.echo("  Start LM Studio: ensure local server is running on port 1234")
        await client.close()

    asyncio.run(_check())


@cli.command("tastytrade-check")
def tastytrade_check():
    """
    Test the Tastytrade API connection.

    Prints: connection status, account list, and current SPY / SPX IV metrics.
    Requires TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD in .env.

    Usage:
        python -m src.main tastytrade-check
    """
    async def _check():
        connector = TastytradeConnector()

        click.echo("\n" + "=" * 60)
        click.echo("  TASTYTRADE CONNECTION CHECK")
        click.echo("=" * 60)

        # ── Connect ──────────────────────────────────────────────────
        click.echo(f"\nConnecting as: {connector.username!r} ...")
        try:
            await connector.connect()
        except DeviceChallengeRequired as exc:
            click.echo(f"\n[OTP REQUIRED] {exc}")
            code = click.prompt("  Enter the OTP code from your phone/email").strip()
            try:
                await connector.connect(otp=code)
            except Exception as exc2:
                click.echo(f"[FAIL] OTP authentication failed: {exc2}")
                return
        except Exception as exc:
            click.echo(f"[FAIL] Connection error: {exc}")
            return

        click.echo("[OK] Connected\n")

        # ── Accounts ─────────────────────────────────────────────────
        accounts = connector.get_accounts()
        click.echo(f"Accounts ({len(accounts)}):")
        for acc in accounts:
            click.echo(
                f"  {acc.get('account-number', '?'):>12}  "
                f"{acc.get('account-type-name', '?'):<20} "
                f"{acc.get('nickname', '')}"
            )

        # ── Market metrics ───────────────────────────────────────────
        symbols = ["SPY", "SPX"]
        click.echo(f"\nMarket Metrics — {', '.join(symbols)}")
        click.echo("-" * 55)
        try:
            metrics = await connector.get_market_metrics(*symbols)
        except Exception as exc:
            click.echo(f"[FAIL] market-metrics error: {exc}")
            await connector.disconnect()
            return

        for sym in symbols:
            m = metrics.get(sym, {})
            if not m:
                click.echo(f"  {sym}: no data returned")
                continue
            iv     = m.get("iv", 0.0)
            iv_r   = m.get("iv_rank", 0.0)
            iv_p   = m.get("iv_pct", 0.0)
            hv30   = m.get("hv30", 0.0)
            iv_rv  = iv / max(hv30, 0.001)
            click.echo(
                f"  {sym:<5}  IV={iv:.1%}  IVR={iv_r:.0f}  "
                f"IV%ile={iv_p:.0f}  HV30={hv30:.1%}  IV/HV={iv_rv:.2f}"
            )

        # ── IV history seed ─────────────────────────────────────────
        click.echo("\nIV History Seed (30 days, SPY):")
        try:
            hist = await connector.seed_iv_history("SPY", days=30)
            click.echo(
                f"  min={min(hist):.2%}  max={max(hist):.2%}  "
                f"current={hist[-1]:.2%}  samples={len(hist)}"
            )
        except Exception as exc:
            click.echo(f"  [WARN] seed failed: {exc}")

        # ── Option chain ─────────────────────────────────────────────
        click.echo("\nOption chain (SPY, first 3 expirations):")
        try:
            exps = await connector.get_nested_chain("SPY")
            for exp in exps[:3]:
                n_strikes = len(exp.get("strikes", []))
                click.echo(
                    f"  {exp.get('expiration-date', '?')}  "
                    f"DTE={exp.get('days-to-expiration', '?'):>3}  "
                    f"strikes={n_strikes}"
                )
        except Exception as exc:
            click.echo(f"  [WARN] option chain error: {exc}")

        click.echo("\n" + "=" * 60)
        await connector.disconnect()
        click.echo("[OK] Disconnected cleanly\n")

    asyncio.run(_check())


if __name__ == "__main__":
    cli()
