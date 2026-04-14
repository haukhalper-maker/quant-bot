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
from src.market_data import MockDataConnector, TastytradeConnector, Candle, Tick
from src.analysis import AnalysisEngine
from src.strategy import (
    VolatilityMeanReversionStrategy,
    GammaScalpingStrategy,
    GammaImbalanceTrader,
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

        # Collect signals from all strategies
        signals = []
        for strategy in self.rule_engine.strategies:
            sig = await strategy.on_tick(tick)
            if sig:
                signals.append(sig)

        if signals:
            await self._process_signals(signals, current_price=tick.price)

    async def _on_candle(self, candle: Candle) -> None:
        """Process a candle through the full pipeline."""
        self.analysis.on_price(candle.symbol, candle.close)
        self.executor.set_market_price(candle.symbol, candle.close)

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


class BacktestRunner:
    """
    Full event-driven backtest with strategy signals, LLM validation,
    realistic execution, and portfolio P&L tracking.
    """

    def __init__(self, config: BotConfig, data_connector=None):
        self.config = config
        self.data_connector = data_connector or MockDataConnector()
        self.analysis = AnalysisEngine()

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

    async def run(self, start_date: str, end_date: str) -> dict:
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")

        logger.info(f"Backtest: {start_date} → {end_date} | LLM={'on' if self.reasoning_engine else 'off'}")

        try:
            await self.data_connector.connect()
            market_data = self._generate_market_data()
            await self._run_simulation(market_data)
            return self._build_results()
        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            return {"status": "FAILED", "error": str(e)}
        finally:
            await self.data_connector.disconnect()
            if self.reasoning_engine:
                await self.reasoning_engine.client.close()

    async def _run_simulation(self, market_data: Dict[datetime, List[Candle]]) -> None:
        current = self.start_date
        while current <= self.end_date:
            if current.weekday() >= 5:  # Skip weekends
                current += timedelta(days=1)
                continue

            day_candles = market_data.get(current, [])
            daily_pnl = 0.0

            for candle in day_candles:
                self.analysis.on_price(candle.symbol, candle.close)
                self.executor.set_market_price(candle.symbol, candle.close)

                # Mark open positions
                self.portfolio.mark_all({candle.symbol: candle.close})

                # Collect strategy signals
                signals = []
                for strategy in self.rule_engine.strategies:
                    sig = await strategy.on_candle(candle)
                    if sig:
                        signals.append(sig)

                if signals and not self.circuit_breaker.is_tripped:
                    approved = await self.rule_engine.evaluate_signals(
                        signals,
                        portfolio=self.portfolio,
                        analysis_engine=self.analysis,
                    )
                    for signal in approved:
                        await self._execute_signal(signal, candle.close)

            # EOD: snapshot portfolio
            current_value = self.portfolio.portfolio_value
            daily_pnl = current_value - self._prev_portfolio_value
            self._prev_portfolio_value = current_value
            self.portfolio.snapshot(daily_pnl=daily_pnl)

            current += timedelta(days=1)

        logger.info(f"Simulation complete — {len(self.portfolio.equity_curve)} trading days")

    async def _execute_signal(self, signal: Signal, mid_price: float) -> None:
        if signal.signal_type == SignalType.CLOSE_POSITION:
            for pos in self.portfolio.positions_for_symbol(signal.symbol):
                self.portfolio.close_position(pos.position_id, mid_price)
            return

        orders = signal_to_orders(signal, mid_price)
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

    def _generate_market_data(self) -> Dict[datetime, List[Candle]]:
        """
        Generate realistic synthetic market data for the backtest period.

        Model: Geometric Brownian Motion with:
          - Mean-reverting volatility (Heston-like vol-of-vol)
          - Occasional vol spikes (jump process)
          - Intraday volume profile (U-shaped)
          - Bid/ask spread proportional to vol
        """
        data: Dict[datetime, List[Candle]] = {}
        current = self.start_date

        # Per-symbol state
        symbol_state = {
            sym: {"price": 450.0 if sym in ("SPY", "SPX") else 100.0, "vol": 0.18}
            for sym in self.config.symbols
        }

        while current <= self.end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            for sym in self.config.symbols:
                state = symbol_state[sym]
                candles = []
                base_price = state["price"]
                vol = state["vol"]

                for minute in range(390):
                    time_of_day = minute / 390
                    # U-shaped intraday vol (higher at open/close)
                    intraday_vol_mult = 1.0 + 0.5 * (1 - 4 * (time_of_day - 0.5) ** 2)

                    # Per-minute price change
                    dt = 1 / (252 * 390)
                    daily_drift = 0.08 / 252          # ~8% annual drift
                    rand = np.random.standard_normal()
                    price_change = base_price * (
                        daily_drift * dt + vol * intraday_vol_mult * np.sqrt(dt) * rand
                    )
                    base_price = max(base_price + price_change, 1.0)

                    # Intraday volume (U-shaped)
                    vol_factor = 1.0 + 1.5 * abs(time_of_day - 0.5)
                    volume = int(np.random.lognormal(10, 0.5) * vol_factor)

                    spread = base_price * 0.0002 * (1 + vol)
                    candles.append(Candle(
                        symbol=sym,
                        timestamp=current.replace(hour=9, minute=30) + timedelta(minutes=minute),
                        open=base_price - spread / 2,
                        high=base_price + abs(np.random.normal(0, base_price * vol * 0.002)),
                        low=base_price - abs(np.random.normal(0, base_price * vol * 0.002)),
                        close=base_price + spread / 2 * np.random.choice([-1, 1]),
                        volume=volume,
                        timeframe="1m",
                    ))

                state["price"] = base_price
                # Vol mean reversion with jumps
                vol_jump = np.random.exponential(0.05) if np.random.random() < 0.02 else 0
                vol = vol + 0.03 * (0.18 - vol) + np.random.normal(0, 0.005) + vol_jump
                state["vol"] = float(np.clip(vol, 0.05, 0.80))

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
@click.option("--symbols", default="SPY,SPX", help="Comma-separated symbols")
@click.option("--capital", default=100_000.0, type=float, help="Starting capital")
@click.option("--no-llm", is_flag=True, default=False, help="Disable LLM validation")
@click.option("--data-source", default="mock", type=click.Choice(["mock", "tastytrade"]))
def backtest(start: str, end: str, symbols: str, capital: float, no_llm: bool, data_source: str):
    """Run historical backtest."""
    config = BotConfig(
        backtesting_mode=True,
        paper_trading=True,
        symbols=symbols.split(","),
        llm_enabled=not no_llm,
    )
    connector = TastytradeConnector() if data_source == "tastytrade" else MockDataConnector()
    runner = BacktestRunner(config, connector)
    results = asyncio.run(runner.run(start, end))

    click.echo("\n" + "=" * 72)
    click.echo("  BACKTEST RESULTS")
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
        if key in ("status",):
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


if __name__ == "__main__":
    cli()
