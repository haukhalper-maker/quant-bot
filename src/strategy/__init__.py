"""
Strategy Engine — Signal generation with LLM-augmented decision making.

Architecture:
  - Each Strategy is a pure signal generator (no execution knowledge)
  - RuleEngine aggregates multi-strategy signals and filters conflicts
  - LLM validation is injected at the RuleEngine level, not the strategy level
    (keeps strategies clean and testable in isolation)

Strategies:
  - VolatilityMeanReversionStrategy: IV percentile mean reversion (straddle / iron condor)
  - GammaScalpingStrategy:           Long gamma, delta-neutral scalping
  - GammaImbalanceTrader:            Trade dealer gamma walls
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


# ============================================================================
# SIGNAL MODEL
# ============================================================================


class SignalType(Enum):
    BUY_CALL = "buy_call"
    SELL_CALL = "sell_call"
    BUY_PUT = "buy_put"
    SELL_PUT = "sell_put"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    CLOSE_POSITION = "close_position"
    GAMMA_SCALP = "gamma_scalp"
    VOLATILITY_TRADE = "volatility_trade"


@dataclass
class Signal:
    """Trading signal emitted by a strategy."""

    signal_type: SignalType
    symbol: str
    timestamp: datetime
    strike: float
    expiry: str
    confidence: float               # 0.0-1.0 strategy-level confidence
    position_size: int              # suggested contracts
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ""
    metadata: Dict = field(default_factory=dict)

    def with_llm(self, size_multiplier: float, stop_loss: Optional[float],
                  take_profit: Optional[float]) -> "Signal":
        """Return a copy of this signal adjusted by LLM recommendations."""
        import math
        adjusted = Signal(
            signal_type=self.signal_type,
            symbol=self.symbol,
            timestamp=self.timestamp,
            strike=self.strike,
            expiry=self.expiry,
            confidence=self.confidence,
            position_size=max(1, math.floor(self.position_size * size_multiplier)),
            stop_loss=stop_loss or self.stop_loss,
            take_profit=take_profit or self.take_profit,
            strategy_name=self.strategy_name,
            metadata=dict(self.metadata),
        )
        return adjusted


# ============================================================================
# STRATEGY BASE
# ============================================================================


class Strategy(ABC):
    """
    Abstract base strategy.
    Strategies are stateless signal generators — they track their own price/vol
    history internally but know nothing about orders, positions, or the LLM.
    """

    def __init__(self, name: str):
        self.name = name
        self.signals_generated = 0
        logger.info(f"Strategy '{name}' initialized")

    @abstractmethod
    async def on_tick(self, tick) -> Optional[Signal]:
        pass

    @abstractmethod
    async def on_candle(self, candle) -> Optional[Signal]:
        pass

    @abstractmethod
    async def on_greek_update(self, greeks) -> Optional[Signal]:
        pass

    def _emit(self, signal: Signal) -> Signal:
        self.signals_generated += 1
        logger.info(
            f"[{self.name}] Signal #{self.signals_generated}: "
            f"{signal.signal_type.value} {signal.symbol} "
            f"conf={signal.confidence:.0%} size={signal.position_size}"
        )
        return signal


# ============================================================================
# VOLATILITY MEAN REVERSION STRATEGY
# ============================================================================


class VolatilityMeanReversionStrategy(Strategy):
    """
    Core vol mean reversion: buy cheap IV, sell rich IV.

    Entry rules:
      - BUY vol (straddle)  when IV%ile < buy_threshold
      - SELL vol (iron condor) when IV%ile > sell_threshold
      - IV/RV filter: only buy if IV/RV < 1.2; only sell if IV/RV > 1.4
      - Confidence scales linearly with distance from threshold

    Exit rules:
      - Close long vol when IV%ile > 65 or days_held >= max_hold_days
      - Close short vol when IV%ile < 35 or days_held >= max_hold_days
    """

    def __init__(
        self,
        lookback_days: int = 30,
        buy_threshold: float = 25.0,
        sell_threshold: float = 75.0,
        min_confidence: float = 0.45,
        min_signal_interval_hours: float = 4.0,
        max_hold_days: int = 21,
        iv_rv_buy_max: float = 1.2,
        iv_rv_sell_min: float = 1.4,
    ):
        super().__init__("VolMeanReversion")
        self.lookback_days = lookback_days
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence
        self.min_signal_interval = timedelta(hours=min_signal_interval_hours)
        self.max_hold_days = max_hold_days
        self.iv_rv_buy_max = iv_rv_buy_max
        self.iv_rv_sell_min = iv_rv_sell_min

        self._price_history: Dict[str, List[float]] = {}
        self._iv_history: Dict[str, List[float]] = {}
        self._positions: Dict[str, Dict] = {}       # symbol → {type, entry_time}
        self._last_signal: Dict[str, datetime] = {} # symbol → last signal time

    async def on_tick(self, tick) -> Optional[Signal]:
        sym = tick.symbol
        self._price_history.setdefault(sym, []).append(tick.price)
        # Trim to avoid unbounded growth
        max_len = self.lookback_days * 390  # 1-min bars
        if len(self._price_history[sym]) > max_len:
            self._price_history[sym] = self._price_history[sym][-max_len:]
        return None

    async def on_candle(self, candle) -> Optional[Signal]:
        sym = candle.symbol
        hist = self._price_history.setdefault(sym, [])
        hist.append(candle.close)
        if len(hist) > self.lookback_days * 2:
            self._price_history[sym] = hist[-self.lookback_days * 2:]

        if len(hist) < 10:
            return None

        realized_vol = self._realized_vol(sym)
        implied_vol = self._get_iv(sym)
        if implied_vol is None:
            return None

        self._iv_history.setdefault(sym, []).append(implied_vol)
        iv_hist = self._iv_history[sym]
        if len(iv_hist) > self.lookback_days * 2:
            self._iv_history[sym] = iv_hist[-self.lookback_days * 2:]

        iv_pct = self._iv_percentile(sym, implied_vol)
        iv_rv = implied_vol / max(realized_vol, 0.001)

        return await self._evaluate(sym, iv_pct, realized_vol, implied_vol, iv_rv, candle)

    async def on_greek_update(self, greeks) -> Optional[Signal]:
        return None

    def _realized_vol(self, symbol: str) -> float:
        prices = self._price_history.get(symbol, [])
        n = min(len(prices), self.lookback_days)
        if n < 2:
            return 0.20
        log_rets = np.diff(np.log(prices[-n:]))
        return float(np.std(log_rets) * np.sqrt(252))

    def _get_iv(self, symbol: str) -> Optional[float]:
        """
        Placeholder: in live mode this comes from options market data.
        In paper/backtest mode we simulate with a realistic mean-reverting process.
        """
        hist = self._iv_history.get(symbol, [])
        if hist:
            # AR(1) mean-reverting IV process
            prev = hist[-1]
            long_run_mean = 0.20
            mean_rev_speed = 0.05
            noise = np.random.normal(0, 0.008)
            iv = prev + mean_rev_speed * (long_run_mean - prev) + noise
            return float(max(0.05, min(0.80, iv)))
        # Initial seed
        return float(np.random.normal(0.20, 0.04))

    def _iv_percentile(self, symbol: str, current_iv: float) -> float:
        hist = self._iv_history.get(symbol, [])
        if len(hist) < 5:
            if current_iv < 0.12:
                return 15.0
            elif current_iv > 0.30:
                return 85.0
            return 50.0
        return float(sum(1 for v in hist if current_iv >= v) / len(hist) * 100)

    def _can_signal(self, symbol: str) -> bool:
        last = self._last_signal.get(symbol)
        if last is None:
            return True
        return datetime.utcnow() - last >= self.min_signal_interval

    def _next_expiry(self, weeks_out: int = 4) -> str:
        """Return the next options expiry (weekly, ~N weeks out)."""
        today = datetime.utcnow().date()
        # Roll to next Friday
        days_to_friday = (4 - today.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        expiry = today + timedelta(days=days_to_friday + weeks_out * 7)
        return expiry.strftime("%Y-%m-%d")

    async def _evaluate(
        self, sym: str, iv_pct: float, realized_vol: float,
        implied_vol: float, iv_rv: float, candle
    ) -> Optional[Signal]:
        if not self._can_signal(sym):
            return None

        pos = self._positions.get(sym, {})
        now = datetime.utcnow()

        # --- BUY vol: IV is cheap ---
        if iv_pct < self.buy_threshold and iv_rv < self.iv_rv_buy_max and pos.get("type") != "long":
            conf = (self.buy_threshold - iv_pct) / self.buy_threshold
            if conf < self.min_confidence:
                return None
            self._last_signal[sym] = now
            self._positions[sym] = {"type": "long", "entry_time": now, "entry_iv_pct": iv_pct}
            return self._emit(Signal(
                signal_type=SignalType.STRADDLE,
                symbol=sym,
                timestamp=now,
                strike=candle.close,
                expiry=self._next_expiry(weeks_out=4),
                confidence=min(conf, 1.0),
                position_size=1,
                strategy_name=self.name,
                metadata={
                    "iv_percentile": iv_pct,
                    "iv_rv_ratio": iv_rv,
                    "realized_vol": realized_vol,
                    "implied_vol": implied_vol,
                    "reason": "buy_cheap_vol",
                },
            ))

        # --- SELL vol: IV is rich ---
        if iv_pct > self.sell_threshold and iv_rv > self.iv_rv_sell_min and pos.get("type") != "short":
            conf = (iv_pct - self.sell_threshold) / (100 - self.sell_threshold)
            if conf < self.min_confidence:
                return None
            self._last_signal[sym] = now
            self._positions[sym] = {"type": "short", "entry_time": now, "entry_iv_pct": iv_pct}
            return self._emit(Signal(
                signal_type=SignalType.IRON_CONDOR,
                symbol=sym,
                timestamp=now,
                strike=candle.close,
                expiry=self._next_expiry(weeks_out=6),
                confidence=min(conf, 1.0),
                position_size=1,
                strategy_name=self.name,
                metadata={
                    "iv_percentile": iv_pct,
                    "iv_rv_ratio": iv_rv,
                    "realized_vol": realized_vol,
                    "implied_vol": implied_vol,
                    "reason": "sell_rich_vol",
                },
            ))

        # --- EXIT: position has mean reverted ---
        if pos:
            entry_time = pos.get("entry_time", now)
            days_held = (now - entry_time).days
            pos_type = pos.get("type")

            should_close = (
                (pos_type == "long" and (iv_pct > 65 or days_held >= self.max_hold_days))
                or (pos_type == "short" and (iv_pct < 35 or days_held >= self.max_hold_days))
            )
            if should_close:
                reason = "mean_reversion_complete" if days_held < self.max_hold_days else "max_hold_exceeded"
                self._positions[sym] = {}
                self._last_signal[sym] = now
                return self._emit(Signal(
                    signal_type=SignalType.CLOSE_POSITION,
                    symbol=sym,
                    timestamp=now,
                    strike=candle.close,
                    expiry=self._next_expiry(),
                    confidence=0.85,
                    position_size=1,
                    strategy_name=self.name,
                    metadata={"iv_percentile": iv_pct, "days_held": days_held, "reason": reason},
                ))

        return None


# ============================================================================
# GAMMA SCALPING STRATEGY
# ============================================================================


class GammaScalpingStrategy(Strategy):
    """
    Long gamma delta-neutral scalping.

    Concept: Hold a long straddle, then delta-hedge by trading the underlying
    (or futures) as the position accumulates delta. Each hedge locks in P&L
    proportional to gamma * (delta move)^2.

    Entry: when gamma-adjusted P&L potential exceeds transaction costs.
    Rebalance: when |position_delta| exceeds rebalance_delta threshold.
    Exit: when theta decay overwhelms gamma harvesting, or max hold exceeded.
    """

    def __init__(
        self,
        gamma_threshold: float = 0.005,     # minimum portfolio gamma to justify scalping
        rebalance_delta: float = 0.10,       # rebalance when |delta| > this fraction
        min_move_pct: float = 0.003,         # minimum underlying move to trigger scalp (30bps)
        max_hold_hours: int = 48,
    ):
        super().__init__("GammaScalp")
        self.gamma_threshold = gamma_threshold
        self.rebalance_delta = rebalance_delta
        self.min_move_pct = min_move_pct
        self.max_hold_hours = max_hold_hours

        self._entry_prices: Dict[str, float] = {}
        self._last_hedge_price: Dict[str, float] = {}
        self._entry_time: Dict[str, datetime] = {}
        self._portfolio_delta: Dict[str, float] = {}
        self._portfolio_gamma: Dict[str, float] = {}

    def update_greeks(self, symbol: str, delta: float, gamma: float) -> None:
        """Called by the main loop after each Greek update."""
        self._portfolio_delta[symbol] = delta
        self._portfolio_gamma[symbol] = gamma

    async def on_tick(self, tick) -> Optional[Signal]:
        sym = tick.symbol
        gamma = self._portfolio_gamma.get(sym, 0.0)
        delta = self._portfolio_delta.get(sym, 0.0)
        last_hedge = self._last_hedge_price.get(sym, tick.price)

        if gamma < self.gamma_threshold:
            return None

        price_move_pct = abs(tick.price - last_hedge) / max(last_hedge, 1.0)
        if price_move_pct < self.min_move_pct:
            return None

        # Delta has drifted: need to rebalance
        if abs(delta) >= self.rebalance_delta:
            self._last_hedge_price[sym] = tick.price
            side = SignalType.SELL_CALL if delta > 0 else SignalType.BUY_CALL
            return self._emit(Signal(
                signal_type=SignalType.GAMMA_SCALP,
                symbol=sym,
                timestamp=datetime.utcnow(),
                strike=tick.price,
                expiry="",
                confidence=0.75,
                position_size=1,
                strategy_name=self.name,
                metadata={
                    "delta": delta,
                    "gamma": gamma,
                    "price_move_pct": price_move_pct,
                    "hedge_direction": "sell" if delta > 0 else "buy",
                },
            ))
        return None

    async def on_candle(self, candle) -> Optional[Signal]:
        return None

    async def on_greek_update(self, greeks) -> Optional[Signal]:
        return None


# ============================================================================
# GAMMA IMBALANCE TRADER
# ============================================================================


class GammaImbalanceTrader(Strategy):
    """
    Trades dealer gamma concentrations (gamma walls).

    When dealers are net short gamma at key strike levels, the market tends
    to be more volatile and directional near those strikes. When long gamma,
    the market reverts.

    Requires: open interest by strike and calculated dealer gamma positioning.
    """

    def __init__(self, imbalance_threshold: float = 1.5):
        super().__init__("GammaImbalance")
        self.imbalance_threshold = imbalance_threshold
        self._gamma_skew: Dict[str, Dict[float, float]] = {}  # symbol → {strike: dealer_gamma}

    def update_gamma_skew(self, symbol: str, gamma_skew: Dict[float, float]) -> None:
        """Feed dealer gamma positioning data (from open interest analysis)."""
        self._gamma_skew[symbol] = gamma_skew

    async def on_tick(self, tick) -> Optional[Signal]:
        sym = tick.symbol
        skew = self._gamma_skew.get(sym, {})
        if not skew:
            return None

        near = {
            k: v for k, v in skew.items()
            if abs(k - tick.price) / max(tick.price, 1) < 0.03  # within 3% of spot
        }
        if not near:
            return None

        net_gamma = sum(near.values())
        if abs(net_gamma) < 0.001:
            return None

        # Short gamma environment → expect trending move, trade with momentum
        if net_gamma < -self.imbalance_threshold * abs(net_gamma):
            return self._emit(Signal(
                signal_type=SignalType.STRADDLE,
                symbol=sym,
                timestamp=datetime.utcnow(),
                strike=tick.price,
                expiry="",
                confidence=0.60,
                position_size=1,
                strategy_name=self.name,
                metadata={"net_dealer_gamma": net_gamma, "regime": "short_gamma"},
            ))
        return None

    async def on_candle(self, candle) -> Optional[Signal]:
        return None

    async def on_greek_update(self, greeks) -> Optional[Signal]:
        return None


# ============================================================================
# RULE ENGINE (with LLM integration)
# ============================================================================


class RuleEngine:
    """
    Aggregates signals from multiple strategies and applies:
      1. Pre-filters (position limits, market hours, cooldown)
      2. LLM validation (if reasoning engine provided)
      3. Conflict resolution (opposite signals on same symbol → skip)
      4. Size adjustment from LLM size multiplier

    Usage:
        engine = RuleEngine(reasoning_engine=trade_reasoning_engine)
        engine.register_strategy(VolatilityMeanReversionStrategy())
        approved = await engine.evaluate_signals(signals, portfolio, analysis_engine)
    """

    def __init__(self, reasoning_engine=None):
        """
        reasoning_engine: optional TradeReasoningEngine instance.
                          If None, signals pass through without LLM validation.
        """
        self.strategies: List[Strategy] = []
        self.rules: List[Callable] = []
        self.reasoning_engine = reasoning_engine
        self.min_llm_confidence = 0.45  # Reject LLM decisions below this threshold
        logger.info(
            f"RuleEngine initialized — LLM reasoning: "
            f"{'enabled' if reasoning_engine else 'disabled'}"
        )

    def register_strategy(self, strategy: Strategy) -> None:
        self.strategies.append(strategy)
        logger.info(f"Registered strategy: {strategy.name}")

    def add_rule(self, rule: Callable) -> None:
        """Add a synchronous pre-filter rule. rule(signal) → bool."""
        self.rules.append(rule)

    async def evaluate_signals(
        self,
        signals: List[Signal],
        portfolio=None,
        analysis_engine=None,
    ) -> List[Signal]:
        """
        Run signals through rules and optional LLM validation.
        Returns list of approved (and possibly size-adjusted) signals.
        """
        if not signals:
            return []

        # Step 1: Apply synchronous pre-filter rules
        filtered = [s for s in signals if all(r(s) for r in self.rules)]
        rejected = len(signals) - len(filtered)
        if rejected:
            logger.debug(f"RuleEngine: {rejected} signal(s) rejected by pre-filters")

        if not filtered:
            return []

        # Step 2: Conflict resolution — skip opposing signals on same symbol
        filtered = self._resolve_conflicts(filtered)

        # Step 3: LLM validation
        if self.reasoning_engine is None:
            return filtered

        approved = []
        for signal in filtered:
            ctx = self._build_context(signal, portfolio, analysis_engine)
            decision = await self.reasoning_engine.evaluate_signal(ctx)

            if decision.action in ("skip",) or decision.confidence < self.min_llm_confidence:
                logger.info(
                    f"LLM rejected signal: {signal.signal_type.value} {signal.symbol} "
                    f"— action={decision.action} conf={decision.confidence:.0%} "
                    f"[{decision.source}] reason: {decision.reasoning[:80]}"
                )
                continue

            if decision.action in ("enter", "hold", "reduce"):
                adjusted = signal.with_llm(
                    size_multiplier=decision.position_size_multiplier,
                    stop_loss=decision.suggested_stop_loss,
                    take_profit=decision.suggested_take_profit,
                )
                adjusted.metadata["llm_reasoning"] = decision.reasoning
                adjusted.metadata["llm_risks"] = decision.key_risks
                adjusted.metadata["llm_action"] = decision.action
                adjusted.metadata["llm_source"] = decision.source
                approved.append(adjusted)
                logger.info(
                    f"LLM approved: {signal.signal_type.value} {signal.symbol} "
                    f"size={adjusted.position_size} "
                    f"[{decision.source}] {decision.reasoning[:80]}"
                )
            elif decision.action == "exit":
                # LLM wants to exit — convert to close signal
                close_signal = Signal(
                    signal_type=SignalType.CLOSE_POSITION,
                    symbol=signal.symbol,
                    timestamp=signal.timestamp,
                    strike=signal.strike,
                    expiry=signal.expiry,
                    confidence=decision.confidence,
                    position_size=signal.position_size,
                    strategy_name=signal.strategy_name,
                    metadata={"llm_reasoning": decision.reasoning, "llm_source": decision.source},
                )
                approved.append(close_signal)

        return approved

    def _resolve_conflicts(self, signals: List[Signal]) -> List[Signal]:
        """Remove opposing signals for the same symbol within the same batch."""
        by_symbol: Dict[str, List[Signal]] = {}
        for s in signals:
            by_symbol.setdefault(s.symbol, []).append(s)

        resolved = []
        for sym, sym_signals in by_symbol.items():
            if len(sym_signals) == 1:
                resolved.extend(sym_signals)
                continue
            # Conflicting? Keep highest-confidence signal
            types = {s.signal_type for s in sym_signals}
            buying = {SignalType.STRADDLE, SignalType.STRANGLE, SignalType.BUY_CALL, SignalType.BUY_PUT}
            selling = {SignalType.IRON_CONDOR, SignalType.SELL_CALL, SignalType.SELL_PUT}
            if types & buying and types & selling:
                best = max(sym_signals, key=lambda s: s.confidence)
                logger.warning(f"Conflicting signals for {sym} — keeping highest confidence: {best.signal_type.value}")
                resolved.append(best)
            else:
                resolved.extend(sym_signals)

        return resolved

    def _build_context(self, signal: Signal, portfolio, analysis_engine) -> "TradeContext":
        """Build a TradeContext for LLM evaluation from available data."""
        from src.llm import TradeContext

        # Pull portfolio Greeks
        p_delta = getattr(portfolio, "net_delta", 0.0) if portfolio else 0.0
        p_gamma = getattr(portfolio, "net_gamma", 0.0) if portfolio else 0.0
        p_vega = getattr(portfolio, "net_vega", 0.0) if portfolio else 0.0
        p_theta = getattr(portfolio, "net_theta", 0.0) if portfolio else 0.0

        # Pull vol metrics
        iv_pct = 50.0
        iv_rank = 50.0
        implied_vol = 0.20
        realized_vol = 0.20
        recent_returns = []
        recent_ivs = []
        vol_poc = signal.strike

        if analysis_engine is not None:
            metrics = analysis_engine.vol_metrics(signal.symbol)
            iv_pct = metrics.get("iv_percentile", 50.0)
            iv_rank = metrics.get("iv_rank", 50.0)
            implied_vol = metrics.get("iv", 0.20)
            realized_vol = metrics.get("realized_vol", 0.20)
            recent_returns = analysis_engine.vol_analyzer.recent_returns(signal.symbol)
            recent_ivs = analysis_engine.vol_analyzer.recent_iv_history(signal.symbol)
            vp = analysis_engine.get_volume_profile(signal.symbol)
            if vp:
                vol_poc = vp.poc

        # Portfolio positions for this symbol
        existing = []
        if portfolio is not None:
            existing = portfolio.open_positions_summary(signal.symbol)

        return TradeContext(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            spot_price=signal.strike,
            iv_percentile=iv_pct,
            iv_rank=iv_rank,
            implied_vol=implied_vol,
            realized_vol=realized_vol,
            portfolio_delta=p_delta,
            portfolio_gamma=p_gamma,
            portfolio_vega=p_vega,
            portfolio_theta=p_theta,
            signal_type=signal.signal_type.value,
            signal_confidence=signal.confidence,
            signal_strike=signal.strike,
            signal_expiry=signal.expiry,
            signal_position_size=signal.position_size,
            recent_returns=recent_returns,
            recent_iv_history=recent_ivs,
            volume_poc=vol_poc,
            existing_positions=existing,
        )


# Back-compat alias
GammaScalping = GammaScalpingStrategy
