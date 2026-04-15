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
from scipy.stats import norm as _norm


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

    Exit rules (checked every candle, no cooldown):
      1. 50% of max profit reached (short: premium decayed 50%; long: premium up 50%)
      2. Stop loss at 200% of premium collected/paid
      3. 21 DTE — close before gamma risk explodes near expiry
      4. IV mean reversion complete (IV%ile crossed back to neutral)
      5. max_hold_days exceeded
    """

    def __init__(
        self,
        lookback_days: int = 30,
        buy_threshold: float = 25.0,
        sell_threshold: float = 75.0,
        min_confidence: float = 0.45,
        min_signal_interval_hours: float = 4.0,
        max_hold_days: int = 45,
        iv_rv_buy_max: float = 1.2,
        iv_rv_sell_min: float = 1.4,
        take_profit_pct: float = 0.50,   # close short at 50% of max profit
        stop_loss_mult: float = 2.00,    # stop loss at 200% of premium collected
        dte_close: int = 21,             # close when DTE reaches this level
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
        self.take_profit_pct = take_profit_pct
        self.stop_loss_mult = stop_loss_mult
        self.dte_close = dte_close

        self._price_history: Dict[str, List[float]] = {}
        self._iv_history: Dict[str, List[float]] = {}
        # symbol → {type, entry_time, entry_premium, expiry, entry_iv_pct, confirmed}
        # 'confirmed' is False until BacktestRunner/Bot calls confirm_entry() after fill.
        self._positions: Dict[str, Dict] = {}
        self._last_signal: Dict[str, datetime] = {} # symbol → last entry signal time

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
        # Use live IV from candle if the connector injected it; otherwise AR(1) simulation.
        iv_override = getattr(candle, "implied_vol", None)
        implied_vol = self._get_iv(sym, override=iv_override)
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

    def seed_iv_history(self, symbol: str, history: List[float]) -> None:
        """
        Pre-populate IV history for `symbol` with real data.

        Called at bot startup by TastytradeConnector.seed_iv_history().
        Overrides any existing history so the initial _iv_percentile()
        reading is anchored to real market data rather than random noise.
        """
        if history:
            self._iv_history[symbol] = list(history)
            logger.info(
                f"[{self.name}] Seeded {len(history)}-day IV history for {symbol}: "
                f"current={history[-1]:.2%}  min={min(history):.2%}  "
                f"max={max(history):.2%}"
            )

    def _get_iv(self, symbol: str, override: Optional[float] = None) -> Optional[float]:
        """
        Return implied vol for `symbol`.

        Priority:
          1. `override` — real IV injected from TastytradeConnector via
             candle.implied_vol (set by QuantBot._on_candle before calling
             strategy.on_candle).  Appended to history so percentile
             tracking stays current.
          2. AR(1) continuation — if history exists but no live feed,
             extend from the last known value.  Uses a tighter noise
             parameter (0.005) than the old simulation (0.008) so the
             synthetic readings don't drift far from the seeded anchor.
          3. Random seed — first call only, no history, no live feed.
             Fallback for pure offline/backtest use.
        """
        if override is not None:
            return float(max(0.04, min(0.90, override)))

        hist = self._iv_history.get(symbol, [])
        if hist:
            # AR(1) continuation from last real (or simulated) value
            prev = hist[-1]
            long_run_mean = 0.20
            mean_rev_speed = 0.05
            noise = np.random.normal(0, 0.005)
            iv = prev + mean_rev_speed * (long_run_mean - prev) + noise
            return float(max(0.04, min(0.90, iv)))

        # No history at all — seed with a plausible random value
        return float(np.clip(np.random.normal(0.20, 0.04), 0.05, 0.60))

    def _iv_percentile(self, symbol: str, current_iv: float) -> float:
        hist = self._iv_history.get(symbol, [])
        if len(hist) < 5:
            if current_iv < 0.12:
                return 15.0
            elif current_iv > 0.30:
                return 85.0
            return 50.0
        return float(sum(1 for v in hist if current_iv >= v) / len(hist) * 100)

    def _can_signal(self, symbol: str, now: datetime) -> bool:
        last = self._last_signal.get(symbol)
        if last is None:
            return True
        return now - last >= self.min_signal_interval

    def _next_expiry(self, weeks_out: int = 4, ref_date=None) -> str:
        """Return the next options expiry (weekly, ~N weeks out) from ref_date."""
        today = ref_date or datetime.utcnow().date()
        # Roll to next Friday
        days_to_friday = (4 - today.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        expiry = today + timedelta(days=days_to_friday + weeks_out * 7)
        return expiry.strftime("%Y-%m-%d")

    def confirm_entry(self, symbol: str) -> None:
        """
        Called by the execution layer after a fill is confirmed.
        Activates exit-rule monitoring for this position.
        """
        pos = self._positions.get(symbol)
        if pos and not pos.get("confirmed", False):
            pos["confirmed"] = True
            logger.info(f"[{self.name}] Entry confirmed for {symbol} — exit monitoring active")

    def on_signal_rejected(self, symbol: str) -> None:
        """
        Called when an entry signal was rejected (LLM or rule filter).
        Clears the pending position so we don't exit a trade that was never opened.
        """
        pos = self._positions.get(symbol)
        if pos and not pos.get("confirmed", False):
            del self._positions[symbol]
            logger.debug(f"[{self.name}] Rolled back unconfirmed entry for {symbol}")

    @staticmethod
    def _atm_premium(spot: float, iv: float, dte: float) -> float:
        """Bachelier ATM approximation: 0.4 * S * IV * sqrt(T). Used for quick estimates."""
        t = max(dte, 1) / 365.0
        return 0.40 * spot * iv * (t ** 0.5)

    @staticmethod
    def _bsm_call(S: float, K: float, sigma: float, T: float) -> float:
        """Black-Scholes call price (r=0 for simplicity)."""
        if T <= 1e-6:
            return max(S - K, 0.0)
        sq = sigma * T ** 0.5
        d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / sq
        d2 = d1 - sq
        return float(S * _norm.cdf(d1) - K * _norm.cdf(d2))

    @staticmethod
    def _bsm_put(S: float, K: float, sigma: float, T: float) -> float:
        """Black-Scholes put price via put-call parity (r=0)."""
        if T <= 1e-6:
            return max(K - S, 0.0)
        sq = sigma * T ** 0.5
        d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / sq
        d2 = d1 - sq
        return float(K * _norm.cdf(-d2) - S * _norm.cdf(-d1))

    @classmethod
    def _condor_net_value(cls, S: float, entry_spot: float, iv: float, T: float) -> float:
        """
        Net market value of the 2%/7% OTM iron condor (cost to close).

        Structure (strikes fixed at entry):
          Short call  entry_spot × 1.02   Short put  entry_spot × 0.98
          Long  call  entry_spot × 1.07   Long  put  entry_spot × 0.93

        Positive return = current cost to close (rising = condor losing money).
        At entry this equals the net credit received.

        Using CURRENT implied_vol (sigma) so IV expansion is properly priced:
          - Vol spike 15% → 85%: condor value ≫ entry credit → stop loss fires
          - Deep spot move: short put/call goes ITM → condor value spikes → stop loss fires
          - Normal theta decay: condor value decays → take profit at 21 DTE
        """
        K_sc = entry_spot * 1.02
        K_lc = entry_spot * 1.07
        K_sp = entry_spot * 0.98
        K_lp = entry_spot * 0.93
        sigma = max(iv, 0.01)
        Tf = max(T, 1e-6)
        sc = cls._bsm_call(S, K_sc, sigma, Tf)
        lc = cls._bsm_call(S, K_lc, sigma, Tf)
        sp = cls._bsm_put(S, K_sp, sigma, Tf)
        lp = cls._bsm_put(S, K_lp, sigma, Tf)
        return sc + sp - lc - lp   # net cost to close (buy shorts, sell longs)

    @classmethod
    def _straddle_value(cls, S: float, K: float, iv: float, T: float) -> float:
        """
        BSM ATM straddle value (call + put at strike K).
        Used for marking long straddle positions using CURRENT vol.
        """
        sigma = max(iv, 0.01)
        Tf = max(T, 1e-6)
        return cls._bsm_call(S, K, sigma, Tf) + cls._bsm_put(S, K, sigma, Tf)

    @staticmethod
    def _dte(expiry_str: str, ref: datetime) -> float:
        """Calendar days from ref to expiry."""
        try:
            exp = datetime.strptime(expiry_str, "%Y-%m-%d")
            return max(0.0, (exp - ref).days)
        except (ValueError, TypeError):
            return 30.0  # fallback if expiry is missing/malformed

    async def _evaluate(
        self, sym: str, iv_pct: float, realized_vol: float,
        implied_vol: float, iv_rv: float, candle
    ) -> Optional[Signal]:
        # Use candle timestamp so backtest timing is correct (not wall-clock time)
        now: datetime = candle.timestamp
        pos = self._positions.get(sym, {})

        # ------------------------------------------------------------------ #
        # EXIT LOGIC — checked every candle, no cooldown gate                 #
        # Only runs after confirm_entry() has been called (real fill received) #
        # ------------------------------------------------------------------ #
        if pos and pos.get("confirmed", False):
            entry_time: datetime = pos["entry_time"]
            pos_type: str = pos["type"]
            entry_premium: float = pos.get("entry_premium", 0.0)
            expiry_str: str = pos.get("expiry", "")
            entry_spot: float = pos.get("entry_spot", candle.close)
            entry_iv: float = pos.get("entry_iv", implied_vol)
            signal_type: str = pos.get("signal_type", "iron_condor" if pos_type == "short" else "straddle")

            days_held = (now - entry_time).days
            dte = self._dte(expiry_str, now)
            T_now = max(dte, 0) / 365.0

            # Mark-to-market using CURRENT implied_vol (not fixed entry_iv).
            #
            # Why current vol matters:
            #   Iron condors: a VIX spike from 15% → 85% makes short legs worth
            #     far more than the original credit → stop loss fires correctly.
            #   Straddles: a vol collapse kills long premium → stop fires.
            #
            # implied_vol here is candle.implied_vol if the regime/connector set it
            # (historically accurate for backtests) or AR(1) continuation otherwise.
            # AR(1) noise is ±0.5% per candle — low enough not to cause false exits.
            if signal_type == "iron_condor":
                current_premium = self._condor_net_value(
                    S=candle.close, entry_spot=entry_spot, iv=implied_vol, T=T_now
                )
            else:   # straddle / strangle — ATM at entry strike
                current_premium = self._straddle_value(
                    S=candle.close, K=entry_spot, iv=implied_vol, T=T_now
                )

            exit_reason: Optional[str] = None

            # Rule 1 — 21 DTE management: close before pin/gamma risk explodes
            if dte <= self.dte_close:
                exit_reason = f"dte_{int(dte)}_management"

            # Rule 2 — 50% of max profit (short: premium decayed; long: premium grew)
            elif entry_premium > 0 and pos_type == "short":
                profit_pct = (entry_premium - current_premium) / entry_premium
                if profit_pct >= self.take_profit_pct:
                    exit_reason = "take_profit_50pct"
                # Rule 3 — stop loss at 200% of premium collected
                elif current_premium >= self.stop_loss_mult * entry_premium:
                    exit_reason = "stop_loss_200pct"

            elif entry_premium > 0 and pos_type == "long":
                profit_pct = (current_premium - entry_premium) / entry_premium
                if profit_pct >= self.take_profit_pct:
                    exit_reason = "take_profit_50pct"
                # Stop long at 200% of premium paid (lost 2x what we put in)
                elif current_premium <= entry_premium / self.stop_loss_mult:
                    exit_reason = "stop_loss_200pct"

            # Rule 4 — IV mean reversion complete
            elif (
                (pos_type == "long" and iv_pct > 65)
                or (pos_type == "short" and iv_pct < 35)
            ):
                exit_reason = "mean_reversion_complete"

            # Rule 5 — max hold days exceeded
            elif days_held >= self.max_hold_days:
                exit_reason = "max_hold_exceeded"

            if exit_reason:
                # Estimated P&L (straddle-equivalent, pre-commission)
                if entry_premium > 0:
                    if pos_type == "short":
                        est_pnl = (entry_premium - current_premium) * 100
                    else:
                        est_pnl = (current_premium - entry_premium) * 100
                    pnl_str = f"+${est_pnl:.2f}" if est_pnl >= 0 else f"-${abs(est_pnl):.2f}"
                else:
                    pnl_str = "n/a"

                logger.info(
                    f"[{self.name}] EXIT {sym} {pos_type.upper()}: "
                    f"reason={exit_reason}  "
                    f"entry=${entry_premium:.3f}  current=${current_premium:.3f}  "
                    f"dte={dte:.0f}  held={days_held}d  est_pnl={pnl_str}"
                )

                self._positions[sym] = {}
                self._last_signal[sym] = now
                return self._emit(Signal(
                    signal_type=SignalType.CLOSE_POSITION,
                    symbol=sym,
                    timestamp=now,
                    strike=candle.close,
                    expiry=expiry_str or self._next_expiry(ref_date=now.date()),
                    confidence=0.90,
                    position_size=1,
                    strategy_name=self.name,
                    metadata={
                        "iv_percentile": iv_pct,
                        "days_held": days_held,
                        "dte": dte,
                        "entry_premium": entry_premium,
                        "current_premium": current_premium,
                        "entry_iv": entry_iv,
                        "current_iv": implied_vol,   # live vol at exit — used for per-leg BSM close pricing
                        "entry_spot": entry_spot,
                        "signal_type": signal_type,
                        "reason": exit_reason,
                    },
                ))

        # ------------------------------------------------------------------ #
        # ENTRY LOGIC — gated by signal cooldown                              #
        # ------------------------------------------------------------------ #
        if not self._can_signal(sym, now):
            return None

        # --- BUY vol: IV is cheap ---
        if iv_pct < self.buy_threshold and iv_rv < self.iv_rv_buy_max and not pos and not self._positions.get(sym):
            conf = (self.buy_threshold - iv_pct) / self.buy_threshold
            if conf < self.min_confidence:
                return None
            expiry = self._next_expiry(weeks_out=4, ref_date=now.date())
            T_entry = (4 * 7) / 365.0
            entry_premium = self._straddle_value(candle.close, candle.close, implied_vol, T_entry)
            self._last_signal[sym] = now
            self._positions[sym] = {
                "type": "long",
                "signal_type": "straddle",
                "entry_time": now,
                "entry_iv_pct": iv_pct,
                "entry_premium": entry_premium,
                "entry_iv": implied_vol,
                "entry_spot": candle.close,
                "expiry": expiry,
                "confirmed": False,
            }
            return self._emit(Signal(
                signal_type=SignalType.STRADDLE,
                symbol=sym,
                timestamp=now,
                strike=candle.close,
                expiry=expiry,
                confidence=min(conf, 1.0),
                position_size=1,
                strategy_name=self.name,
                metadata={
                    "iv_percentile": iv_pct,
                    "iv_rv_ratio": iv_rv,
                    "realized_vol": realized_vol,
                    "implied_vol": implied_vol,
                    "entry_premium": entry_premium,
                    "reason": "buy_cheap_vol",
                },
            ))

        # --- SELL vol: IV is rich ---
        if iv_pct > self.sell_threshold and iv_rv > self.iv_rv_sell_min and not pos and not self._positions.get(sym):
            conf = (iv_pct - self.sell_threshold) / (100 - self.sell_threshold)
            if conf < self.min_confidence:
                return None
            expiry = self._next_expiry(weeks_out=6, ref_date=now.date())
            T_entry = (6 * 7) / 365.0
            entry_premium = self._condor_net_value(
                S=candle.close, entry_spot=candle.close, iv=implied_vol, T=T_entry
            )
            self._last_signal[sym] = now
            self._positions[sym] = {
                "type": "short",
                "signal_type": "iron_condor",
                "entry_time": now,
                "entry_iv_pct": iv_pct,
                "entry_premium": entry_premium,
                "entry_iv": implied_vol,
                "entry_spot": candle.close,
                "expiry": expiry,
                "confirmed": False,
            }
            return self._emit(Signal(
                signal_type=SignalType.IRON_CONDOR,
                symbol=sym,
                timestamp=now,
                strike=candle.close,
                expiry=expiry,
                confidence=min(conf, 1.0),
                position_size=1,
                strategy_name=self.name,
                metadata={
                    "iv_percentile": iv_pct,
                    "iv_rv_ratio": iv_rv,
                    "realized_vol": realized_vol,
                    "implied_vol": implied_vol,
                    "entry_premium": entry_premium,
                    "reason": "sell_rich_vol",
                },
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
    Trades dealer gamma imbalances (gamma walls / GEX).

    Concept
    -------
    When market-makers are net SHORT gamma near spot (large negative GEX),
    they must sell into rallies and buy dips to delta-hedge, amplifying moves.
    A straddle benefits from this vol expansion.

    When dealers are net LONG gamma (positive GEX), they dampen moves and
    the market tends to pin — no trade is taken.

    Data pipeline
    -------------
    update_gamma_skew(symbol, gamma_skew) is called by QuantBot after each
    options chain refresh.  gamma_skew is a dict {strike: net_dealer_gamma}
    produced by TastytradeConnector.get_dealer_gamma_skew().

    Entry condition (fixed)
    -----------------------
    BUG in original code:
      `net_gamma < -imbalance_threshold * abs(net_gamma)`
    expands to `net_gamma < imbalance_threshold * net_gamma` when net_gamma < 0,
    which simplifies to `1 < imbalance_threshold` — always False for threshold=1.5.
    The condition was mathematically impossible to trigger.

    Fixed condition:
      net_gamma < -imbalance_threshold
    Meaning: the summed dealer gamma within 3% of spot is more negative than
    the threshold (threshold is in BSM gamma units per share, typically ~1e-3
    for SPY near ATM).

    Position tracking
    -----------------
    One open position per symbol.  After an entry signal the strategy waits
    for confirm_entry() before considering a new signal.  Exits on:
      1. DTE ≤ 7 (roll risk)
      2. IV regime flip (net_gamma crosses to positive — dealers now long gamma)
      3. Max hold exceeded
    """

    def __init__(
        self,
        imbalance_threshold: float = 1e-4,   # BSM gamma units (≈ ATM gamma on $100 stock)
        near_strike_pct: float = 0.03,        # strikes within 3% of spot
        max_hold_days: int = 14,
        dte_close: int = 7,
    ):
        super().__init__("GammaImbalance")
        self.imbalance_threshold = imbalance_threshold
        self.near_strike_pct = near_strike_pct
        self.max_hold_days = max_hold_days
        self.dte_close = dte_close

        self._gamma_skew: Dict[str, Dict[float, float]] = {}   # symbol → {strike: dealer_gamma}
        self._positions: Dict[str, dict] = {}                  # symbol → position state
        self._last_signal: Dict[str, datetime] = {}

    def update_gamma_skew(self, symbol: str, gamma_skew: Dict[float, float]) -> None:
        """
        Inject dealer gamma positioning data produced by
        TastytradeConnector.get_dealer_gamma_skew().
        Called by QuantBot after each options chain refresh (every ~5 min).
        """
        self._gamma_skew[symbol] = gamma_skew
        total = sum(gamma_skew.values())
        logger.debug(
            f"[GammaImbalance] Gamma skew updated for {symbol}: "
            f"{len(gamma_skew)} strikes  net_total={total:.6f}"
        )

    def confirm_entry(self, symbol: str) -> None:
        """Called by the execution layer after a fill is confirmed."""
        pos = self._positions.get(symbol)
        if pos and not pos.get("confirmed", False):
            pos["confirmed"] = True
            logger.info(f"[GammaImbalance] Entry confirmed for {symbol} — exit monitoring active")

    def on_signal_rejected(self, symbol: str) -> None:
        """Clear unconfirmed position if the signal was rejected."""
        pos = self._positions.get(symbol)
        if pos and not pos.get("confirmed", False):
            del self._positions[symbol]

    @staticmethod
    def _next_expiry_dte(target_dte: int = 14, ref_date=None) -> str:
        """Return the next weekly expiry approximately `target_dte` calendar days out."""
        from datetime import date, timedelta
        today = ref_date or date.today()
        target = today + timedelta(days=target_dte)
        days_to_fri = (4 - target.weekday()) % 7
        expiry = target + timedelta(days=days_to_fri)
        return expiry.strftime("%Y-%m-%d")

    @staticmethod
    def _dte(expiry_str: str, ref: datetime) -> float:
        try:
            exp = datetime.strptime(expiry_str, "%Y-%m-%d")
            return max(0.0, (exp - ref).days)
        except (ValueError, TypeError):
            return 14.0

    async def on_tick(self, tick) -> Optional[Signal]:
        sym = tick.symbol
        skew = self._gamma_skew.get(sym)
        if not skew:
            return None

        now = datetime.utcnow()
        pos = self._positions.get(sym, {})

        # ------------------------------------------------------------------ #
        # EXIT — only after confirm_entry()                                    #
        # ------------------------------------------------------------------ #
        if pos and pos.get("confirmed", False):
            entry_time: datetime = pos["entry_time"]
            expiry_str: str = pos.get("expiry", "")
            days_held = (now - entry_time).days
            dte = self._dte(expiry_str, now)

            exit_reason: Optional[str] = None

            # Rule 1 — DTE management
            if dte <= self.dte_close:
                exit_reason = f"dte_{int(dte)}_management"

            # Rule 2 — gamma regime flipped (dealers now long gamma → market reverts)
            elif self._net_gamma_near(skew, tick.price) >= 0:
                exit_reason = "gamma_regime_flip_long"

            # Rule 3 — max hold
            elif days_held >= self.max_hold_days:
                exit_reason = "max_hold_exceeded"

            if exit_reason:
                self._positions[sym] = {}
                self._last_signal[sym] = now
                return self._emit(Signal(
                    signal_type=SignalType.CLOSE_POSITION,
                    symbol=sym,
                    timestamp=now,
                    strike=tick.price,
                    expiry=expiry_str or self._next_expiry_dte(14),
                    confidence=0.85,
                    position_size=1,
                    strategy_name=self.name,
                    metadata={
                        "days_held": days_held,
                        "dte": dte,
                        "reason": exit_reason,
                    },
                ))

        # ------------------------------------------------------------------ #
        # ENTRY — one position per symbol, 4-hour cooldown                    #
        # ------------------------------------------------------------------ #
        if pos:
            return None  # already in a position

        last = self._last_signal.get(sym)
        if last and (now - last).total_seconds() < 4 * 3600:
            return None

        net_gamma = self._net_gamma_near(skew, tick.price)
        if abs(net_gamma) < 1e-10:
            return None

        # FIXED condition: dealers net short gamma → expect trending/volatile move
        # Original bug: `net_gamma < -threshold * abs(net_gamma)` is always False.
        # Fixed: `net_gamma < -threshold` (direct magnitude check)
        if net_gamma < -self.imbalance_threshold:
            expiry = self._next_expiry_dte(14, ref_date=now.date())
            confidence = min(0.50 + abs(net_gamma) / (self.imbalance_threshold + 1e-10) * 0.15, 0.85)
            self._last_signal[sym] = now
            self._positions[sym] = {
                "entry_time": now,
                "expiry": expiry,
                "net_gamma_entry": net_gamma,
                "confirmed": False,
            }
            return self._emit(Signal(
                signal_type=SignalType.STRADDLE,
                symbol=sym,
                timestamp=now,
                strike=tick.price,
                expiry=expiry,
                confidence=round(confidence, 2),
                position_size=1,
                strategy_name=self.name,
                metadata={
                    "net_dealer_gamma": net_gamma,
                    "imbalance_threshold": self.imbalance_threshold,
                    "regime": "short_gamma_wall",
                },
            ))
        return None

    def _net_gamma_near(self, skew: Dict[float, float], spot: float) -> float:
        """Sum dealer gamma for strikes within near_strike_pct of spot."""
        near = {
            k: v for k, v in skew.items()
            if abs(k - spot) / max(spot, 1.0) < self.near_strike_pct
        }
        return sum(near.values()) if near else 0.0

    async def on_candle(self, candle) -> Optional[Signal]:
        return None

    async def on_greek_update(self, greeks) -> Optional[Signal]:
        return None


# ============================================================================
# TAIL HEDGE STRATEGY
# ============================================================================


class TailHedgeStrategy(Strategy):
    """
    Dynamic drawdown-triggered tail hedge.

    Monitors portfolio equity vs its high-water mark.  When drawdown exceeds
    `trigger_pct`, buys OTM puts sized to cover `target_cover_pct` of the open
    loss via delta-adjusted notional.  Spend is capped at `max_hedge_cost_pct`
    of starting capital per activation.

    The hedge is closed when drawdown recovers below `close_pct`.

    IMPORTANT: update_portfolio_state() must be called each candle by the
    simulation loop / bot before on_candle() is processed.

    How it interacts with iron condors:
      - Condor is short vol — loses when spot moves or IV spikes.
      - OTM put gains intrinsic value when spot drops and vega when IV spikes.
      - The combination caps the left-tail loss of the condor book.
      - Net cost: put premium (~1-2% of capital at 2% DD trigger).
    """

    def __init__(
        self,
        trigger_pct: float = 0.02,        # buy puts when DD > 2%
        close_pct: float = 0.005,          # close hedge when DD < 0.5%
        otm_pct: float = 0.05,             # buy 5% OTM puts
        dte_weeks: int = 4,                # ~30 DTE
        target_cover_pct: float = 0.50,    # hedge 50% of open loss via delta
        max_hedge_cost_pct: float = 0.015, # cap spend at 1.5% of capital per activation
    ):
        super().__init__("TailHedge")
        self.trigger_pct = trigger_pct
        self.close_pct = close_pct
        self.otm_pct = otm_pct
        self.dte_weeks = dte_weeks
        self.target_cover_pct = target_cover_pct
        self.max_hedge_cost_pct = max_hedge_cost_pct

        # State injected by simulation loop before each candle
        self._portfolio_value: float = 0.0
        self._peak_value: float = 0.0
        self._capital: float = 100_000.0

        # Per-symbol hedge tracking
        self._hedged: Dict[str, bool] = {}
        self._hedge_entry: Dict[str, Dict] = {}  # symbol → {expiry, strike, contracts}

    def update_portfolio_state(
        self,
        current_value: float,
        peak_value: float,
        capital: float,
    ) -> None:
        """Must be called each candle by the simulation loop before on_candle()."""
        self._portfolio_value = current_value
        self._peak_value = peak_value
        self._capital = capital

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    async def on_greek_update(self, greeks) -> Optional[Signal]:
        return None

    @staticmethod
    def _bsm_put(S: float, K: float, sigma: float, T: float) -> float:
        if T <= 1e-6:
            return max(K - S, 0.0)
        sq = sigma * T ** 0.5
        d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / sq
        return float(K * _norm.cdf(sq - d1) - S * _norm.cdf(-d1))

    @staticmethod
    def _put_delta(S: float, K: float, sigma: float, T: float) -> float:
        """BSM put delta (negative value)."""
        if T <= 1e-6:
            return -1.0 if K > S else 0.0
        sq = sigma * T ** 0.5
        d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / sq
        return float(_norm.cdf(d1) - 1.0)

    def _next_expiry(self, weeks_out: int, ref_date=None) -> str:
        from datetime import date as _date
        today = ref_date or _date.today()
        days_to_friday = (4 - today.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        expiry = today + timedelta(days=days_to_friday + weeks_out * 7)
        return expiry.strftime("%Y-%m-%d")

    async def on_candle(self, candle) -> Optional[Signal]:
        sym = candle.symbol
        S = candle.close
        now = candle.timestamp
        iv = max(getattr(candle, "implied_vol", None) or 0.20, 0.05)

        peak = self._peak_value if self._peak_value > 0 else self._capital
        current = self._portfolio_value if self._portfolio_value > 0 else peak
        drawdown = max((peak - current) / peak, 0.0)
        loss_dollars = max(peak - current, 0.0)

        has_hedge = self._hedged.get(sym, False)

        # ── Close hedge when portfolio recovers ──────────────────────────────
        if has_hedge and drawdown <= self.close_pct:
            self._hedged[sym] = False
            entry_info = self._hedge_entry.pop(sym, {})
            expiry = entry_info.get("expiry", self._next_expiry(self.dte_weeks, now.date()))

            logger.info(
                f"[{self.name}] Closing hedge on {sym}: DD recovered to {drawdown:.2%}"
            )
            return self._emit(Signal(
                signal_type=SignalType.CLOSE_POSITION,
                symbol=sym,
                timestamp=now,
                strike=S,
                expiry=expiry,
                confidence=0.95,
                position_size=1,
                strategy_name=self.name,
                metadata={
                    "reason": "hedge_dd_recovered",
                    "drawdown": drawdown,
                    "current_iv": iv,
                    "dte": 0,
                    "signal_type": "tail_hedge_put",
                    "close_strategy_name": self.name,   # only close OUR positions
                },
            ))

        # ── Activate hedge when drawdown breaches trigger ────────────────────
        if not has_hedge and drawdown >= self.trigger_pct:
            K = S * (1.0 - self.otm_pct)
            T = (self.dte_weeks * 7) / 365.0
            put_price = max(self._bsm_put(S, K, iv, T), 0.05)

            # Size: cover target_cover_pct of loss via delta-adjusted notional
            put_delta_abs = abs(self._put_delta(S, K, iv, T))
            if put_delta_abs > 1e-4:
                # Each contract gains put_delta_abs * 100 per $1 move in spot
                # We want to cover `loss * target_cover_pct` of dollar-delta exposure
                contracts = max(1, int(
                    (loss_dollars * self.target_cover_pct)
                    / (put_delta_abs * 100 * S)
                ))
            else:
                contracts = 1

            # Cap total premium spend
            cost = put_price * contracts * 100
            max_cost = self._capital * self.max_hedge_cost_pct
            if cost > max_cost:
                contracts = max(1, int(max_cost / (put_price * 100)))
                cost = put_price * contracts * 100

            expiry = self._next_expiry(self.dte_weeks, now.date())
            self._hedged[sym] = True
            self._hedge_entry[sym] = {
                "expiry": expiry,
                "strike": K,
                "contracts": contracts,
                "entry_price": put_price,
            }

            logger.info(
                f"[{self.name}] HEDGE ACTIVATED {sym}: DD={drawdown:.2%} "
                f"loss=${loss_dollars:.0f} → {contracts}x put K={K:.1f} "
                f"iv={iv:.1%} cost=${cost:.0f}"
            )
            return self._emit(Signal(
                signal_type=SignalType.BUY_PUT,
                symbol=sym,
                timestamp=now,
                strike=K,
                expiry=expiry,
                confidence=0.92,
                position_size=contracts,
                strategy_name=self.name,
                metadata={
                    "implied_vol": iv,
                    "put_price": put_price,
                    "reason": "tail_hedge_dd_trigger",
                    "drawdown": drawdown,
                    "loss_dollars": loss_dollars,
                },
            ))

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
