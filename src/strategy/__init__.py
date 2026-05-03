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
# MARKET REGIME — live detection from price + vol data
# ============================================================================


class MarketRegime(Enum):
    CALM_BULL    = "calm_bull"      # VIX 12-18, trending up  → loosen sell filters
    ELEVATED_VOL = "elevated_vol"   # VIX 18-28, choppy       → tighter filters
    TRENDING_BEAR = "trending_bear" # VIX 28-45, downtrend    → almost no condors
    CRASH        = "crash"          # VIX 45+, extreme moves  → long vol only
    RECOVERY     = "recovery"       # VIX falling from high   → moderate sells


# Per-regime entry/exit parameters — the core of adaptive filtering
_REGIME_PARAMS: Dict[str, dict] = {
    # ── CALM BULL ─────────────────────────────────────────────────────────────
    # Low realized vol, IV/RV structurally > 1.3 — ideal condor environment.
    # Primary edge: sell condors (IV premium capture). Allow straddle buys only
    # when IV/RV drops below 1.0 (genuinely cheap vol), gated by heuristic.
    "calm_bull": {
        "sell_threshold":  65.0,   # sell condors when IV%ile > 65
        "buy_threshold":   28.0,   # regime pre-filter; heuristic tightens to iv_rv < 1.10
        "iv_rv_sell_min":  1.30,   # regime pre-filter; heuristic tightens to iv_rv > 1.45
        "iv_rv_buy_max":   1.15,
        "take_profit_pct": 0.50,
        "stop_loss_mult":  2.00,
        "min_interval_h":  18,
        "capital_pct":     0.040,  # 4% — this is the money regime
        "min_contracts":   2,
    },
    # ── ELEVATED VOL ──────────────────────────────────────────────────────────
    # VIX 27-30. IV still rich vs RV but underlying is choppier.
    # Sell at tighter threshold; allow vol buys when IV/RV < 1.0.
    "elevated_vol": {
        "sell_threshold":  73.0,
        "buy_threshold":   22.0,
        "iv_rv_sell_min":  1.45,
        "iv_rv_buy_max":   1.00,   # only buy if IV actually flat/below realized vol
        "take_profit_pct": 0.40,
        "stop_loss_mult":  1.75,
        "min_interval_h":  28,
        "capital_pct":     0.025,
        "min_contracts":   1,
    },
    # ── TRENDING BEAR ─────────────────────────────────────────────────────────
    # VIX 30-45, sustained downtrend. Condors die in directional markets.
    # Sell only at extreme premium (88th %ile + IV/RV > 1.8 = vol way overpriced).
    # Buy straddles when IV is cheap relative to realized movement.
    "trending_bear": {
        "sell_threshold":  88.0,
        "buy_threshold":   22.0,
        "iv_rv_sell_min":  1.80,
        "iv_rv_buy_max":   1.05,   # buy when IV hasn't spiked yet
        "take_profit_pct": 0.45,
        "stop_loss_mult":  1.50,
        "min_interval_h":  36,
        "capital_pct":     0.015,  # 1.5% — stay small in brutal environment
        "min_contracts":   1,
    },
    # ── CRASH ─────────────────────────────────────────────────────────────────
    # VIX 45+. Never sell condors. Buy straddles as IV still cheap vs coming RV.
    "crash": {
        "sell_threshold":  999.0,  # disabled
        "buy_threshold":   25.0,
        "iv_rv_sell_min":  999.0,  # disabled
        "iv_rv_buy_max":   1.30,   # buy when IV hasn't fully spiked yet
        "take_profit_pct": 1.50,   # let crash vol runs pay 2.5x
        "stop_loss_mult":  3.00,
        "min_interval_h":  10,
        "capital_pct":     0.012,
        "min_contracts":   1,
    },
    # ── RECOVERY ──────────────────────────────────────────────────────────────
    # Vol falling from elevated levels. Sweet spot for condor selling.
    # Also allow straddle buys on extreme cheapness — recovery can stall and re-spike.
    "recovery": {
        "sell_threshold":  68.0,
        "buy_threshold":   25.0,
        "iv_rv_sell_min":  1.35,
        "iv_rv_buy_max":   1.05,
        "take_profit_pct": 0.50,
        "stop_loss_mult":  1.75,
        "min_interval_h":  20,
        "capital_pct":     0.035,  # 3.5% — recovery is the sweet spot for selling
        "min_contracts":   2,
    },
}


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
    SELL_STRADDLE = "sell_straddle"
    SELL_STRANGLE = "sell_strangle"
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
        min_confidence: float = 0.45,
        max_hold_days: int = 45,
        dte_close: int = 21,
        # Sizing — base sizing uses capital_pct floor; Kelly scales it up
        min_contracts: int = 2,
        max_contracts: int = 15,
        capital_pct_per_trade: float = 0.03,  # 3% of capital at risk per trade
    ):
        super().__init__("VolMeanReversion")
        self.lookback_days = lookback_days
        self.min_confidence = min_confidence
        self.max_hold_days = max_hold_days
        self.dte_close = dte_close
        self.min_contracts = min_contracts
        self.max_contracts = max_contracts
        self.capital_pct_per_trade = capital_pct_per_trade

        # These remain as fallbacks when regime detection has insufficient history
        self._default_params = _REGIME_PARAMS["elevated_vol"]

        self._price_history: Dict[str, List[float]] = {}
        self._iv_history: Dict[str, List[float]] = {}
        # symbol → {type, entry_time, entry_premium, expiry, entry_iv_pct, confirmed}
        # 'confirmed' is False until BacktestRunner/Bot calls confirm_entry() after fill.
        self._positions: Dict[str, Dict] = {}
        self._last_signal: Dict[str, datetime] = {} # symbol → last entry signal time

        # Kelly criterion: rolling trade log for dynamic position sizing
        self._trade_log: List[float] = []   # per-trade realized P&L (dollars, estimated)
        self._capital: float = 100_000.0    # updated by update_portfolio_state()

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

    def _detect_market_regime(
        self, sym: str, implied_vol: float, iv_rv: float
    ) -> MarketRegime:
        """
        Classify the current market environment from price history and vol metrics.

        Detection priority (highest to lowest):
          CRASH        — IV >= 0.45 (extreme fear, VIX equivalent)
          TRENDING_BEAR — IV >= 0.28 AND (price downtrend OR IV/RV < 1.05)
          ELEVATED_VOL — IV >= 0.22 OR IV/RV < 1.15 (vol structurally high)
          RECOVERY     — IV was high recently but now falling (mean reverting down)
          CALM_BULL    — default low-vol uptrend environment
        """
        prices  = self._price_history.get(sym, [])
        iv_hist = self._iv_history.get(sym, [])

        # 20-day price momentum
        momentum_20d = 0.0
        if len(prices) >= 20:
            momentum_20d = (prices[-1] - prices[-20]) / max(prices[-20], 1e-6)

        # Is IV falling from recent elevated levels? (recovery signature)
        iv_was_high = False
        iv_falling  = False
        if len(iv_hist) >= 15:
            recent_peak = max(iv_hist[-15:])
            iv_was_high = recent_peak >= 0.25
            iv_falling  = iv_hist[-1] < recent_peak * 0.85   # dropped >15% from peak

        if implied_vol >= 0.45:
            return MarketRegime.CRASH

        if implied_vol >= 0.30 and (momentum_20d < -0.05 or iv_rv < 1.05):
            return MarketRegime.TRENDING_BEAR

        if iv_was_high and iv_falling and implied_vol >= 0.20:
            return MarketRegime.RECOVERY

        # Raised from 0.22 → 0.27: VIX 22-27 is normal chop, not truly "elevated".
        # Also removed the iv_rv < 1.15 catch-all — it was misclassifying calm days
        # where IV happens to be close to RV.
        if implied_vol >= 0.27:
            return MarketRegime.ELEVATED_VOL

        return MarketRegime.CALM_BULL

    def _regime_params(self, sym: str, implied_vol: float, iv_rv: float) -> dict:
        """Return the entry/exit parameter dict for the current market regime."""
        regime = self._detect_market_regime(sym, implied_vol, iv_rv)
        params = _REGIME_PARAMS[regime.value]
        logger.debug(f"[{self.name}] {sym} regime={regime.value} iv={implied_vol:.2%} iv_rv={iv_rv:.2f}")
        return params

    def _can_signal(self, symbol: str, now: datetime, min_interval_h: float = 24.0) -> bool:
        last = self._last_signal.get(symbol)
        if last is None:
            return True
        return now - last >= timedelta(hours=min_interval_h)

    def update_portfolio_state(
        self, current_value: float, peak_value: float, capital: float
    ) -> None:
        """Receive current portfolio value from the simulation loop (used for Kelly sizing)."""
        self._capital = capital

    def _position_size(self, risk_per_contract: float, rp: dict) -> int:
        """
        Hybrid base-sizing + quarter-Kelly position sizing.

        Base sizing (always active):
          target_risk = capital * capital_pct   (regime-specific, e.g. 3% calm, 1% bear)
          base_contracts = max(min_contracts, floor(target_risk / risk_per_contract))

          At $100k calm_bull + $700 risk/contract: base = max(2, floor(3000/700)) = 4
          At $100k trending_bear + $525 risk/contract: base = max(1, floor(1000/525)) = 1

        Quarter-Kelly multiplier (once 10+ trades recorded):
          f* = (W * avg_win - L * avg_loss) / avg_win
          If f_quarter > 0 and kelly_contracts > base_contracts, use kelly_contracts.
          This grows sizing as proven edge accumulates but never below base.

        Capped at max_contracts (default 15).
        """
        capital = max(self._capital, 10_000.0)
        risk_per_contract = max(risk_per_contract, 1.0)

        # Regime-specific sizing: calm bull is generous, bear/crash is tiny
        capital_pct = rp.get("capital_pct", self.capital_pct_per_trade)
        min_c = rp.get("min_contracts", self.min_contracts)

        base = max(
            min_c,
            int(capital * capital_pct / risk_per_contract),
        )

        # Kelly multiplier once history exists
        if len(self._trade_log) >= 10:
            wins   = [p for p in self._trade_log if p > 0]
            losses = [abs(p) for p in self._trade_log if p < 0]
            if wins and losses:
                win_rate = len(wins) / len(self._trade_log)
                avg_win  = float(np.mean(wins))
                avg_loss = float(np.mean(losses))
                if avg_win > 0:
                    f_star    = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    f_quarter = max(0.0, f_star * 0.25)
                    kelly = int(f_quarter * capital / risk_per_contract)
                    base  = max(base, kelly)

        return min(base, self.max_contracts)

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

            # Use regime-aware exit params based on CURRENT market conditions
            rp = self._regime_params(sym, implied_vol, iv_rv)
            take_profit_pct = rp["take_profit_pct"]
            stop_loss_mult  = rp["stop_loss_mult"]

            exit_reason: Optional[str] = None

            # Rule 1 — 21 DTE management: close before pin/gamma risk explodes
            if dte <= self.dte_close:
                exit_reason = f"dte_{int(dte)}_management"

            # Rule 2 — take profit (regime-aware threshold)
            elif entry_premium > 0 and pos_type == "short":
                profit_pct = (entry_premium - current_premium) / entry_premium
                if profit_pct >= take_profit_pct:
                    exit_reason = f"take_profit_{take_profit_pct:.0%}"
                # Rule 3 — stop loss (regime-aware multiplier)
                elif current_premium >= stop_loss_mult * entry_premium:
                    exit_reason = f"stop_loss_{stop_loss_mult:.0f}x"

            elif entry_premium > 0 and pos_type == "long":
                profit_pct = (current_premium - entry_premium) / entry_premium
                if profit_pct >= take_profit_pct:
                    exit_reason = f"take_profit_{take_profit_pct:.0%}"
                elif current_premium <= entry_premium / stop_loss_mult:
                    exit_reason = f"stop_loss_{stop_loss_mult:.0f}x"

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
                    # Feed into Kelly rolling log so future entries size correctly
                    self._trade_log.append(est_pnl)
                    if len(self._trade_log) > 200:       # keep rolling 200-trade window
                        self._trade_log = self._trade_log[-200:]
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
        # ENTRY LOGIC — regime-aware filters + hybrid sizing                  #
        # ------------------------------------------------------------------ #
        rp = self._regime_params(sym, implied_vol, iv_rv)
        if not self._can_signal(sym, now, min_interval_h=rp["min_interval_h"]):
            return None

        sell_threshold  = rp["sell_threshold"]
        buy_threshold   = rp["buy_threshold"]
        iv_rv_sell_min  = rp["iv_rv_sell_min"]
        iv_rv_buy_max   = rp["iv_rv_buy_max"]
        stop_loss_mult  = rp["stop_loss_mult"]
        regime          = self._detect_market_regime(sym, implied_vol, iv_rv)

        # --- BUY vol: IV is cheap relative to regime ---
        if buy_threshold > 0 and iv_pct < buy_threshold and iv_rv < iv_rv_buy_max and not pos and not self._positions.get(sym):
            conf = (buy_threshold - iv_pct) / max(buy_threshold, 1.0)
            if conf < self.min_confidence:
                return None
            expiry = self._next_expiry(weeks_out=4, ref_date=now.date())
            T_entry = (4 * 7) / 365.0
            entry_premium = self._straddle_value(candle.close, candle.close, implied_vol, T_entry)
            risk_per_contract = max(entry_premium * 100, 1.0)
            position_size = self._position_size(risk_per_contract, rp)
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
                position_size=position_size,
                strategy_name=self.name,
                metadata={
                    "iv_percentile": iv_pct,
                    "iv_rv_ratio": iv_rv,
                    "realized_vol": realized_vol,
                    "implied_vol": implied_vol,
                    "entry_premium": entry_premium,
                    "reason": "buy_cheap_vol",
                    "market_regime": regime.value,
                    "contracts": position_size,
                },
            ))

        # --- SELL vol: IV is rich relative to regime ---
        if sell_threshold < 999 and iv_pct > sell_threshold and iv_rv > iv_rv_sell_min and not pos and not self._positions.get(sym):
            conf = (iv_pct - sell_threshold) / max(100 - sell_threshold, 1.0)
            if conf < self.min_confidence:
                return None
            expiry = self._next_expiry(weeks_out=6, ref_date=now.date())
            T_entry = (6 * 7) / 365.0
            entry_premium = self._condor_net_value(
                S=candle.close, entry_spot=candle.close, iv=implied_vol, T=T_entry
            )
            risk_per_contract = max(stop_loss_mult * entry_premium * 100, 1.0)
            position_size = self._position_size(risk_per_contract, rp)
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
                position_size=position_size,
                strategy_name=self.name,
                metadata={
                    "iv_percentile": iv_pct,
                    "iv_rv_ratio": iv_rv,
                    "realized_vol": realized_vol,
                    "implied_vol": implied_vol,
                    "entry_premium": entry_premium,
                    "reason": "sell_rich_vol",
                    "market_regime": regime.value,
                    "contracts": position_size,
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
            market_regime=signal.metadata.get("market_regime"),
        )


# Back-compat alias
GammaScalping = GammaScalpingStrategy


# ============================================================================
# ZERO-DTE STRATEGY
# ============================================================================


class PlayType(Enum):
    PIN       = "pin"        # Sell strangle around positive-GEX wall
    EXPLOSIVE = "explosive"  # Buy straddle / direction on negative-GEX breakout
    IV_CRUSH  = "iv_crush"   # Sell ATM straddle to capture elevated 0DTE IV


class TimeRegime(Enum):
    AVOID  = "avoid"   # 9:30–9:50 — let opening settle
    OPEN   = "open"    # 9:50–10:30 — IV crush + explosive plays
    MIDDAY = "midday"  # 10:30–13:30 — only very high conviction
    CLOSE  = "close"   # 13:30–15:55 — all plays, peak gamma activity
    CLOSED = "closed"  # after 15:55 — no new trades


@dataclass
class ConditionScores:
    """
    Eight weighted conditions for 0DTE setup quality.
    Weighted average ≥ 0.65 → entry.
    """
    gamma_wall_quality: float = 0.0   # 2.0× weight — is there a real GEX wall?
    strike_confluence:  float = 0.0   # 1.5× — GEX + volume + OI all at same strike
    iv_term_structure:  float = 0.0   # 1.0× — IV edge for the play type
    price_velocity:     float = 0.0   # 1.0× — price decelerating (pin) / accelerating (explosive)
    volume_surge:       float = 0.0   # 1.0× — unusual options volume at the wall strike
    delta_flow:         float = 0.0   # 1.0× — net delta flow direction aligns with play
    time_regime:        float = 0.0   # 0.5× — time of day allows this play
    risk_headroom:      float = 0.0   # 0.5× — sufficient BP, not at position limit

    _W = {
        "gamma_wall_quality": 2.0,
        "strike_confluence":  1.5,
        "iv_term_structure":  1.0,
        "price_velocity":     1.0,
        "volume_surge":       1.0,
        "delta_flow":         1.0,
        "time_regime":        0.5,
        "risk_headroom":      0.5,
    }

    @property
    def weighted_score(self) -> float:
        total_w = sum(self._W.values())  # 8.5
        return sum(getattr(self, k) * v for k, v in self._W.items()) / total_w

    def to_dict(self) -> dict:
        return {k: round(getattr(self, k), 3) for k in self._W}


def _gamma_wall_strike(walls: List, spot: float, direction: int, fallback: float = 3.0) -> float:
    """
    Return the strike of the highest-gamma wall in the direction of the intended move.

    Gamma walls act as price magnets — buying options at the wall strike lets
    delta work in our favour as spot is pulled toward the wall.  The wall is
    naturally OTM (it's above/below spot), so the premium is cheaper than ATM.

    direction: +1 → above spot (calls), -1 → below spot (puts)
    fallback:  OTM offset if no directional wall exists in the list
    """
    if direction > 0:
        cands = [w for w in walls if w.strike > spot]
    else:
        cands = [w for w in walls if w.strike < spot]
    if cands:
        best = max(cands, key=lambda w: abs(w.net_gex_dollars) * w.confluence_score)
        return float(best.strike)
    return round(spot + direction * fallback)


class ZeroDTEStrategy(Strategy):
    """
    0-1-3 DTE SPY options strategy driven by gamma exposure walls.

    Entry: 8-condition weighted scorer, threshold 0.65.
    Three play types auto-selected by GEX sign + price behavior:
      PIN       — positive GEX + deceleration → sell strangle around wall
      EXPLOSIVE — negative GEX + acceleration → buy straddle / direction
      IV_CRUSH  — 0DTE IV >> 1DTE IV → sell straddle to capture crush

    DTE is chosen dynamically — whichever expiry has the best edge for the
    selected play type (from IVTermStructureAnalyzer).

    Sizing: Kelly criterion capped at 8% of account BP.
    Win rates are calibrated from PredictionLogger (real traded history).

    Self-learning: every setup is logged (traded or not). After ~30 sessions
    the accuracy calibration improves Kelly sizing automatically.
    """

    ENTRY_THRESHOLD = 0.65

    def __init__(
        self,
        account_bp: float = 2500.0,
        risk_pct: float = 0.18,
        r: float = 0.05,
        prediction_logger=None,  # PredictionLogger | None
        defined_risk: bool = False,  # True = iron condor (wings), False = naked strangle
    ):
        super().__init__("ZeroDTE")
        self.account_bp = account_bp
        self.risk_pct = risk_pct
        self.r = r
        self.prediction_logger = prediction_logger
        self.defined_risk = defined_risk

        self._price_history: List[float] = []
        self._active_pred_ids: Dict[str, int] = {}
        self._open_play: Optional[dict] = None
        self._vol_hist_per_strike: Dict[float, List[int]] = {}
        self._gap_pct: float = 0.0      # today's gap vs prior close (set each morning)
        self._prior_close: float = 0.0  # prior day close

        # Intraday order flow proxy — reset each day
        self._cum_delta:    float = 0.0   # cumulative volume-weighted bar direction
        self._cum_abs_vol:  float = 0.0   # total absolute volume for normalisation
        self._vix:          float = 18.0  # today's VIX level (set each morning)

    def set_daily_context(
        self, gap_pct: float, prior_close: float = 0.0, vix: float = 18.0
    ) -> None:
        """Call once before market open each day with gap info."""
        self._gap_pct     = gap_pct
        self._prior_close = prior_close
        self._vix         = vix
        self._cum_delta   = 0.0
        self._cum_abs_vol = 0.0

    # ------------------------------------------------------------------ #
    # Abstract interface                                                    #
    # ------------------------------------------------------------------ #

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    async def on_candle(self, candle) -> Optional[Signal]:
        return None

    async def on_greek_update(self, greeks) -> Optional[Signal]:
        return None

    # ------------------------------------------------------------------ #
    # Main intraday entry point                                            #
    # ------------------------------------------------------------------ #

    async def on_bar(
        self,
        candle,                         # Candle
        walls: List,                    # List[GammaWall]
        term_analyses: List,            # List[DTEAnalysis]
        available_bp: float,
    ) -> Optional[Signal]:
        """Called every 1-minute bar with fresh GEX + IV data."""
        spot = candle.close
        now  = candle.timestamp

        self._price_history.append(spot)
        if len(self._price_history) > 40:
            self._price_history = self._price_history[-40:]

        # Accumulate intraday order flow proxy each bar
        bar_move = candle.close - candle.open
        vol = float(candle.volume) if candle.volume else 1.0
        self._cum_delta   += bar_move * vol
        self._cum_abs_vol += abs(bar_move) * vol

        regime = self._time_regime(now)
        if regime in (TimeRegime.AVOID, TimeRegime.CLOSED):
            return None

        if not walls or not term_analyses:
            return None

        # Pick best wall: highest GEX × confluence within distance
        best_wall = max(walls, key=lambda w: abs(w.net_gex_dollars) * w.confluence_score)

        # Select play type and DTE
        play_type, best_dte = self._select_play(best_wall, spot, regime, term_analyses)
        if best_dte is None:
            return None

        # Score all 8 conditions
        scores = self._score(best_wall, spot, play_type, regime, best_dte, available_bp)
        ws = scores.weighted_score

        logger.debug(
            f"[ZeroDTE] {play_type.value}  K={best_wall.strike:.1f}  "
            f"score={ws:.3f}  {scores.to_dict()}"
        )

        # Log prediction regardless of entry decision
        pred_low  = spot - best_dte.expected_move_1sd
        pred_high = spot + best_dte.expected_move_1sd
        outcome_label = (
            "pin"      if play_type == PlayType.PIN else
            "crush"    if play_type == PlayType.IV_CRUSH else
            ("break_up" if self._velocity() > 0 else "break_down")
        )

        pred = SetupPrediction(
            timestamp=now,
            symbol=candle.symbol,
            wall_strike=best_wall.strike,
            spot_price=spot,
            dte=best_dte.dte,
            net_gex_dollars=best_wall.net_gex_dollars,
            confluence_score=best_wall.confluence_score,
            play_type=play_type.value,
            predicted_outcome=outcome_label,
            predicted_low=pred_low,
            predicted_high=pred_high,
            condition_score=ws,
            traded=(ws >= self.ENTRY_THRESHOLD),
        )

        pred_id = 0
        if self.prediction_logger:
            pred_id = self.prediction_logger.log(pred)
            self._active_pred_ids[candle.symbol] = pred_id

        if ws < self.ENTRY_THRESHOLD:
            return None

        # Kelly sizing — use empirical priors until prediction_logger accumulates history.
        # PIN prior = 0.80 (conservative vs observed 85-93%); EXPLOSIVE never trades.
        _play_priors = {
            PlayType.PIN:       0.80,
            PlayType.IV_CRUSH:  0.70,
            PlayType.EXPLOSIVE: 0.45,
        }
        win_rate = _play_priors.get(play_type, 0.60)
        if self.prediction_logger:
            calibrated = self.prediction_logger.get_calibrated_win_rate(play_type.value)
            if calibrated and calibrated > 0.50:
                win_rate = calibrated

        # Estimate ATM premium for sizing
        T = max(best_dte.dte, 0.25) / 365.0
        atm_premium = best_dte.atm_iv * spot * float(np.sqrt(T)) * 0.4

        n_contracts = self._kelly_contracts(
            premium=max(atm_premium, 0.50),
            play_type=play_type,
            win_rate=win_rate,
        )

        # Strike selection per play type
        target_reasoning = ""
        wing_call = wing_put = 0  # only used by EXPLOSIVE iron condor

        if play_type == PlayType.PIN:
            offset = max(2.0, best_dte.expected_move_1sd * 0.3)
            call_k       = round(best_wall.strike + offset)
            put_k        = round(best_wall.strike - offset)
            sig_type     = SignalType.IRON_CONDOR if self.defined_risk else SignalType.SELL_STRANGLE
            entry_strike = best_wall.strike

        elif play_type == PlayType.EXPLOSIVE:
            # PIN-only mode: skip high-vol days rather than buying premium.
            # ATM straddles require actual move > implied move (~35% probability)
            # which is negative EV. Sitting out beats losing consistently.
            return None

        else:  # IV_CRUSH
            sig_type     = SignalType.SELL_STRADDLE
            entry_strike = round(spot)
            call_k = put_k = round(spot)

        return self._emit(Signal(
            signal_type=sig_type,
            symbol=candle.symbol,
            timestamp=now,
            strike=entry_strike,
            expiry=best_dte.expiry,
            confidence=ws,
            position_size=n_contracts,
            strategy_name=self.name,
            metadata={
                "play_type":         play_type.value,
                "wall_strike":       best_wall.strike,
                "wall_type":         best_wall.wall_type,
                "net_gex_dollars":   best_wall.net_gex_dollars,
                "confluence_score":  best_wall.confluence_score,
                "condition_score":   ws,
                "conditions":        scores.to_dict(),
                "dte":               best_dte.dte,
                "expiry":            best_dte.expiry,
                "implied_vol":       best_dte.atm_iv,
                "expected_move":     best_dte.expected_move_1sd,
                "call_strike":       call_k,
                "put_strike":        put_k,
                "wing_call_strike":  wing_call if play_type == PlayType.EXPLOSIVE else call_k + 5,
                "wing_put_strike":   wing_put  if play_type == PlayType.EXPLOSIVE else put_k  - 5,
                "target_reasoning":  target_reasoning,
                "kelly_contracts":   n_contracts,
                "win_rate_used":     win_rate,
                "time_regime":       regime.value,
                "prediction_id":     pred_id,
            },
        ))

    def on_expiry_resolution(self, symbol: str, actual_price: float, pnl: Optional[float] = None) -> None:
        """Call at option expiry to resolve predictions and improve calibration."""
        pred_id = self._active_pred_ids.pop(symbol, None)
        if pred_id and self.prediction_logger:
            self.prediction_logger.resolve(pred_id, actual_price, pnl)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _time_regime(now: datetime) -> TimeRegime:
        """Map UTC datetime to market-time regime (ET = UTC-4 EDT / UTC-5 EST)."""
        # Approximate DST: EDT (UTC-4) Mar 2nd Sun → Nov 1st Sun; else EST (UTC-5).
        # Good enough for intraday trading without pytz dependency.
        import calendar
        y, mo, d = now.year, now.month, now.day
        # Find second Sunday of March and first Sunday of November
        def _nth_sunday(year: int, month: int, n: int) -> int:
            first_day = calendar.weekday(year, month, 1)  # 0=Mon
            first_sun = (6 - first_day) % 7 + 1          # day-of-month of 1st Sunday
            return first_sun + 7 * (n - 1)
        dst_start = _nth_sunday(y, 3, 2)   # 2nd Sun March
        dst_end   = _nth_sunday(y, 11, 1)  # 1st Sun November
        in_edt = (mo > 3 or (mo == 3 and d >= dst_start)) and \
                 (mo < 11 or (mo == 11 and d < dst_end))
        offset_h = 4 if in_edt else 5
        et_hour = (now.hour - offset_h) % 24
        m = et_hour * 60 + now.minute
        if m < 9 * 60 + 50:
            return TimeRegime.AVOID
        if m < 10 * 60 + 30:
            return TimeRegime.OPEN
        if m < 13 * 60 + 30:
            return TimeRegime.MIDDAY
        if m < 15 * 60 + 55:
            return TimeRegime.CLOSE
        return TimeRegime.CLOSED

    def _velocity(self) -> float:
        """Recent price velocity (fraction per bar). Positive = moving up."""
        p = self._price_history
        if len(p) < 4:
            return 0.0
        return (p[-1] - p[-4]) / max(p[-4], 1.0)

    def _deceleration(self) -> float:
        """Positive = price slowing down (good for pin)."""
        p = self._price_history
        if len(p) < 8:
            return 0.0
        v_recent = abs(p[-1] - p[-4]) / max(p[-4], 1.0)
        v_prior  = abs(p[-5] - p[-8]) / max(p[-8], 1.0)
        return (v_prior - v_recent) / max(v_prior, 1e-6)

    def _select_play(
        self,
        wall,              # GammaWall
        spot: float,
        regime: TimeRegime,
        term_analyses: List,
    ) -> tuple:
        """Return (PlayType, DTEAnalysis) for the best play given the wall."""
        from src.analysis import IVTermStructureAnalyzer
        analyzer = IVTermStructureAnalyzer()

        # Hard VIX gate: above 28 dealers are short gamma and SPY won't pin.
        # Sell-strangle risk/reward collapses — skip entirely regardless of GEX sign.
        if self._vix >= 28:
            return None, None

        dte0 = next((a for a in term_analyses if a.dte == 0), None)
        vel  = self._velocity()

        # 1. EXPLOSIVE — checked first so momentum plays aren't blocked by IV_CRUSH.
        #    Triggers on: negative GEX (dealers short gamma → explosive moves), OR
        #    strong intrabar momentum (0.15%+ per bar regardless of GEX sign).
        #    Requires VIX > 28: at moderate vol, OTM premiums can't overcome theta
        #    unless the market delivers a real move (>1%). Below VIX 28, "explosive"
        #    GEX triggers on normal chop and the edge collapses.
        is_neg_gex       = wall.net_gex_dollars < -200_000
        is_strong_move   = abs(vel) > 0.0015   # 0.15% per bar ≈ $0.80 SPY move
        if (is_neg_gex or is_strong_move) and self._vix >= 28:
            best = analyzer.select_best_dte(term_analyses, "explosive")
            if best:
                return PlayType.EXPLOSIVE, best

        # 2. IV_CRUSH — only when 0DTE crush_probability is genuinely elevated
        #    (not just "0DTE IV > 1DTE IV" which is almost always true on SPY).
        if (
            dte0 and dte0.crush_probability > 0.30
            and regime in (TimeRegime.OPEN, TimeRegime.CLOSE)
        ):
            return PlayType.IV_CRUSH, dte0

        # 3. PIN — positive GEX with price decelerating into wall (magnet effect)
        if wall.net_gex_dollars > 200_000 and self._deceleration() > 0.03:
            best = analyzer.select_best_dte(term_analyses, "pin")
            if best:
                return PlayType.PIN, best

        # 4. Default: follow GEX sign (require VIX >= 28 for EXPLOSIVE)
        if abs(wall.net_gex_dollars) > 200_000:
            if wall.net_gex_dollars < 0 and self._vix < 28:
                play = PlayType.PIN  # low vol → pin, not explosive
            else:
                play = PlayType.PIN if wall.net_gex_dollars > 0 else PlayType.EXPLOSIVE
            best = analyzer.select_best_dte(term_analyses, play.value)
            return play, best or (term_analyses[0] if term_analyses else None)

        return PlayType.PIN, (term_analyses[0] if term_analyses else None)

    def _score(
        self,
        wall,
        spot: float,
        play_type: PlayType,
        regime: TimeRegime,
        dte_analysis,
        available_bp: float,
    ) -> ConditionScores:
        s = ConditionScores()

        # 1. Gamma Wall Quality
        a = abs(wall.net_gex_dollars)
        s.gamma_wall_quality = (
            1.0 if a >= 5_000_000 else
            0.75 if a >= 2_000_000 else
            0.50 if a >= 1_000_000 else
            0.25 if a >= 500_000  else 0.05
        )

        # 2. Strike Confluence
        s.strike_confluence = wall.confluence_score

        # 3. IV Term Structure
        if dte_analysis:
            if play_type == PlayType.IV_CRUSH:
                s.iv_term_structure = float(np.clip(dte_analysis.crush_probability * 1.3, 0, 1))
            elif play_type == PlayType.PIN:
                s.iv_term_structure = dte_analysis.crush_probability
            else:  # explosive — high IV means larger potential move
                s.iv_term_structure = float(np.clip(dte_analysis.atm_iv / 0.25, 0, 1))

        # 4. Price Velocity
        vel   = self._velocity()
        decel = self._deceleration()
        if play_type == PlayType.PIN:
            s.price_velocity = float(np.clip(decel * 2, 0, 1))
        elif play_type == PlayType.EXPLOSIVE:
            s.price_velocity = float(np.clip(abs(vel) * 300, 0, 1))
        else:  # iv_crush — want stable price
            s.price_velocity = float(np.clip(1.0 - abs(vel) * 500, 0, 1))

        # 5. Volume Surge at Wall Strike
        hist = self._vol_hist_per_strike.get(wall.strike, [])
        avg_vol = float(np.mean(hist)) if hist else max(wall.total_volume * 0.5, 100)
        s.volume_surge = float(np.clip(wall.total_volume / max(avg_vol, 1) / 2.0, 0, 1))

        # Update history
        self._vol_hist_per_strike.setdefault(wall.strike, []).append(wall.total_volume)
        if len(self._vol_hist_per_strike[wall.strike]) > 30:
            self._vol_hist_per_strike[wall.strike] = self._vol_hist_per_strike[wall.strike][-30:]

        # 6. Delta Flow — call vs put volume balance
        total = wall.call_volume + wall.put_volume
        call_ratio = wall.call_volume / max(total, 1)  # 0.5 = balanced
        imbalance = abs(call_ratio - 0.5) * 2          # 0 = balanced, 1 = all one side
        if play_type == PlayType.PIN:
            s.delta_flow = float(np.clip(1.0 - imbalance, 0, 1))
        elif play_type == PlayType.EXPLOSIVE:
            direction_ok = (
                (call_ratio > 0.6 and vel > 0) or  # buying calls + moving up
                (call_ratio < 0.4 and vel < 0)     # buying puts + moving down
            )
            s.delta_flow = 0.80 if direction_ok else 0.40
        else:
            s.delta_flow = 0.60

        # 7. Time Regime
        regime_scores = {
            TimeRegime.AVOID:  0.0,
            TimeRegime.OPEN:   1.0,
            TimeRegime.MIDDAY: 0.70 if play_type == PlayType.EXPLOSIVE else 0.55,
            TimeRegime.CLOSE:  1.0,
            TimeRegime.CLOSED: 0.0,
        }
        s.time_regime = regime_scores.get(regime, 0.0)

        # 8. Risk Headroom
        trade_cost = self.account_bp * self.risk_pct
        s.risk_headroom = (
            1.0 if available_bp >= trade_cost * 3 else
            0.80 if available_bp >= trade_cost * 2 else
            0.50 if available_bp >= trade_cost     else 0.0
        )

        return s

    def update_bp(self, new_bp: float) -> None:
        """Update account buying power — called by backtest after each trade."""
        self.account_bp = max(new_bp, 100.0)

    def _dynamic_risk_pct(self) -> float:
        """Scale risk from 40% at $1k down to 15% at $10k, linear."""
        bp = self.account_bp
        if bp <= 1_000:   return 0.40
        if bp >= 10_000:  return 0.15
        t = (bp - 1_000) / 9_000.0
        return 0.40 - t * 0.25

    def _kelly_contracts(
        self,
        premium: float,
        play_type: PlayType,
        win_rate: float,
    ) -> int:
        """Kelly criterion → contracts, risk scales dynamically with account size."""
        payoff_map = {
            PlayType.PIN:      (0.50, 2.0),
            PlayType.EXPLOSIVE:(2.50, 1.0),
            PlayType.IV_CRUSH: (0.40, 2.0),
        }
        tp_frac, sl_mult = payoff_map[play_type]
        p, q = win_rate, 1.0 - win_rate
        b = tp_frac
        kelly_f = float(np.clip((p * b - q) / b, 0.0, 0.25))

        risk_pct     = self._dynamic_risk_pct()
        max_risk     = self.account_bp * risk_pct
        kelly_risk   = kelly_f * self.account_bp
        risk_dollars = min(max_risk, kelly_risk)

        risk_per_contract = sl_mult * premium * 100
        if risk_per_contract <= 0:
            return 1

        n = int(risk_dollars / risk_per_contract)
        # Iron condors: max loss = wing width (~$500); naked: margin ~$10k+
        if self.defined_risk:
            liquidity_cap = max(1, min(10, int(self.account_bp / 500)))
        else:
            liquidity_cap = max(2, min(10, int(self.account_bp / 25_000) * 2))
        return max(1, min(n, liquidity_cap))

    @staticmethod
    def _predict_move_target(
        walls: List,
        spot: float,
        direction: int,          # +1 = up, -1 = down
        expected_move: float,    # 1-sigma expected move from IV
    ) -> tuple:
        """
        Predict the OTM strike to buy on an explosive move.

        Logic:
          When negative GEX breaks, price accelerates toward the next
          positive-GEX wall — dealers flip long gamma there and pin price.
          That wall is the natural target. We buy OTM options at a strike
          roughly 60% of the way from spot to that wall (cheaper premium,
          higher payout if target is reached).

          Fallback: if no clear positive wall exists in that direction,
          use 0.8× the 1-sigma expected move as the OTM distance
          (slightly inside the market's expected range → better probability).

        Returns (strike: float, reasoning: str)
        """
        # Separate positive-GEX walls in the direction of the move
        if direction > 0:
            candidates = sorted(
                [w for w in walls if w.strike > spot + 0.5 and w.net_gex_dollars > 0],
                key=lambda w: w.strike
            )
        else:
            candidates = sorted(
                [w for w in walls if w.strike < spot - 0.5 and w.net_gex_dollars > 0],
                key=lambda w: w.strike, reverse=True
            )

        if candidates:
            target_wall = candidates[0]
            dist = abs(target_wall.strike - spot)
            # Buy at 60% of the way to the target wall — OTM but within reach
            otm_offset = max(dist * 0.60, 1.0) * direction
            otm_strike = round(spot + otm_offset)
            reasoning = (
                f"GEX target wall K={target_wall.strike:.0f} "
                f"(+${target_wall.net_gex_dollars/1e6:.1f}M) "
                f"{'above' if direction > 0 else 'below'} — "
                f"buying OTM at {otm_strike} ({dist*0.60:.1f}pts OTM)"
            )
            return float(otm_strike), reasoning

        # Fallback: use 0.8× expected move OTM
        otm_offset = max(expected_move * 0.80, 1.0) * direction
        otm_strike = round(spot + otm_offset)
        reasoning = (
            f"No clear GEX target — using 0.8x expected_move "
            f"({expected_move:.2f}pts) -> OTM at {otm_strike}"
        )
        return float(otm_strike), reasoning

    def seed_iv_history(self, symbol: str, history: List[float]) -> None:
        pass  # ZeroDTE does not use multi-day IV history seeding


# Helper import for SetupPrediction used in on_bar
try:
    from src.analysis import SetupPrediction  # noqa: F401
except ImportError:
    pass


# ============================================================================
# OPENING RANGE BREAKOUT (ORB)
# First 30-min high/low defines the range. Break above = buy call.
# Break below = buy put. Price has momentum committed to a direction.
# ============================================================================

class ORBStrategy(Strategy):
    """
    Opening Range Breakout: first 30-min high/low → trade the break.

    Edge: institutional orders placed at open define daily direction.
    A clean break of the ORB with volume confirmation = committed move.
    Buy options just inside the break so you're already slightly ITM on move.
    """

    CONFIRM_BARS = 2        # bars price must hold above/below ORB before entry
    MIN_RANGE_PCT = 0.0015  # ORB must be at least 0.15% wide to be meaningful
    SCORE_THRESHOLD = 0.68  # raised: ORB needs clean decisive break

    def __init__(self, account_bp: float = 2500.0, risk_pct: float = 0.15):
        super().__init__("ORB")
        self.account_bp = account_bp
        self.risk_pct   = risk_pct
        self._orb_high: Optional[float] = None
        self._orb_low:  Optional[float] = None
        self._orb_set:  bool = False
        self._fired:    bool = False   # one trade per day
        self._bars_above: int = 0
        self._bars_below: int = 0
        self._date: str = ""

    def set_daily_context(self, trade_date: str) -> None:
        if trade_date != self._date:
            self._orb_high = None
            self._orb_low  = None
            self._orb_set  = False
            self._fired    = False
            self._bars_above = 0
            self._bars_below = 0
            self._date = trade_date

    async def on_tick(self, tick) -> Optional[Signal]: return None
    async def on_candle(self, candle) -> Optional[Signal]: return None
    async def on_greek_update(self, greeks) -> Optional[Signal]: return None

    async def on_bar(
        self,
        candle,
        walls: List,
        term_analyses: List,
        available_bp: float,
        trade_date: str = "",
    ) -> Optional[Signal]:
        if trade_date:
            self.set_daily_context(trade_date)
        if self._fired or not term_analyses:
            return None

        spot = candle.close
        ts   = candle.timestamp
        # ET hour (UTC-4 EDT approximation)
        et_h = ts.hour - 4
        et_m = et_h * 60 + ts.minute

        # Build ORB from 9:30–10:00 ET (first 30 min)
        if 9 * 60 + 30 <= et_m < 10 * 60:
            if self._orb_high is None:
                self._orb_high = candle.high
                self._orb_low  = candle.low
            else:
                self._orb_high = max(self._orb_high, candle.high)
                self._orb_low  = min(self._orb_low,  candle.low)
            return None

        # Lock in ORB at 10:00
        if not self._orb_set and self._orb_high and self._orb_low:
            rng = (self._orb_high - self._orb_low) / self._orb_low
            if rng < self.MIN_RANGE_PCT:
                self._orb_set = True  # too narrow — won't trade today
                self._fired   = True
                return None
            self._orb_set = True

        if not self._orb_set or et_m >= 15 * 60 + 30:
            return None

        # Count confirmation bars
        if spot > self._orb_high:
            self._bars_above += 1
            self._bars_below  = 0
        elif spot < self._orb_low:
            self._bars_below += 1
            self._bars_above  = 0
        else:
            self._bars_above = max(0, self._bars_above - 1)
            self._bars_below = max(0, self._bars_below - 1)
            return None

        if self._bars_above < self.CONFIRM_BARS and self._bars_below < self.CONFIRM_BARS:
            return None

        # Score the setup
        orb_width  = self._orb_high - self._orb_low
        breakout   = (spot - self._orb_high) / orb_width if self._bars_above >= self.CONFIRM_BARS \
                     else (self._orb_low - spot) / orb_width
        score = float(np.clip(0.55 + breakout * 2.0, 0.0, 1.0))
        if score < self.SCORE_THRESHOLD:
            return None

        dte0   = next((a for a in term_analyses if a.dte == 0), term_analyses[0])
        expiry = dte0.expiry

        # Strike: highest-gamma wall in breakout direction — OTM magnet target
        if self._bars_above >= self.CONFIRM_BARS:
            sig_type = SignalType.BUY_CALL
            strike   = round(_gamma_wall_strike(walls, spot, 1, fallback=3.0))
            reasoning = f"ORB breakout UP  range={self._orb_low:.1f}-{self._orb_high:.1f}  spot={spot:.1f}  K={strike}"
        else:
            sig_type = SignalType.BUY_PUT
            strike   = round(_gamma_wall_strike(walls, spot, -1, fallback=3.0))
            reasoning = f"ORB breakdown DOWN  range={self._orb_low:.1f}-{self._orb_high:.1f}  spot={spot:.1f}  K={strike}"

        # Kelly: ORB wins ~55-60% historically when confirmed
        premium = max(dte0.atm_iv * spot * (0.25 / 365.0) ** 0.5 * 0.4, 0.50)
        kelly_f = np.clip((0.57 * 2.5 - 0.43) / 2.5, 0.0, 0.25)
        n = max(1, int(kelly_f * self.account_bp * self.risk_pct / (premium * 100)))

        self._fired = True
        return Signal(
            signal_type=sig_type, symbol=candle.symbol, timestamp=candle.timestamp,
            strike=strike, expiry=expiry, confidence=score, position_size=n,
            strategy_name=self.name,
            metadata={"play_type": "orb", "condition_score": score,
                      "reasoning": reasoning, "dte": dte0.dte,
                      "call_strike": strike, "put_strike": strike},
        )


# ============================================================================
# VWAP DEVIATION
# Price stretched far from VWAP = mean reversion opportunity OR trend signal.
# VWAP cross with volume = trend confirmation for directional options.
# ============================================================================

class VWAPStrategy(Strategy):
    """
    Two modes:
    - VWAP cross (price crosses VWAP with vol expansion) → trend continuation
    - VWAP stretch (>0.5% from VWAP + volume spike) → mean reversion

    VWAP cross is higher-conviction for 0DTE. Reversion used when gap is extreme.
    """

    STRETCH_PCT   = 0.005   # 0.5% from VWAP to consider stretched
    VOL_MULT      = 1.5     # volume must be 1.5× avg to confirm cross
    SCORE_THRESHOLD = 0.82  # max formula score is 0.78 → effectively disabled until score logic is reworked

    def __init__(self, account_bp: float = 2500.0, risk_pct: float = 0.15):
        super().__init__("VWAP")
        self.account_bp  = account_bp
        self.risk_pct    = risk_pct
        self._cum_pv:    float = 0.0   # cumulative price×volume
        self._cum_vol:   float = 0.0   # cumulative volume
        self._vwap:      float = 0.0
        self._vol_hist:  List[float] = []
        self._prev_above: Optional[bool] = None  # was price above VWAP last bar?
        self._fired:     bool = False
        self._date:      str  = ""

    def set_daily_context(self, trade_date: str) -> None:
        if trade_date != self._date:
            self._cum_pv   = 0.0
            self._cum_vol  = 0.0
            self._vwap     = 0.0
            self._vol_hist = []
            self._prev_above = None
            self._fired    = False
            self._date     = trade_date

    async def on_tick(self, tick) -> Optional[Signal]: return None
    async def on_candle(self, candle) -> Optional[Signal]: return None
    async def on_greek_update(self, greeks) -> Optional[Signal]: return None

    async def on_bar(
        self,
        candle,
        walls: List,
        term_analyses: List,
        available_bp: float,
        trade_date: str = "",
    ) -> Optional[Signal]:
        if trade_date:
            self.set_daily_context(trade_date)
        if self._fired or not term_analyses:
            return None

        spot = candle.close
        vol  = float(candle.volume) if candle.volume else 1.0
        typ  = (candle.high + candle.low + candle.close) / 3.0

        # Update VWAP
        self._cum_pv  += typ * vol
        self._cum_vol += vol
        self._vwap     = self._cum_pv / max(self._cum_vol, 1.0)
        self._vol_hist.append(vol)
        if len(self._vol_hist) > 20:
            self._vol_hist.pop(0)

        ts   = candle.timestamp
        et_m = (ts.hour - 4) * 60 + ts.minute
        if et_m < 10 * 60 or et_m > 15 * 60 + 30:
            self._prev_above = spot > self._vwap
            return None

        avg_vol    = float(np.mean(self._vol_hist)) if self._vol_hist else 1.0
        dev_pct    = (spot - self._vwap) / self._vwap
        above_vwap = spot > self._vwap
        crossed    = (self._prev_above is not None) and (above_vwap != self._prev_above)
        vol_conf   = vol >= avg_vol * self.VOL_MULT

        self._prev_above = above_vwap

        sig_type = None
        score    = 0.0

        if crossed and vol_conf:
            # VWAP cross with volume = trend confirmation
            sig_type = SignalType.BUY_CALL if above_vwap else SignalType.BUY_PUT
            score    = float(np.clip(0.63 + min(vol / avg_vol - 1.5, 1.0) * 0.15, 0.0, 1.0))
        elif abs(dev_pct) > self.STRETCH_PCT and vol_conf:
            # Extreme stretch: fade back toward VWAP
            sig_type = SignalType.BUY_PUT if above_vwap else SignalType.BUY_CALL
            score    = float(np.clip(0.55 + (abs(dev_pct) - self.STRETCH_PCT) * 20, 0.0, 1.0))

        if sig_type is None or score < self.SCORE_THRESHOLD:
            return None

        dte0   = next((a for a in term_analyses if a.dte == 0), term_analyses[0])
        _dir   = 1 if sig_type == SignalType.BUY_CALL else -1
        strike = round(_gamma_wall_strike(walls, spot, _dir, fallback=2.0))
        premium = max(dte0.atm_iv * spot * (0.25 / 365.0) ** 0.5 * 0.4, 0.50)
        kelly_f = np.clip((0.55 * 2.5 - 0.45) / 2.5, 0.0, 0.25)
        n = max(1, int(kelly_f * self.account_bp * self.risk_pct / (premium * 100)))

        self._fired = True
        return Signal(
            signal_type=sig_type, symbol=candle.symbol, timestamp=candle.timestamp,
            strike=strike, expiry=dte0.expiry, confidence=score, position_size=n,
            strategy_name=self.name,
            metadata={"play_type": "vwap", "condition_score": score,
                      "vwap": round(self._vwap, 2), "dev_pct": round(dev_pct * 100, 3),
                      "crossed": crossed, "dte": dte0.dte,
                      "call_strike": strike, "put_strike": strike},
        )


# ============================================================================
# GAP FILL
# SPY gaps up or down vs prior close. Small gaps (<1%) fill ~70% of the time.
# Large gaps (>1.5%) often continue. Trade the statistically likely outcome.
# ============================================================================

class GapFillStrategy(Strategy):
    """
    Gap fill: if SPY opens with a gap, bet on whether it fills or continues.
    - Small gap (0.3-1.0%): fade → buy against the gap direction
    - Large gap (>1.5%): continuation → buy with the gap direction
    - Confirm with first 15-min price action before entry
    """

    SMALL_GAP  = 0.003  # 0.3%
    LARGE_GAP  = 0.015  # 1.5%
    SCORE_THRESHOLD = 0.78  # raised: gap-fills need strong confirmation to overcome premium

    def __init__(self, account_bp: float = 2500.0, risk_pct: float = 0.15):
        super().__init__("GapFill")
        self.account_bp  = account_bp
        self.risk_pct    = risk_pct
        self._gap_pct:   float = 0.0
        self._prior_close: float = 0.0
        self._fired:     bool = False
        self._date:      str  = ""

    def set_daily_context(self, gap_pct: float, prior_close: float, trade_date: str) -> None:
        if trade_date != self._date:
            self._gap_pct     = gap_pct
            self._prior_close = prior_close
            self._fired       = False
            self._date        = trade_date

    async def on_tick(self, tick) -> Optional[Signal]: return None
    async def on_candle(self, candle) -> Optional[Signal]: return None
    async def on_greek_update(self, greeks) -> Optional[Signal]: return None

    async def on_bar(
        self,
        candle,
        walls: List,
        term_analyses: List,
        available_bp: float,
        trade_date: str = "",
        gap_pct: float = 0.0,
        prior_close: float = 0.0,
    ) -> Optional[Signal]:
        if trade_date:
            self.set_daily_context(gap_pct, prior_close, trade_date)
        if self._fired or not term_analyses or abs(self._gap_pct) < self.SMALL_GAP:
            return None

        spot = candle.close
        ts   = candle.timestamp
        et_m = (ts.hour - 4) * 60 + ts.minute

        # Enter between 10:00-10:30 AM (after opening noise settles)
        if et_m < 10 * 60 or et_m > 10 * 60 + 30:
            return None

        gap = self._gap_pct
        abs_gap = abs(gap)

        # Small gap: expect fill (fade)
        # Large gap: expect continuation
        if abs_gap <= self.LARGE_GAP:
            # Fade the gap
            sig_type  = SignalType.BUY_PUT if gap > 0 else SignalType.BUY_CALL
            target    = self._prior_close
            conf_move = abs(spot - target) / max(target, 1.0)
            score     = float(np.clip(0.55 + abs_gap * 10 + conf_move * 5, 0.0, 1.0))
            reasoning = f"gap_fill: gap={gap*100:+.2f}% fading back to {target:.1f}"
        else:
            # Continuation
            sig_type  = SignalType.BUY_CALL if gap > 0 else SignalType.BUY_PUT
            score     = float(np.clip(0.60 + (abs_gap - self.LARGE_GAP) * 5, 0.0, 1.0))
            reasoning = f"gap_continuation: gap={gap*100:+.2f}% continuing"

        if score < self.SCORE_THRESHOLD:
            return None

        dte0   = next((a for a in term_analyses if a.dte == 0), term_analyses[0])
        _dir   = 1 if sig_type == SignalType.BUY_CALL else -1
        strike = round(_gamma_wall_strike(walls, spot, _dir, fallback=3.0))
        premium = max(dte0.atm_iv * spot * (0.25 / 365.0) ** 0.5 * 0.4, 0.50)
        kelly_f = np.clip((0.57 * 2.5 - 0.43) / 2.5, 0.0, 0.25)
        n = max(1, int(kelly_f * self.account_bp * self.risk_pct / (premium * 100)))

        self._fired = True
        return Signal(
            signal_type=sig_type, symbol=candle.symbol, timestamp=candle.timestamp,
            strike=strike, expiry=dte0.expiry, confidence=score, position_size=n,
            strategy_name=self.name,
            metadata={"play_type": "gap_fill", "condition_score": score,
                      "gap_pct": round(gap * 100, 3), "prior_close": self._prior_close,
                      "reasoning": reasoning, "dte": dte0.dte,
                      "call_strike": strike, "put_strike": strike},
        )


# ============================================================================
# MOMENTUM CONTINUATION
# After a strong directional move (first hour), ride the continuation.
# High VIX days with strong early momentum often continue into close.
# ============================================================================

class MomentumStrategy(Strategy):
    """
    If SPY moves >0.6% in one direction in the first 60 minutes with
    accelerating volume, the move tends to continue into close.
    Buy options in the direction of the move. Enter on any pullback to VWAP.
    """

    MIN_MOVE_PCT  = 0.006   # 0.6% move in first hour
    VOL_ACCEL     = 1.3     # volume must be accelerating (1.3× prior period)
    SCORE_THRESHOLD = 0.70  # raised: momentum needs clear vol-confirmed move

    def __init__(self, account_bp: float = 2500.0, risk_pct: float = 0.15):
        super().__init__("Momentum")
        self.account_bp   = account_bp
        self.risk_pct     = risk_pct
        self._open_price: float = 0.0
        self._vol_first:  float = 0.0   # volume first 30 min
        self._vol_second: float = 0.0   # volume next 30 min
        self._phase:      int   = 0     # 0=building, 1=watching, 2=fired
        self._direction:  int   = 0
        self._date:       str   = ""

    def set_daily_context(self, trade_date: str) -> None:
        if trade_date != self._date:
            self._open_price = 0.0
            self._vol_first  = 0.0
            self._vol_second = 0.0
            self._phase      = 0
            self._direction  = 0
            self._date       = trade_date

    async def on_tick(self, tick) -> Optional[Signal]: return None
    async def on_candle(self, candle) -> Optional[Signal]: return None
    async def on_greek_update(self, greeks) -> Optional[Signal]: return None

    async def on_bar(
        self,
        candle,
        walls: List,
        term_analyses: List,
        available_bp: float,
        trade_date: str = "",
    ) -> Optional[Signal]:
        if trade_date:
            self.set_daily_context(trade_date)
        if self._phase == 2 or not term_analyses:
            return None

        spot = candle.close
        vol  = float(candle.volume) if candle.volume else 0.0
        ts   = candle.timestamp
        et_m = (ts.hour - 4) * 60 + ts.minute

        # Record open
        if self._open_price == 0.0 and et_m >= 9 * 60 + 30:
            self._open_price = candle.open

        # Accumulate first-half volume (9:30-10:00)
        if et_m < 10 * 60:
            self._vol_first += vol
            return None

        # Accumulate second-half volume (10:00-10:30)
        if et_m < 10 * 60 + 30:
            self._vol_second += vol
            return None

        # After 10:30 — evaluate setup
        if et_m > 13 * 60 or self._open_price == 0.0:
            return None

        move_pct  = (spot - self._open_price) / self._open_price
        vol_accel = self._vol_second / max(self._vol_first, 1.0)

        if abs(move_pct) < self.MIN_MOVE_PCT or vol_accel < self.VOL_ACCEL:
            return None

        direction = 1 if move_pct > 0 else -1
        score = float(np.clip(
            0.55 + abs(move_pct) * 20 + (vol_accel - 1.3) * 0.15, 0.0, 1.0
        ))

        if score < self.SCORE_THRESHOLD:
            return None

        dte0   = next((a for a in term_analyses if a.dte == 0), term_analyses[0])
        strike = round(_gamma_wall_strike(walls, spot, direction, fallback=3.0))
        sig_type = SignalType.BUY_CALL if direction > 0 else SignalType.BUY_PUT

        premium = max(dte0.atm_iv * spot * (0.25 / 365.0) ** 0.5 * 0.4, 0.50)
        kelly_f = np.clip((0.55 * 2.5 - 0.45) / 2.5, 0.0, 0.25)
        n = max(1, int(kelly_f * self.account_bp * self.risk_pct / (premium * 100)))

        self._phase = 2
        return Signal(
            signal_type=sig_type, symbol=candle.symbol, timestamp=candle.timestamp,
            strike=strike, expiry=dte0.expiry, confidence=score, position_size=n,
            strategy_name=self.name,
            metadata={"play_type": "momentum", "condition_score": score,
                      "move_pct": round(move_pct * 100, 3),
                      "vol_accel": round(vol_accel, 2),
                      "dte": dte0.dte, "call_strike": strike, "put_strike": strike},
        )


# ============================================================================
# IV PERCENTILE MEAN REVERSION
# When IV rank > 80: sell premium (straddle/strangle).
# When IV rank < 20: buy straddle (vol is cheap, expect expansion).
# ============================================================================

class IVPercentileStrategy(Strategy):
    """
    Volatility mean reversion: IV always reverts to its historical average.
    High IVR (>80) = sell straddle. Low IVR (<20) = buy straddle.
    Uses the IVR seed data from Tastytrade IV history.
    """

    SELL_IVR  = 80
    BUY_IVR   = 20
    SCORE_THRESHOLD = 0.65

    def __init__(self, account_bp: float = 2500.0, risk_pct: float = 0.12):
        super().__init__("IVPercentile")
        self.account_bp = account_bp
        self.risk_pct   = risk_pct
        self._ivr:  float = 50.0
        self._fired: bool = False
        self._date:  str  = ""

    def set_daily_context(self, ivr: float, trade_date: str) -> None:
        if trade_date != self._date:
            self._ivr   = ivr
            self._fired = False
            self._date  = trade_date

    async def on_tick(self, tick) -> Optional[Signal]: return None
    async def on_candle(self, candle) -> Optional[Signal]: return None
    async def on_greek_update(self, greeks) -> Optional[Signal]: return None

    async def on_bar(
        self,
        candle,
        walls: List,
        term_analyses: List,
        available_bp: float,
        trade_date: str = "",
        ivr: float = 50.0,
    ) -> Optional[Signal]:
        if trade_date:
            self.set_daily_context(ivr, trade_date)
        if self._fired or not term_analyses:
            return None

        ts   = candle.timestamp
        et_m = (ts.hour - 4) * 60 + ts.minute
        if et_m < 10 * 60 or et_m > 14 * 60:
            return None

        spot  = candle.close
        ivr   = self._ivr
        dte0  = next((a for a in term_analyses if a.dte == 0), term_analyses[0])

        if ivr >= self.SELL_IVR:
            sig_type = SignalType.SELL_STRADDLE
            strike   = round(spot)
            score    = float(np.clip(0.60 + (ivr - 80) / 100, 0.0, 1.0))
        elif ivr <= self.BUY_IVR:
            sig_type = SignalType.STRADDLE
            strike   = round(spot)
            score    = float(np.clip(0.60 + (20 - ivr) / 100, 0.0, 1.0))
        else:
            return None

        if score < self.SCORE_THRESHOLD:
            return None

        premium = max(dte0.atm_iv * spot * (0.25 / 365.0) ** 0.5 * 0.4 * 2, 1.0)
        kelly_f = 0.10
        n = max(1, int(kelly_f * self.account_bp * self.risk_pct / (premium * 100)))

        self._fired = True
        return Signal(
            signal_type=sig_type, symbol=candle.symbol, timestamp=candle.timestamp,
            strike=strike, expiry=dte0.expiry, confidence=score, position_size=n,
            strategy_name=self.name,
            metadata={"play_type": "iv_percentile", "condition_score": score,
                      "ivr": ivr, "dte": dte0.dte,
                      "call_strike": strike, "put_strike": strike},
        )
