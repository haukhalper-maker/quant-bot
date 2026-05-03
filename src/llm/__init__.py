"""
LLM Reasoning Engine — Local LLM integration for institutional trade intelligence.

Supports any OpenAI-compatible local endpoint:
  - Ollama       → http://localhost:11434/v1
  - LM Studio    → http://localhost:1234/v1
  - LocalAI      → http://localhost:8080/v1
  - vLLM         → http://localhost:8000/v1

Usage:
    client = LocalLLMClient(base_url="http://localhost:11434/v1", model="llama3.1:8b")
    engine = TradeReasoningEngine(client)
    decision = await engine.evaluate_signal(ctx)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger


# ============================================================================
# DATA CONTRACTS
# ============================================================================


@dataclass
class TradeContext:
    """Complete market context snapshot fed to the LLM for analysis."""

    symbol: str
    timestamp: datetime
    spot_price: float

    # Volatility
    iv_percentile: float        # 0-100, where current IV sits vs 30d history
    implied_vol: float          # Current front-month ATM IV
    realized_vol: float         # 30-day realized vol (HV30)
    iv_rank: float = 0.0        # 0-100, IV rank (different from percentile)

    # Portfolio Greeks at portfolio level
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0

    # Proposed signal
    signal_type: Optional[str] = None
    signal_confidence: float = 0.5
    signal_strike: float = 0.0
    signal_expiry: Optional[str] = None
    signal_position_size: int = 1

    # Price action context
    recent_returns: List[float] = field(default_factory=list)   # Last 20 daily returns
    recent_iv_history: List[float] = field(default_factory=list)  # Last 20 IV readings

    # Market microstructure
    volume_poc: float = 0.0     # Point of Control (highest volume price)
    bid_ask_spread_bps: float = 0.0

    # Open interest (key strikes only)
    open_interest: Dict[str, int] = field(default_factory=dict)

    # Current open positions for this symbol
    existing_positions: List[Dict[str, Any]] = field(default_factory=list)

    # Detected regime (if already computed)
    market_regime: Optional[str] = None
    regime_confidence: float = 0.0


@dataclass
class LLMTradeDecision:
    """
    Structured trade decision returned by the LLM reasoning engine.
    All fields have sensible defaults so callers never need to branch on None.
    """

    action: str                         # 'enter' | 'skip' | 'reduce' | 'exit' | 'hold'
    confidence: float                   # 0.0-1.0 LLM confidence
    position_size_multiplier: float     # 0.0-2.0 applied to base position size
    reasoning: str                      # 2-3 sentence rationale
    key_risks: List[str]                # Top risk factors identified
    suggested_stop_loss: Optional[float] = None     # % from entry (e.g. 0.20 = 20%)
    suggested_take_profit: Optional[float] = None   # % from entry
    monitoring_triggers: List[str] = field(default_factory=list)  # Conditions to watch
    source: str = "llm"                 # 'llm' | 'heuristic'


@dataclass
class RegimeAnalysis:
    """Market regime classification from LLM."""

    regime: str          # 'trending_up' | 'trending_down' | 'mean_reverting' | 'high_vol' | 'low_vol' | 'breakout'
    confidence: float    # 0.0-1.0
    strength: str        # 'weak' | 'moderate' | 'strong'
    key_observation: str
    source: str = "llm"


# ============================================================================
# PROMPT LIBRARY
# ============================================================================

_SYSTEM_PROMPT = """\
You are a senior quantitative options trader at a tier-1 institutional hedge fund.
You have deep expertise in volatility trading, options Greeks, and market microstructure.

Your task is to evaluate trade signals from algorithmic strategies and provide precise,
risk-adjusted decisions. You are disciplined: you skip marginal setups.

Rules:
- Respond ONLY with valid JSON matching the exact schema provided.
- Never add commentary outside the JSON object.
- Be conservative: when in doubt, action = "skip".
- Account for portfolio-level Greeks, not just the proposed trade in isolation.
- IV/RV ratio is the single most important factor for vol strategies."""


def _trade_eval_prompt(ctx: TradeContext) -> str:
    pos_summary = "None"
    if ctx.existing_positions:
        pos_summary = "; ".join(
            "{symbol} {type} qty={qty} pnl={pnl:.0f}".format(
                symbol=p.get("symbol", "?"),
                type=p.get("option_type", "?"),
                qty=p.get("quantity", 0),
                pnl=p.get("unrealized_pnl", 0),
            )
            for p in ctx.existing_positions[:5]
        )

    rets = ctx.recent_returns[-15:]
    ret_str = ", ".join(f"{r:.3%}" for r in rets) if rets else "N/A"
    iv_hist = ctx.recent_iv_history[-10:]
    iv_str = ", ".join(f"{v:.1%}" for v in iv_hist) if iv_hist else "N/A"
    iv_rv = ctx.implied_vol / max(ctx.realized_vol, 0.001)

    return f"""\
Evaluate this options signal and return a JSON trade decision.

=== MARKET: {ctx.symbol} @ {ctx.timestamp.strftime('%Y-%m-%d %H:%M')} UTC ===
Spot: ${ctx.spot_price:.2f}
Regime: {ctx.market_regime or 'unknown'} (confidence: {ctx.regime_confidence:.0%})
Volume POC: ${ctx.volume_poc:.2f}  Distance from POC: {(ctx.spot_price - ctx.volume_poc) / max(ctx.volume_poc, 1):.2%}

=== VOLATILITY ===
IV Percentile (30d): {ctx.iv_percentile:.1f}th
IV Rank (30d):       {ctx.iv_rank:.1f}
Implied Vol (ATM):   {ctx.implied_vol:.1%}
Realized Vol (HV30): {ctx.realized_vol:.1%}
IV/RV Ratio:         {iv_rv:.2f}x  {"(vol CHEAP — favor buying)" if iv_rv < 1.1 else "(vol RICH — favor selling)" if iv_rv > 1.5 else "(vol FAIR)"}

=== PORTFOLIO GREEKS ===
Delta: {ctx.portfolio_delta:.1f}  Gamma: {ctx.portfolio_gamma:.4f}
Vega:  {ctx.portfolio_vega:.1f}   Theta: {ctx.portfolio_theta:.2f}

=== PROPOSED SIGNAL ===
Type:            {ctx.signal_type}
Strike:          {ctx.signal_strike or 'ATM'}
Expiry:          {ctx.signal_expiry or 'front-month'}
Position Size:   {ctx.signal_position_size} contract(s)
Strategy Score:  {ctx.signal_confidence:.1%}

=== PRICE ACTION ===
Recent Daily Returns: {ret_str}
IV History:           {iv_str}

=== OPEN POSITIONS ({ctx.symbol}) ===
{pos_summary}

Respond with ONLY this JSON (no markdown, no extra text):
{{
  "action": "enter" | "skip" | "reduce" | "exit" | "hold",
  "confidence": <float 0.0-1.0>,
  "position_size_multiplier": <float 0.0-2.0>,
  "reasoning": "<2-3 sentence rationale>",
  "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
  "suggested_stop_loss": <null | float percent e.g. 0.20>,
  "suggested_take_profit": <null | float percent e.g. 0.50>,
  "monitoring_triggers": ["<condition1>", "<condition2>"]
}}"""


def _regime_prompt(
    symbol: str,
    spot: float,
    poc: float,
    returns: List[float],
    iv_history: List[float],
) -> str:
    rets = returns[-20:]
    ivs = iv_history[-20:]
    ret_str = ", ".join(f"{r:.3%}" for r in rets) if rets else "N/A"
    iv_str = ", ".join(f"{v:.1%}" for v in ivs) if ivs else "N/A"

    return f"""\
Classify the current market regime for {symbol}.

Spot: ${spot:.2f}
POC:  ${poc:.2f}  (Distance: {(spot - poc) / max(poc, 1):.2%})

Daily Returns (last 20): {ret_str}
IV History   (last 20):  {iv_str}

Respond with ONLY this JSON:
{{
  "regime": "trending_up" | "trending_down" | "mean_reverting" | "high_vol" | "low_vol" | "breakout",
  "confidence": <float 0.0-1.0>,
  "strength": "weak" | "moderate" | "strong",
  "key_observation": "<one precise sentence>"
}}"""


# ============================================================================
# LOCAL LLM CLIENT
# ============================================================================


class LocalLLMClient:
    """
    Async, OpenAI-compatible client for local language models.

    Configuration:
        base_url  — endpoint root (e.g. http://localhost:11434/v1 for Ollama)
        model     — model identifier (e.g. "llama3.1:8b", "mistral:7b", "qwen2.5:14b")
        api_key   — any string; local servers typically ignore this
        timeout   — seconds before giving up on a generation
        temperature — keep LOW (0.05-0.15) for consistent structured decisions
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.1:8b",
        api_key: str = "local",
        timeout: int = 90,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        min_call_interval: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._min_call_interval = min_call_interval
        self._last_call_ts: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"LocalLLMClient -> {self.base_url}  model={self.model}")

    async def _session_(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _rate_limit(self) -> None:
        gap = self._min_call_interval - (time.monotonic() - self._last_call_ts)
        if gap > 0:
            await asyncio.sleep(gap)
        self._last_call_ts = time.monotonic()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
    ) -> Optional[str]:
        """
        Chat completion call. Returns raw response text or None on failure.
        json_mode hints to the model to return valid JSON (supported by most local servers).
        """
        await self._rate_limit()
        session = await self._session_()

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with session.post(
                f"{self.base_url}/chat/completions", json=payload
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"LLM HTTP {resp.status}: {body[:300]}")
                    return None
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except asyncio.TimeoutError:
            logger.error(f"LLM timed out (>{self._timeout.total}s) — model={self.model}")
        except aiohttp.ClientConnectorError:
            logger.error(f"LLM connection refused at {self.base_url}")
        except Exception as e:
            logger.error(f"LLM request error: {e}")
        return None

    async def is_available(self) -> bool:
        """Probe the LLM server with a 5-second timeout."""
        try:
            session = await self._session_()
            async with session.get(
                f"{self.base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                ok = resp.status == 200
                if ok:
                    logger.info(f"LLM server reachable at {self.base_url}")
                return ok
        except Exception:
            return False


# ============================================================================
# TRADE REASONING ENGINE
# ============================================================================


class TradeReasoningEngine:
    """
    Bridges quantitative signals to local LLM for intelligent validation.

    Architecture:
      1. Strategy generates a Signal
      2. QuantBot builds a TradeContext and calls evaluate_signal()
      3. Engine sends structured prompt → local LLM → parses JSON response
      4. Returns LLMTradeDecision (action, confidence, sizing, risks, stops)
      5. If LLM unavailable: falls back to IV/RV heuristics (no outage)

    Regime detection is cached (default TTL=300s) since it's expensive.
    """

    def __init__(
        self,
        client: LocalLLMClient,
        fallback_on_error: bool = True,
        regime_cache_ttl: int = 300,
    ):
        self.client = client
        self.fallback_on_error = fallback_on_error
        self._regime_ttl = regime_cache_ttl
        self._regime_cache: Dict[str, tuple] = {}   # symbol → (RegimeAnalysis, ts)
        self._llm_available: Optional[bool] = None   # cached availability
        self._avail_checked_at: float = 0.0
        self._avail_recheck_interval: float = 60.0   # re-probe every 60s if unavailable
        logger.info("TradeReasoningEngine initialized")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def evaluate_signal(self, ctx: TradeContext) -> LLMTradeDecision:
        """
        Core entry point: evaluate a trade signal with LLM reasoning.
        Always returns a valid LLMTradeDecision — never raises.
        """
        if not await self._is_available():
            return self._heuristic_decision(ctx)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _trade_eval_prompt(ctx)},
        ]
        raw = await self.client.chat(messages, json_mode=True)
        if raw is None:
            return self._heuristic_decision(ctx)

        decision = self._parse_decision(raw, ctx)
        logger.info(
            f"LLM decision for {ctx.symbol} {ctx.signal_type}: "
            f"action={decision.action} conf={decision.confidence:.0%} "
            f"size_mult={decision.position_size_multiplier:.1f}x"
        )
        return decision

    async def detect_regime(
        self,
        symbol: str,
        spot: float,
        poc: float,
        returns: List[float],
        iv_history: List[float],
    ) -> RegimeAnalysis:
        """
        Detect market regime (cached). Returns RegimeAnalysis.
        Falls back to heuristic if LLM unavailable.
        """
        cached = self._regime_cache.get(symbol)
        if cached:
            analysis, ts = cached
            if time.monotonic() - ts < self._regime_ttl:
                return analysis

        if not await self._is_available():
            analysis = self._heuristic_regime(returns, iv_history)
            self._regime_cache[symbol] = (analysis, time.monotonic())
            return analysis

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _regime_prompt(symbol, spot, poc, returns, iv_history)},
        ]
        raw = await self.client.chat(messages, json_mode=True)
        if raw is None:
            analysis = self._heuristic_regime(returns, iv_history)
        else:
            analysis = self._parse_regime(raw, returns, iv_history)

        self._regime_cache[symbol] = (analysis, time.monotonic())
        logger.info(
            f"Regime for {symbol}: {analysis.regime} "
            f"({analysis.strength}, conf={analysis.confidence:.0%}) [{analysis.source}]"
        )
        return analysis

    def invalidate_availability_cache(self) -> None:
        """Force re-probe of LLM on next call (call after recovering from an outage)."""
        self._llm_available = None
        self._avail_checked_at = 0.0

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    async def _is_available(self) -> bool:
        now = time.monotonic()
        if self._llm_available is None or (
            not self._llm_available
            and now - self._avail_checked_at > self._avail_recheck_interval
        ):
            self._llm_available = await self.client.is_available()
            self._avail_checked_at = now
            if not self._llm_available:
                logger.warning(
                    f"Local LLM unreachable at {self.client.base_url}. "
                    "Falling back to heuristic mode. "
                    "Start Ollama with: `ollama run llama3.1:8b`"
                )
        return bool(self._llm_available)

    def _parse_decision(self, raw: str, ctx: TradeContext) -> LLMTradeDecision:
        try:
            # Strip markdown code fences if model emits them
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
                text = text.rstrip("`").strip()

            data = json.loads(text)
            return LLMTradeDecision(
                action=data.get("action", "skip"),
                confidence=float(data.get("confidence", 0.5)),
                position_size_multiplier=float(data.get("position_size_multiplier", 1.0)),
                reasoning=data.get("reasoning", ""),
                key_risks=data.get("key_risks", []),
                suggested_stop_loss=data.get("suggested_stop_loss"),
                suggested_take_profit=data.get("suggested_take_profit"),
                monitoring_triggers=data.get("monitoring_triggers", []),
                source="llm",
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM decision: {e}. Raw: {raw[:300]}")
            return self._heuristic_decision(ctx)

    def _parse_regime(self, raw: str, returns: List[float], iv_history: List[float]) -> RegimeAnalysis:
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
            data = json.loads(text)
            return RegimeAnalysis(
                regime=data.get("regime", "unknown"),
                confidence=float(data.get("confidence", 0.5)),
                strength=data.get("strength", "moderate"),
                key_observation=data.get("key_observation", ""),
                source="llm",
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return self._heuristic_regime(returns, iv_history)

    def _heuristic_decision(self, ctx: TradeContext) -> LLMTradeDecision:
        """
        Strict IV/RV quality gate. The regime engine does primary filtering;
        the heuristic acts as a final edge-quality check.

        Buy straddles only when IV is genuinely cheap vs realized vol (iv_rv < 1.10).
        Sell condors only when IV is genuinely rich vs realized vol (iv_rv > 1.45).
        The 1.10 / 1.45 thresholds represent real edge — between them is fair value.

        In crash/trending_bear regimes: relax sell requirement (no condors anyway due
        to sell_threshold=999) and focus on vol-buying quality.
        """
        from src.strategy import _REGIME_PARAMS

        sig = (ctx.signal_type or "").lower()
        iv_rv = ctx.implied_vol / max(ctx.realized_vol, 0.001)
        regime_key = ctx.market_regime or "elevated_vol"
        rp = _REGIME_PARAMS.get(regime_key, _REGIME_PARAMS["elevated_vol"])

        # --- Vol buying strategies (straddle, strangle, calendar) ---
        if any(k in sig for k in ("straddle", "strangle", "calendar", "buy_call", "buy_put")):
            # In crash/bear regimes, be slightly more lenient on the buy threshold
            # (IV hasn't spiked to full RV yet but will)
            buy_iv_rv_max = 1.30 if regime_key in ("crash", "trending_bear") else 1.10
            if iv_rv <= buy_iv_rv_max and ctx.iv_percentile < rp["buy_threshold"]:
                return LLMTradeDecision(
                    action="enter",
                    confidence=0.72,
                    position_size_multiplier=1.0,
                    reasoning=(
                        f"[heuristic] {regime_key}: IV/RV={iv_rv:.2f}x ≤ {buy_iv_rv_max:.2f} "
                        f"and IV%ile={ctx.iv_percentile:.0f} < {rp['buy_threshold']:.0f} — vol cheap, buying justified."
                    ),
                    key_risks=["vol could stay suppressed", "theta decay if no move"],
                    suggested_stop_loss=0.50,
                    suggested_take_profit=rp["take_profit_pct"],
                    source="heuristic",
                )
            return LLMTradeDecision(
                action="skip",
                confidence=0.6,
                position_size_multiplier=0.0,
                reasoning=(
                    f"[heuristic] {regime_key}: IV/RV={iv_rv:.2f}x (max={buy_iv_rv_max:.2f}) "
                    f"or IV%ile={ctx.iv_percentile:.0f} ≥ {rp['buy_threshold']:.0f} — vol not cheap enough to buy."
                ),
                key_risks=["overpaying for premium"],
                source="heuristic",
            )

        # --- Vol selling strategies (iron condor, butterfly, short strangle) ---
        if any(k in sig for k in ("iron_condor", "butterfly", "sell_call", "sell_put")):
            # Require genuine IV premium: iv_rv > 1.45 — this is where condors have edge.
            # Regime already filtered to iv_rv > 1.30 (calm_bull) or 1.45 (elevated).
            sell_iv_rv_min = max(rp["iv_rv_sell_min"], 1.40)
            if iv_rv >= sell_iv_rv_min and ctx.iv_percentile > rp["sell_threshold"]:
                return LLMTradeDecision(
                    action="enter",
                    confidence=0.75,
                    position_size_multiplier=1.0,
                    reasoning=(
                        f"[heuristic] {regime_key}: IV/RV={iv_rv:.2f}x ≥ {sell_iv_rv_min:.2f} "
                        f"and IV%ile={ctx.iv_percentile:.0f} > {rp['sell_threshold']:.0f} — premium rich, selling justified."
                    ),
                    key_risks=["gap risk", "gamma spike near expiry", "earnings surprise"],
                    suggested_stop_loss=rp["stop_loss_mult"],
                    suggested_take_profit=rp["take_profit_pct"],
                    source="heuristic",
                )
            return LLMTradeDecision(
                action="skip",
                confidence=0.6,
                position_size_multiplier=0.0,
                reasoning=(
                    f"[heuristic] {regime_key}: IV/RV={iv_rv:.2f}x (min={sell_iv_rv_min:.2f}) "
                    f"or IV%ile={ctx.iv_percentile:.0f} ≤ {rp['sell_threshold']:.0f} — premium not rich enough to sell."
                ),
                key_risks=["risk/reward unfavorable"],
                source="heuristic",
            )

        # --- Close / gamma scalp ---
        if "close" in sig:
            return LLMTradeDecision(
                action="exit",
                confidence=0.8,
                position_size_multiplier=1.0,
                reasoning="[heuristic] Close position signal received.",
                key_risks=[],
                source="heuristic",
            )

        return LLMTradeDecision(
            action="skip",
            confidence=0.5,
            position_size_multiplier=0.0,
            reasoning=f"[heuristic] Unknown signal type '{sig}' — defaulting to skip.",
            key_risks=["unknown strategy"],
            source="heuristic",
        )

    @staticmethod
    def _heuristic_regime(returns: List[float], iv_history: List[float]) -> RegimeAnalysis:  # noqa: E303
        import numpy as np

        if not returns or len(returns) < 5:
            return RegimeAnalysis(
                regime="unknown", confidence=0.0, strength="weak",
                key_observation="Insufficient price history.", source="heuristic"
            )

        arr = np.array(returns[-20:] if len(returns) >= 20 else returns)
        avg = float(np.mean(arr))
        vol = float(np.std(arr))
        recent_iv = iv_history[-1] if iv_history else 0.20

        if recent_iv > 0.35 or vol > 0.025:
            regime, obs = "high_vol", f"Realized daily vol={vol:.2%} | IV={recent_iv:.1%}"
        elif abs(avg) > 0.006 and vol < 0.015:
            regime = "trending_up" if avg > 0 else "trending_down"
            obs = f"Directional drift avg={avg:.3%} with low scatter vol={vol:.2%}"
        elif vol < 0.008:
            regime, obs = "low_vol", f"Suppressed daily vol={vol:.2%} — range-bound"
        else:
            regime, obs = "mean_reverting", f"avg={avg:.3%} vol={vol:.2%} — no trend"

        strength = "strong" if vol > 0.02 or abs(avg) > 0.008 else "moderate" if vol > 0.01 else "weak"

        return RegimeAnalysis(
            regime=regime, confidence=0.6, strength=strength,
            key_observation=obs, source="heuristic"
        )


# ============================================================================
# ZERO-DTE REASONING ENGINE
# ============================================================================


@dataclass
class ZeroDTEContext:
    """Structured context snapshot for 0DTE setup evaluation."""

    symbol: str
    timestamp: datetime
    spot: float

    # GEX / wall data
    wall_strike: float
    wall_type: str              # 'pin' | 'explosive'
    net_gex_dollars: float
    confluence_score: float     # 0-1

    # IV term structure
    dte: int
    atm_iv: float
    iv_premium_vs_next: float   # ratio of this DTE's IV vs next DTE out
    crush_probability: float    # 0-1
    expected_move_1sd: float    # $

    # Price behavior
    velocity_pct: float         # recent price velocity fraction
    deceleration: float         # positive = slowing (good for pin)

    # Play type chosen by quant layer
    play_type: str              # 'pin' | 'explosive' | 'iv_crush'
    condition_score: float      # 0-1 weighted setup score
    kelly_contracts: int

    # Portfolio state
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    available_bp: float = 0.0
    open_positions: int = 0

    # Call/put volume ratio at wall strike
    call_put_volume_ratio: float = 1.0  # >1 = more calls, <1 = more puts


@dataclass
class ZeroDTEDecision:
    """Decisive LLM output for a 0DTE setup."""
    action: str              # 'enter' | 'skip'
    confidence: float        # 0.0-1.0
    size_multiplier: float   # applied to kelly_contracts (0.5 = half size)
    reasoning: str           # ≤2 sentences
    key_risk: str            # single biggest risk
    source: str = "llm"      # 'llm' | 'heuristic'


_ZDTE_SYSTEM = """\
You are a decisive 0DTE options trader. You receive pre-computed quant signals.
Your ONLY job: confirm or reject the trade based on whether the edge is real.
Rules:
- Respond ONLY with valid JSON matching the schema exactly.
- Be binary: if the edge is marginal, action = "skip".
- Never add commentary outside the JSON.
- Trust the GEX data — gamma walls are real market structure."""


def _zdteprompt(ctx: ZeroDTEContext) -> str:
    gex_m = ctx.net_gex_dollars / 1_000_000
    return f"""\
0DTE SETUP EVALUATION — {ctx.symbol} @ ${ctx.spot:.2f}
Time: {ctx.timestamp.strftime('%H:%M ET')}  DTE: {ctx.dte}

GEX WALL:
  Strike: {ctx.wall_strike:.1f}  Type: {ctx.wall_type.upper()}
  Net GEX: ${gex_m:+.2f}M  Confluence: {ctx.confluence_score:.2f}/1.0

IV STRUCTURE:
  ATM IV: {ctx.atm_iv:.1%}  IV vs next DTE: {ctx.iv_premium_vs_next:.2f}×
  Crush probability: {ctx.crush_probability:.0%}
  Expected 1σ move: ${ctx.expected_move_1sd:.2f}

PRICE BEHAVIOR:
  Velocity: {ctx.velocity_pct:+.3%}/bar  Deceleration: {ctx.deceleration:+.2f}
  Call/Put vol ratio at wall: {ctx.call_put_volume_ratio:.2f}

QUANT RECOMMENDATION:
  Play: {ctx.play_type.upper()}
  Setup score: {ctx.condition_score:.2f}/1.00  (threshold 0.65)
  Kelly size: {ctx.kelly_contracts} contract(s)

PORTFOLIO:
  Delta: {ctx.portfolio_delta:.1f}  Gamma: {ctx.portfolio_gamma:.4f}
  Available BP: ${ctx.available_bp:,.0f}  Open positions: {ctx.open_positions}

Question: Does this setup have genuine edge, or is it marginal noise?

Respond ONLY with this JSON (no markdown):
{{
  "action": "enter" | "skip",
  "confidence": <float 0.0-1.0>,
  "size_multiplier": <float 0.5-1.5>,
  "reasoning": "<max 2 sentences>",
  "key_risk": "<single biggest risk in ≤10 words>"
}}"""


class ZeroDTEReasoningEngine:
    """
    LLM gate for 0DTE setups.

    Receives pre-computed quant signals (GEX, IV term structure, condition score)
    and returns a binary enter/skip decision with confidence.

    The LLM is NOT doing the math — that's already done. It's doing a sanity
    check: does everything cohere? Is there a reason to doubt the setup?

    Fallback: if LLM is unavailable, uses a simple heuristic (condition_score
    threshold + crush_probability) so the bot never goes dark.
    """

    MIN_CONFIDENCE = 0.65

    def __init__(self, client: LocalLLMClient, fallback_on_error: bool = True):
        self.client = client
        self.fallback_on_error = fallback_on_error
        self._available: Optional[bool] = None
        self._checked_at: float = 0.0
        logger.info("ZeroDTEReasoningEngine initialized")

    async def evaluate(self, ctx: ZeroDTEContext) -> ZeroDTEDecision:
        """Evaluate a 0DTE setup. Always returns a valid decision — never raises."""
        if not await self._is_available():
            return self._heuristic(ctx)

        messages = [
            {"role": "system", "content": _ZDTE_SYSTEM},
            {"role": "user",   "content": _zdteprompt(ctx)},
        ]
        raw = await self.client.chat(messages, json_mode=True)
        if raw is None:
            return self._heuristic(ctx)

        try:
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
            data = json.loads(text)
            dec = ZeroDTEDecision(
                action=data.get("action", "skip"),
                confidence=float(data.get("confidence", 0.5)),
                size_multiplier=float(data.get("size_multiplier", 1.0)),
                reasoning=data.get("reasoning", ""),
                key_risk=data.get("key_risk", ""),
                source="llm",
            )
            logger.info(
                f"[ZeroDTE-LLM] {ctx.play_type} {ctx.symbol} K={ctx.wall_strike:.1f} "
                f"→ {dec.action.upper()} conf={dec.confidence:.0%} | {dec.key_risk}"
            )
            return dec
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(f"[ZeroDTE-LLM] parse error: {exc}  raw={raw[:200]}")
            return self._heuristic(ctx)

    async def _is_available(self) -> bool:
        if self.client is None:
            return False
        now = time.monotonic()
        if self._available is None or (
            not self._available and now - self._checked_at > 60.0
        ):
            self._available = await self.client.is_available()
            self._checked_at = now
        return bool(self._available)

    def _heuristic(self, ctx: ZeroDTEContext) -> ZeroDTEDecision:
        """
        Fallback when LLM is down.
        Enter if condition_score ≥ 0.65 AND there is genuine IV/GEX edge.
        """
        has_gex_edge  = abs(ctx.net_gex_dollars) >= 500_000
        has_iv_edge   = ctx.crush_probability >= 0.25 or ctx.iv_premium_vs_next >= 1.10
        score_ok      = ctx.condition_score >= 0.65

        if score_ok and has_gex_edge and has_iv_edge:
            return ZeroDTEDecision(
                action="enter",
                confidence=min(ctx.condition_score, 0.78),
                size_multiplier=1.0,
                reasoning=(
                    f"[heuristic] {ctx.play_type}: GEX ${ctx.net_gex_dollars/1e6:.1f}M "
                    f"confluence={ctx.confluence_score:.2f} score={ctx.condition_score:.2f}"
                ),
                key_risk="GEX wall breaks intraday",
                source="heuristic",
            )
        reason = []
        if not score_ok:
            reason.append(f"score {ctx.condition_score:.2f} < 0.65")
        if not has_gex_edge:
            reason.append(f"|GEX| ${abs(ctx.net_gex_dollars)/1e6:.1f}M < $0.5M")
        if not has_iv_edge:
            reason.append(f"IV premium {ctx.iv_premium_vs_next:.2f}× / crush {ctx.crush_probability:.0%}")
        return ZeroDTEDecision(
            action="skip",
            confidence=0.6,
            size_multiplier=0.0,
            reasoning=f"[heuristic] Marginal setup: {'; '.join(reason)}",
            key_risk="insufficient edge",
            source="heuristic",
        )
