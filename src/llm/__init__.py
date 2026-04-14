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
        Rule-based fallback. Uses IV/RV ratio as primary signal quality filter.
        Mirrors institutional desk heuristics for vol strategies.
        """
        sig = (ctx.signal_type or "").lower()
        iv_rv = ctx.implied_vol / max(ctx.realized_vol, 0.001)

        # --- Vol buying strategies (straddle, strangle, calendar) ---
        if any(k in sig for k in ("straddle", "strangle", "calendar", "buy_call", "buy_put")):
            if iv_rv < 1.1 and ctx.iv_percentile < 30:
                return LLMTradeDecision(
                    action="enter",
                    confidence=0.65,
                    position_size_multiplier=1.0,
                    reasoning=f"[heuristic] IV/RV={iv_rv:.2f}x and IV%ile={ctx.iv_percentile:.0f} — vol is cheap, buying justified.",
                    key_risks=["vol could stay suppressed", "theta decay if no move"],
                    suggested_stop_loss=0.50,
                    suggested_take_profit=1.00,
                    source="heuristic",
                )
            return LLMTradeDecision(
                action="skip",
                confidence=0.6,
                position_size_multiplier=0.0,
                reasoning=f"[heuristic] IV/RV={iv_rv:.2f}x — vol not cheap enough to justify buying.",
                key_risks=["overpaying for premium"],
                source="heuristic",
            )

        # --- Vol selling strategies (iron condor, short strangle) ---
        if any(k in sig for k in ("iron_condor", "butterfly", "sell_call", "sell_put")):
            if iv_rv > 1.5 and ctx.iv_percentile > 70:
                return LLMTradeDecision(
                    action="enter",
                    confidence=0.70,
                    position_size_multiplier=1.0,
                    reasoning=f"[heuristic] IV/RV={iv_rv:.2f}x and IV%ile={ctx.iv_percentile:.0f} — vol is rich, selling justified.",
                    key_risks=["gap risk", "gamma spike near expiry", "earnings surprise"],
                    suggested_stop_loss=2.0,  # Close at 200% of premium received
                    suggested_take_profit=0.5,
                    source="heuristic",
                )
            return LLMTradeDecision(
                action="skip",
                confidence=0.6,
                position_size_multiplier=0.0,
                reasoning=f"[heuristic] IV/RV={iv_rv:.2f}x — premium not rich enough to sell.",
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
    def _heuristic_regime(returns: List[float], iv_history: List[float]) -> RegimeAnalysis:
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
