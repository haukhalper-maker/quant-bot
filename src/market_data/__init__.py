"""
Market Data Layer - Data ingestion, normalization, storage
"""

import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

import httpx
import numpy as np
from dotenv import load_dotenv
from loguru import logger


class DeviceChallengeRequired(Exception):
    """
    Raised by TastytradeConnector.connect() when Tastytrade requires a
    one-time device challenge.  An OTP has already been sent to the user's
    registered phone.  Retry with connect(otp='<code>').
    """


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class Tick:
    """Single market tick (quote or trade)"""

    symbol: str
    timestamp: datetime
    price: float
    size: int
    bid: float = 0.0
    bid_size: int = 0
    ask: float = 0.0
    ask_size: int = 0
    tick_type: str = "trade"  # 'trade', 'bid', 'ask', 'mid'
    open_interest: int = 0  # For options
    option_expiry: Optional[str] = None  # For options (YYYY-MM-DD)

    def __post_init__(self):
        if self.bid == 0 and self.ask == 0:
            self.bid = self.ask = self.price


@dataclass
class Candle:
    """OHLCV candle, optionally enriched with live IV from the data connector."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "1m"  # '1m', '5m', '15m', '1h', '1d'
    tick_count: int = 0
    # Set by TastytradeConnector when available; strategies prefer this over AR(1) simulation.
    implied_vol: Optional[float] = None
    iv_rank: Optional[float] = None
    hv30: Optional[float] = None


@dataclass
class Trade:
    """Executed trade record"""

    trade_id: str
    symbol: str
    timestamp: datetime
    side: str  # 'BUY', 'SELL'
    quantity: int
    price: float
    filled_quantity: int
    status: str  # 'PENDING', 'FILLED', 'REJECTED'
    pnl: float = 0.0


@dataclass
class OptionsChain:
    """Options contract specification"""

    symbol: str  # e.g., SPY
    expiry: str  # YYYY-MM-DD
    strike: float
    option_type: str  # 'CALL', 'PUT'
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0
    open_interest: int = 0
    volume: int = 0
    impliedVol: float = 0.0


# ============================================================================
# Data Source Interface
# ============================================================================


class DataConnector(ABC):
    """Abstract base for data sources"""

    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        logger.info(f"DataConnector '{name}' initialized")

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source"""
        pass

    @abstractmethod
    async def subscribe_ticks(self, symbol: str, callback) -> None:
        """Subscribe to real-time ticks"""
        pass

    @abstractmethod
    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        """Get historical tick data"""
        pass

    @abstractmethod
    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        """Get available options contracts"""
        pass


# ============================================================================
# MOCK DATA CONNECTOR (for testing & development)
# ============================================================================


class MockDataConnector(DataConnector):
    """Mock data source for backtesting without API"""

    def __init__(self):
        super().__init__("MockConnector")
        self.ticks: Dict[str, List[Tick]] = {}

    async def connect(self) -> bool:
        """Mock connection"""
        self.is_connected = True
        logger.info("Mock connector connected")
        return True

    async def disconnect(self) -> None:
        """Mock disconnection"""
        self.is_connected = False
        logger.info("Mock connector disconnected")

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        """Mock tick subscription"""
        logger.debug(f"Mock subscribed to ticks for {symbol}")

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        """Return empty list (to be overridden in tests)"""
        return []

    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        """Return empty chain (to be overridden in tests)"""
        return []


# ============================================================================
# REAL DATA CONNECTORS (Stubs for integration)
# ============================================================================


class InteractiveBrokersConnector(DataConnector):
    """
    [API: Interactive Brokers TWS]
    To implement: Use ibapi library, connect to TWS running locally
    Docs: https://github.com/InteractiveBrokers/tws-api
    """

    async def connect(self) -> bool:
        raise NotImplementedError("[API: Interactive Brokers] not yet implemented")

    async def disconnect(self) -> None:
        pass

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        raise NotImplementedError()

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        raise NotImplementedError()

    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        raise NotImplementedError()


class PolygonConnector(DataConnector):
    """
    [API: Polygon.io Real-time & Historical Data]
    To implement: Use polygon-api-client library
    Docs: https://polygon.io/docs/options/getting-started
    """

    async def connect(self) -> bool:
        raise NotImplementedError("[API: Polygon.io] not yet implemented")

    async def disconnect(self) -> None:
        pass

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        raise NotImplementedError()

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        raise NotImplementedError()

    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        raise NotImplementedError()


class AlpacaConnector(DataConnector):
    """
    [API: Alpaca Trade API]
    To implement: Use alpaca-trade-api library
    Docs: https://alpaca.markets/docs/api-references/
    """

    async def connect(self) -> bool:
        raise NotImplementedError("[API: Alpaca] not yet implemented")

    async def disconnect(self) -> None:
        pass

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        raise NotImplementedError()

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        raise NotImplementedError()

    async def get_options_chain(self, symbol: str, expiry: str) -> List[OptionsChain]:
        raise NotImplementedError()


class TastytradeConnector(DataConnector):
    """
    Live connection to the Tastytrade REST API.

    Authentication
    --------------
    Uses POST /sessions with username + password (the same credentials you use
    to log into tastytrade.com).  The v12 tastytrade SDK dropped username/password
    in favour of OAuth provider keys, so we talk to the underlying REST API
    directly via httpx.  No OAuth app or API key is required.

    IV / Vol data  (GET /market-metrics)
    --------------
    Per-symbol: implied_volatility_index, iv_rank, iv_percentile,
                implied_volatility_30_day, historical_volatility_30_day.
    Cached for 60 s so repeated candle ticks don't hammer the API.

    Options chain  (GET /option-chains/{symbol}/nested)
    ---------------
    Returns expiration dates and strikes.  Used to compute a BSM gamma surface
    for GammaImbalanceTrader.  Cached for 5 min.

    Open interest / real-time greeks
    ---------------------------------
    Require the DXLink WebSocket streamer.  Planned — for now gamma skew is
    computed from BSM gamma (IV × strike geometry) without OI weighting.
    The get_dealer_gamma_skew() docstring explains the approximation.
    """

    _BASE_URL = "https://api.tastytrade.com"
    _METRICS_TTL = 60.0     # seconds — market metrics cache TTL
    _CHAIN_TTL = 300.0      # seconds — option chain cache TTL

    def __init__(self, username: str = None, password: str = None):
        super().__init__("TastytradeConnector")
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.username = username or os.getenv("TASTYTRADE_USERNAME", "")
        self.password = password or os.getenv("TASTYTRADE_PASSWORD", "")
        self._session_token: Optional[str] = None
        self._challenge_token: Optional[str] = None   # device challenge token (transient)
        self._client: Optional[httpx.AsyncClient] = None
        self._accounts: List[dict] = []
        # symbol → {iv, iv_rank, iv_pct, hv30, iv30, fetched_at}
        self._metrics_cache: Dict[str, dict] = {}
        # symbol → {expirations: [...], fetched_at}
        self._chain_cache: Dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Connection lifecycle                                                  #
    # ------------------------------------------------------------------ #

    async def connect(self, otp: Optional[str] = None) -> bool:
        """
        Authenticate via POST /sessions and open a persistent httpx session.

        Device challenge flow
        ---------------------
        Tastytrade requires a one-time device challenge when connecting from a
        new IP / device.  The flow is:

          1. POST /sessions → 403  +  x-tastyworks-challenge-token response header
          2. POST /device-challenge  (using the challenge token)
             → Tastytrade sends an OTP via SMS
          3. POST /sessions again, passing both:
               X-Tastyworks-Challenge-Token: <token>
               X-Tastyworks-OTP: <code-from-sms>
             → 201 with session-token

        If `otp` is provided, step 3 is performed immediately.
        If not provided and the challenge is required, this method:
          - Triggers the SMS automatically
          - Raises DeviceChallengeRequired so the caller can prompt the user

        Once a device is authenticated with remember-me=True, subsequent
        connections from the same IP succeed without OTP.

        Parameters
        ----------
        otp : str, optional
            The SMS / email OTP code from step 2.  Pass this on retry.
        """
        if not self.username or not self.password:
            raise ValueError(
                "TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD must be set in .env"
            )

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._BASE_URL,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

        login_body = {
            "login": self.username,
            "password": self.password,
            "remember-me": True,
        }

        extra_headers: dict = {}
        if self._challenge_token:
            extra_headers["X-Tastyworks-Challenge-Token"] = self._challenge_token
        if otp:
            extra_headers["X-Tastyworks-OTP"] = otp

        resp = await self._client.post(
            "/sessions", json=login_body, headers=extra_headers
        )

        # ── Device challenge required ───────────────────────────────
        if resp.status_code == 403:
            body = resp.json().get("error", {})
            if body.get("code") == "device_challenge_required":
                token = resp.headers.get("x-tastyworks-challenge-token", "")
                if not token:
                    raise RuntimeError(
                        "Tastytrade returned device_challenge_required but "
                        "no challenge token was present in the response headers."
                    )
                self._challenge_token = token
                # Trigger the SMS/email OTP
                await self._client.post(
                    "/device-challenge",
                    headers={"X-Tastyworks-Challenge-Token": token},
                )
                raise DeviceChallengeRequired(
                    "Tastytrade sent a one-time code to your registered phone/email.\n"
                    "Call connect(otp='<code>') with the code to complete authentication.\n"
                    "Or run:  python -m src.main tastytrade-check"
                )
            await self._client.aclose()
            self._client = None
            raise RuntimeError(
                f"Tastytrade auth failed: HTTP {resp.status_code}\n{resp.text[:400]}"
            )

        if resp.status_code not in (200, 201):
            await self._client.aclose()
            self._client = None
            raise RuntimeError(
                f"Tastytrade auth failed: HTTP {resp.status_code}\n{resp.text[:400]}"
            )

        # ── Auth successful ─────────────────────────────────────────
        self._challenge_token = None   # clear — no longer needed
        payload = resp.json()["data"]
        self._session_token = payload["session-token"]
        self._client.headers.update({"Authorization": self._session_token})

        # Fetch and cache accounts
        acc_resp = await self._client.get("/customers/me/accounts")
        if acc_resp.status_code == 200:
            items = acc_resp.json()["data"]["items"]
            self._accounts = [item["account"] for item in items]

        self.is_connected = True
        username_display = payload.get("user", {}).get("username", self.username)
        logger.info(
            f"Tastytrade connected — user={username_display!r} "
            f"accounts={len(self._accounts)}"
        )
        return True

    async def disconnect(self) -> None:
        """Logout and close the HTTP client."""
        if self._client:
            try:
                await self._client.delete("/sessions")
            except Exception:
                pass
            await self._client.aclose()
            self._client = None
        self._session_token = None
        self.is_connected = False
        logger.info("Tastytrade disconnected")

    def _assert_connected(self) -> None:
        if not self.is_connected or self._client is None:
            raise RuntimeError(
                "TastytradeConnector is not connected — call connect() first."
            )

    # ------------------------------------------------------------------ #
    # Accounts                                                              #
    # ------------------------------------------------------------------ #

    def get_accounts(self) -> List[dict]:
        """Return the cached account list (populated during connect)."""
        return self._accounts

    # ------------------------------------------------------------------ #
    # Market metrics — IV, IV rank, historical vol                         #
    # ------------------------------------------------------------------ #

    async def get_market_metrics(self, *symbols: str) -> Dict[str, dict]:
        """
        Fetch implied-vol metrics for one or more symbols from
        GET /market-metrics.

        Returns a dict keyed by symbol.  Each value is:
          iv      — ATM implied vol (float, e.g. 0.20 = 20 %)
          iv_rank — IV Rank 0-100 (52-week high/low)
          iv_pct  — IV Percentile 0-100
          hv30    — 30-day historical/realised vol
          iv30    — 30-day implied vol (term-structure point)

        JSON keys from the API are kebab-case; the Tastytrade pydantic
        models dasherize snake_case field names to match, so we use the
        same transformation here.
        """
        self._assert_connected()
        now = time.monotonic()

        missing = [
            s for s in symbols
            if s not in self._metrics_cache
            or now - self._metrics_cache[s].get("fetched_at", 0) > self._METRICS_TTL
        ]

        if missing:
            resp = await self._client.get(
                "/market-metrics",
                params={"symbols": ",".join(missing)},
            )
            if resp.status_code != 200:
                logger.warning(
                    f"market-metrics HTTP {resp.status_code} for {missing}: "
                    f"{resp.text[:200]}"
                )
            else:
                for item in resp.json()["data"]["items"]:
                    sym = item.get("symbol", "")
                    self._metrics_cache[sym] = {
                        "iv":      _parse_float(item.get("implied-volatility-index"), 0.20),
                        "iv_rank": _parse_float(item.get("implied-volatility-index-rank"), 50.0),
                        "iv_pct":  _parse_float(item.get("implied-volatility-percentile"), 50.0),
                        "hv30":    _parse_float(item.get("historical-volatility-30-day"), 0.15),
                        "iv30":    _parse_float(item.get("implied-volatility-30-day"), 0.20),
                        "fetched_at": now,
                    }

        return {s: self._metrics_cache.get(s, {}) for s in symbols}

    async def get_atm_iv(self, symbol: str) -> float:
        """Return the current ATM IV for a symbol (float, annualised)."""
        metrics = await self.get_market_metrics(symbol)
        return metrics.get(symbol, {}).get("iv", 0.20)

    async def seed_iv_history(self, symbol: str, days: int = 30) -> List[float]:
        """
        Build a plausible `days`-long IV history bootstrapped from real
        market metrics.

        Why not random?
          The AR(1) simulation in VolatilityMeanReversionStrategy used to
          seed from np.random.normal(0.20, 0.04) — a pure guess. This
          replaces the starting value with the real observed IV from the
          Tastytrade API and uses the IV Rank to estimate the recent range.

        The strategy's `_iv_percentile()` needs enough history to rank the
        current IV.  We generate N AR(1) samples centred on `current_iv`,
        bounded by the inferred 52-week range, so the initial percentile
        reading is realistic.  History builds up in real-time as the bot
        appends new readings every candle.
        """
        metrics = await self.get_market_metrics(symbol)
        m = metrics.get(symbol, {})
        current_iv = m.get("iv", 0.20)
        iv_rank = m.get("iv_rank", 50.0) / 100.0       # normalise to 0–1

        # Infer approximate 52-week IV range from IVR.
        # IVR = (current - 52wk_low) / (52wk_high - 52wk_low)
        # Assume 52-week range ≈ 70 % of current IV (typical for SPY/SPX).
        range_52wk = current_iv * 0.70
        iv_low  = max(0.04, current_iv - iv_rank * range_52wk)
        iv_high = iv_low + range_52wk

        # AR(1) backward simulation ending at current_iv
        history = [current_iv]
        mean_rev = 0.08
        noise_std = 0.007
        for _ in range(days - 1):
            prev = history[-1]
            step = prev - mean_rev * (current_iv - prev) - np.random.normal(0, noise_std)
            history.append(float(np.clip(step, iv_low, iv_high)))

        history.reverse()  # chronological order — oldest first, current last
        logger.info(
            f"[TastytradeConnector] Seeded {len(history)}-day IV history for "
            f"{symbol}: anchor={current_iv:.2%}  IVR={iv_rank*100:.0f}  "
            f"range=[{iv_low:.2%}, {iv_high:.2%}]"
        )
        return history

    # ------------------------------------------------------------------ #
    # Option chain                                                          #
    # ------------------------------------------------------------------ #

    async def get_nested_chain(self, symbol: str) -> List[dict]:
        """
        Fetch GET /option-chains/{symbol}/nested and return a flat list of
        expiration dicts.  Each expiration has:
          expiration-date, days-to-expiration, strikes (list of strike dicts).

        Cached for 5 minutes.
        """
        self._assert_connected()
        now = time.monotonic()
        cached = self._chain_cache.get(symbol, {})
        if cached and now - cached.get("fetched_at", 0) < self._CHAIN_TTL:
            return cached["expirations"]

        resp = await self._client.get(f"/option-chains/{symbol}/nested")
        if resp.status_code != 200:
            logger.warning(
                f"option-chains/nested HTTP {resp.status_code} for {symbol}"
            )
            return []

        expirations: List[dict] = []
        for root in resp.json()["data"]["items"]:
            expirations.extend(root.get("expirations", []))

        self._chain_cache[symbol] = {"expirations": expirations, "fetched_at": now}
        logger.debug(
            f"[TastytradeConnector] Cached {len(expirations)} expirations for {symbol}"
        )
        return expirations

    async def get_options_chain(self, symbol: str, expiry: str = None) -> List[OptionsChain]:
        """
        Return OptionsChain objects for all strikes/expirations of `symbol`.
        If `expiry` (YYYY-MM-DD) is provided, filter to that date only.

        Note: real-time bid/ask/IV/OI per option requires DXLink streaming.
        These objects carry contract spec only; impliedVol is 0.0.
        """
        expirations = await self.get_nested_chain(symbol)
        result: List[OptionsChain] = []
        for exp in expirations:
            exp_date = exp.get("expiration-date", "")
            if expiry and exp_date != expiry:
                continue
            for strike_info in exp.get("strikes", []):
                K = _parse_float(strike_info.get("strike-price"), 0.0)
                if K <= 0:
                    continue
                for opt_type in ("CALL", "PUT"):
                    result.append(OptionsChain(
                        symbol=symbol,
                        expiry=exp_date,
                        strike=K,
                        option_type=opt_type,
                    ))
        return result

    async def get_dealer_gamma_skew(
        self, symbol: str, spot: float
    ) -> Dict[float, float]:
        """
        Estimate net dealer gamma by strike using Black-Scholes gamma.

        Model
        -----
        Gamma is computed at each listed strike using:
            d1 = (ln(S/K) + 0.5 σ² T) / (σ √T)
            Γ  = N'(d1) / (S σ √T)

        We use the ATM IV from market-metrics as a flat vol surface
        (no skew — acceptable for a first-order gamma profile).

        Dealer convention: market makers are assumed net SHORT gamma
        at near-ATM strikes.  We assign negative gamma to all near-expiry
        strikes within 5 % of spot (the region where dealers hedge most
        actively), positive elsewhere.

        Limitation: without per-strike open interest (requires DXLink WS),
        this gives the *shape* of the gamma surface but not its magnitude.
        OI weighting is the next planned improvement.

        Expirations used: DTE 1–60 only (shorter expirations dominate gamma).
        """
        self._assert_connected()

        metrics = await self.get_market_metrics(symbol)
        iv = metrics.get(symbol, {}).get("iv", 0.20)
        if iv <= 0:
            iv = 0.20

        expirations = await self.get_nested_chain(symbol)
        if not expirations or spot <= 0:
            return {}

        gamma_skew: Dict[float, float] = {}

        for exp in expirations:
            dte = int(exp.get("days-to-expiration", 0))
            if dte < 1 or dte > 60:
                continue
            T = dte / 365.0
            sqrt_T = math.sqrt(T)

            for strike_info in exp.get("strikes", []):
                K = _parse_float(strike_info.get("strike-price"), 0.0)
                if K <= 0:
                    continue
                try:
                    d1 = (math.log(spot / K) + 0.5 * iv ** 2 * T) / (iv * sqrt_T)
                    # Standard normal PDF
                    gamma_val = math.exp(-0.5 * d1 ** 2) / (
                        math.sqrt(2 * math.pi) * spot * iv * sqrt_T
                    )
                    # Dealers are net short gamma near ATM — negative sign
                    sign = -1.0 if abs(K - spot) / spot < 0.05 else 1.0
                    gamma_skew[K] = gamma_skew.get(K, 0.0) + sign * gamma_val
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue

        return gamma_skew

    # ------------------------------------------------------------------ #
    # Ticks / historical                                                    #
    # ------------------------------------------------------------------ #

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        """
        Real-time equity tick streaming requires the DXLink WebSocket streamer.
        Full implementation is planned.  For now the strategy loop polls via
        REST (get_market_metrics) on each candle event.
        """
        logger.info(
            f"[TastytradeConnector] Tick streaming for {symbol} not yet wired "
            "(DXLink WebSocket — planned). Using REST polling per candle."
        )

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        """Historical tick data is not yet implemented."""
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce a Tastytrade API value (str, Decimal, None) to float."""
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
