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

import asyncio
import json
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


@dataclass
class PolygonOptionsContract:
    """Single options contract from Polygon v3 snapshot API."""
    ticker: str           # e.g. O:SPY240419C00530000
    symbol: str           # underlying, e.g. SPY
    strike: float
    expiry: str           # YYYY-MM-DD
    contract_type: str    # 'call' | 'put'
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    open_interest: int = 0
    volume: int = 0
    implied_vol: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    vwap: float = 0.0
    dte: int = 0
    underlying_price: float = 0.0


class PolygonConnector(DataConnector):
    """
    Polygon.io (Massive) — full REST + WebSocket implementation.

    Options snapshot  GET /v3/snapshot/options/{underlyingAsset}
      Returns per-contract greeks, OI, volume, IV, bid/ask.
      Auto-paginates (250 per page). Cached 60 seconds.

    Historical bars   GET /v2/aggs/ticker/{sym}/range/{mult}/{ts}/{from}/{to}
      1-minute OHLCV for intraday replay and backtest.

    WebSocket         wss://socket.polygon.io/stocks
      Real-time per-second aggregates for live price feed.
    """

    _BASE = "https://api.polygon.io"
    _WS_URL = "wss://socket.polygon.io/stocks"
    _CHAIN_TTL = 60.0   # seconds

    def __init__(self, api_key: str = None):
        super().__init__("PolygonConnector")
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self._client: Optional[httpx.AsyncClient] = None
        self._chain_cache: Dict[str, dict] = {}
        self._tick_callbacks: Dict[str, List] = {}
        self._ws_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY must be set in .env")
        self._client = httpx.AsyncClient(timeout=30.0)
        for attempt in range(4):
            resp = await self._client.get(
                f"{self._BASE}/v2/aggs/ticker/SPY/prev",
                params={"apiKey": self.api_key},
            )
            if resp.status_code == 200:
                break
            if resp.status_code == 429 and attempt < 3:
                wait = 15 * (attempt + 1)
                logger.warning(f"[Polygon] Rate limited on connect — retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            await self._client.aclose()
            raise RuntimeError(
                f"Polygon key validation failed: HTTP {resp.status_code}"
            )
        self.is_connected = True
        logger.info(f"[Polygon] Connected  key=...{self.api_key[-6:]}")
        return True

    async def disconnect(self) -> None:
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
        if self._client:
            await self._client.aclose()
        self.is_connected = False
        logger.info("[Polygon] Disconnected")

    # ------------------------------------------------------------------ #
    # Options chain snapshot                                               #
    # ------------------------------------------------------------------ #

    async def get_options_snapshot(
        self,
        symbol: str,
        max_dte: int = 7,
    ) -> List[PolygonOptionsContract]:
        """
        Full options chain from Polygon v3 snapshot.
        Fetches all pages, filters to DTE 0–max_dte.
        Greeks are provided by Polygon — no BSM needed here.
        """
        now = time.monotonic()
        cached = self._chain_cache.get(symbol, {})
        if cached and now - cached.get("fetched_at", 0) < self._CHAIN_TTL:
            return cached["contracts"]

        today = datetime.utcnow().date()
        contracts: List[PolygonOptionsContract] = []
        url = f"{self._BASE}/v3/snapshot/options/{symbol}"
        params: dict = {"apiKey": self.api_key, "limit": 250}

        while url:
            try:
                resp = await self._client.get(url, params=params)
                if resp.status_code != 200:
                    logger.warning(
                        f"[Polygon] snapshot HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    break
                body = resp.json()
            except Exception as exc:
                logger.warning(f"[Polygon] snapshot error: {exc}")
                break

            for r in body.get("results", []):
                details = r.get("details") or {}
                greeks  = r.get("greeks")  or {}
                day     = r.get("day")     or {}
                lq      = r.get("last_quote") or {}

                expiry_str = details.get("expiration_date", "")
                if not expiry_str:
                    continue
                try:
                    exp_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                    dte = (exp_dt - today).days
                except ValueError:
                    continue
                if dte < 0 or dte > max_dte:
                    continue

                bid = float(lq.get("bid") or r.get("bid") or 0)
                ask = float(lq.get("ask") or r.get("ask") or 0)
                mid_p = (bid + ask) / 2 if (bid + ask) > 0 else 0.0

                contracts.append(PolygonOptionsContract(
                    ticker=r.get("ticker", ""),
                    symbol=symbol,
                    strike=float(details.get("strike_price") or 0),
                    expiry=expiry_str,
                    contract_type=(details.get("contract_type") or "call").lower(),
                    bid=bid,
                    ask=ask,
                    mid=mid_p,
                    open_interest=int(r.get("open_interest") or 0),
                    volume=int(day.get("volume") or 0),
                    implied_vol=float(r.get("implied_volatility") or 0),
                    delta=float(greeks.get("delta") or 0),
                    gamma=float(greeks.get("gamma") or 0),
                    theta=float(greeks.get("theta") or 0),
                    vega=float(greeks.get("vega") or 0),
                    vwap=float(day.get("vwap") or 0),
                    dte=dte,
                    underlying_price=float(
                        (r.get("underlying_asset") or {}).get("price") or 0
                    ),
                ))

            next_url = body.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": self.api_key}
            else:
                break

        self._chain_cache[symbol] = {"contracts": contracts, "fetched_at": now}
        logger.debug(
            f"[Polygon] {symbol}: {len(contracts)} contracts (DTE 0-{max_dte})"
        )
        return contracts

    async def get_bars(
        self,
        symbol: str,
        date: str,
        multiplier: int = 1,
        timespan: str = "minute",
    ) -> List[Candle]:
        """Historical intraday bars. date = 'YYYY-MM-DD'."""
        url = (
            f"{self._BASE}/v2/aggs/ticker/{symbol}/range/"
            f"{multiplier}/{timespan}/{date}/{date}"
        )
        try:
            resp = await self._client.get(
                url,
                params={
                    "apiKey": self.api_key,
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": 50000,
                },
            )
            if resp.status_code != 200:
                logger.warning(f"[Polygon] bars HTTP {resp.status_code}")
                return []
        except Exception as exc:
            logger.warning(f"[Polygon] bars error: {exc}")
            return []

        try:
            results = resp.json().get("results", [])
        except Exception as exc:
            logger.warning(f"[Polygon] bars JSON parse error (likely truncated 429 body): {exc}")
            return []

        candles = []
        for r in results:
            ts = datetime.utcfromtimestamp(r["t"] / 1000)
            candles.append(Candle(
                symbol=symbol,
                timestamp=ts,
                open=float(r["o"]),
                high=float(r["h"]),
                low=float(r["l"]),
                close=float(r["c"]),
                volume=int(r.get("v", 0)),
                timeframe=f"{multiplier}{timespan[0]}",
            ))
        return candles

    # ------------------------------------------------------------------ #
    # WebSocket — real-time per-second aggregates                          #
    # ------------------------------------------------------------------ #

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        self._tick_callbacks.setdefault(symbol, []).append(callback)
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_run())

    async def _ws_run(self) -> None:
        import websockets as _ws
        backoff = 1.0
        while True:
            try:
                async with _ws.connect(self._WS_URL) as ws:
                    backoff = 1.0
                    await ws.send(json.dumps(
                        {"action": "auth", "params": self.api_key}
                    ))
                    for sym in self._tick_callbacks:
                        await ws.send(json.dumps(
                            {"action": "subscribe", "params": f"A.{sym}"}
                        ))
                    async for raw in ws:
                        for ev in json.loads(raw):
                            if ev.get("ev") != "A":
                                continue
                            sym = ev.get("sym", "")
                            if sym not in self._tick_callbacks:
                                continue
                            tick = Tick(
                                symbol=sym,
                                timestamp=datetime.utcfromtimestamp(
                                    ev.get("e", 0) / 1000
                                ),
                                price=float(ev.get("c") or ev.get("vw") or 0),
                                size=int(ev.get("av", 0)),
                            )
                            for cb in self._tick_callbacks[sym]:
                                asyncio.create_task(cb(tick))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    f"[Polygon] WS error: {exc}. Reconnecting in {backoff:.0f}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    # ------------------------------------------------------------------ #
    # DataConnector interface                                               #
    # ------------------------------------------------------------------ #

    async def get_historical_ticks(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Tick]:
        return []

    async def get_options_chain(
        self, symbol: str, expiry: str = None
    ) -> List[OptionsChain]:
        contracts = await self.get_options_snapshot(symbol)
        result = []
        for c in contracts:
            if expiry and c.expiry != expiry:
                continue
            result.append(OptionsChain(
                symbol=c.symbol,
                expiry=c.expiry,
                strike=c.strike,
                option_type=c.contract_type.upper(),
                bid=c.bid,
                ask=c.ask,
                open_interest=c.open_interest,
                volume=c.volume,
                impliedVol=c.implied_vol,
            ))
        return result


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

    _BASE_URL      = "https://api.tastytrade.com"
    _BASE_URL_CERT = "https://api.cert.tastyworks.com"   # paper/sandbox
    _METRICS_TTL = 60.0
    _CHAIN_TTL   = 300.0

    def __init__(self, username: str = None, password: str = None,
                 api_token: str = None, cert: bool = False):
        super().__init__("TastytradeConnector")
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.username      = username  or os.getenv("TASTYTRADE_USERNAME", "")
        self.password      = password  or os.getenv("TASTYTRADE_PASSWORD", "")
        # OAuth credentials from developer.tastytrade.com
        self.client_id     = os.getenv("TASTYTRADE_CLIENT_ID", "")
        self.client_secret = os.getenv("TASTYTRADE_CLIENT_SECRET", "")
        self.grant_token   = os.getenv("TASTYTRADE_GRANT_TOKEN", "")
        # Legacy: pre-issued raw session token (still supported)
        self.api_token     = api_token or os.getenv("TASTYTRADE_API_TOKEN", "")
        self._base         = self._BASE_URL_CERT if cert else self._BASE_URL
        self._oauth_url    = f"{self._BASE_URL}/oauth/token"  # identity server is always live
        self.cert          = cert
        self._session_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._challenge_token: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._accounts: List[dict] = []
        self._metrics_cache: Dict[str, dict] = {}
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
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

        # ── Auth priority ──────────────────────────────────────────────────
        # 1. OAuth (grant_token + client_id + client_secret) — preferred
        # 2. Raw session token (TASTYTRADE_API_TOKEN) — legacy
        # 3. Username + password — fallback

        if self.grant_token and self.client_id and self.client_secret:
            result = await self._oauth_connect()
            if result:
                return result
            # OAuth failed — fall through to username/password

        if self.api_token:
            self._session_token = self.api_token
            self._client.headers.update({"Authorization": self.api_token})
            logger.info("Tastytrade: using raw API token")
            return await self._finish_connect("(api-token)")

        if not self.username or not self.password:
            raise ValueError(
                "Set TASTYTRADE_GRANT_TOKEN + TASTYTRADE_CLIENT_ID + TASTYTRADE_CLIENT_SECRET "
                "(or TASTYTRADE_USERNAME + TASTYTRADE_PASSWORD) in .env"
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
        self._challenge_token = None
        payload = resp.json()["data"]
        self._session_token = payload["session-token"]
        self._client.headers.update({"Authorization": self._session_token})
        return await self._finish_connect(payload.get("user", {}).get("username", self.username))

    async def _oauth_connect(self) -> bool:
        """
        Exchange grant token for an access token using OAuth 2.0
        client_credentials + grant_token flow.

        Tastytrade OAuth endpoint: https://id.tastytrade.com/oauth/token
        Body (application/x-www-form-urlencoded):
            grant_type=authorization_code
            code=<TASTYTRADE_GRANT_TOKEN>
            client_id=<TASTYTRADE_CLIENT_ID>
            client_secret=<TASTYTRADE_CLIENT_SECRET>
            redirect_uri=https://localhost  (required by Tastytrade OAuth)

        The access_token returned is used as the Authorization header.
        The refresh_token is stored so we can silently renew when it expires
        (access tokens are typically valid for 24h).
        """
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=15.0) as oauth_client:
            resp = await oauth_client.post(
                self._oauth_url,
                data={
                    "grant_type":    "authorization_code",
                    "code":          self.grant_token,
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri":  "http://localhost:8000/",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if resp.status_code != 200:
            logger.warning(
                f"Tastytrade OAuth failed ({resp.status_code}), falling back to username/password"
            )
            return False  # caller will try next auth method

        data = resp.json()
        access_token          = data["access_token"]
        self._refresh_token   = data.get("refresh_token", "")
        self._session_token   = access_token
        self._client.headers.update({"Authorization": f"Bearer {access_token}"})
        logger.info(
            f"Tastytrade OAuth OK  env={'cert' if self.cert else 'live'}  "
            f"expires_in={data.get('expires_in', '?')}s"
        )
        return await self._finish_connect("(oauth)")

    async def _refresh_access_token(self) -> bool:
        """Use the refresh token to get a new access token without re-auth."""
        if not self._refresh_token:
            return False
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=15.0) as c:
            resp = await c.post(
                self._oauth_url,
                data={
                    "grant_type":    "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if resp.status_code != 200:
            logger.warning(f"Tastytrade token refresh failed: {resp.status_code}")
            return False
        data = resp.json()
        self._session_token = data["access_token"]
        self._refresh_token = data.get("refresh_token", self._refresh_token)
        self._client.headers.update({"Authorization": f"Bearer {self._session_token}"})
        logger.info("Tastytrade access token refreshed")
        return True

    async def _finish_connect(self, username_display: str = "") -> bool:
        """Shared post-auth step: fetch accounts and mark connected."""
        acc_resp = await self._client.get("/customers/me/accounts")
        if acc_resp.status_code == 200:
            items = acc_resp.json()["data"]["items"]
            self._accounts = [item["account"] for item in items]
        self.is_connected = True
        logger.info(
            f"Tastytrade connected — user={username_display!r}  "
            f"env={'cert' if self.cert else 'live'}  accounts={len(self._accounts)}"
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
                        # Already a decimal fraction (e.g. "0.1838" = 18.38%)
                        "iv":      _parse_float(item.get("implied-volatility-index"), 0.20),
                        # 0-1 fraction from API (e.g. "0.3725" = 37.25 rank) → ×100 for 0-100 scale
                        "iv_rank": _parse_float(item.get("implied-volatility-index-rank"), 0.50) * 100,
                        "iv_pct":  _parse_float(item.get("implied-volatility-percentile"), 0.50) * 100,
                        # Percentage-point strings from API (e.g. "8.7" = 8.7%) → ÷100 for decimal
                        "hv30":    _parse_float(item.get("historical-volatility-30-day"), 15.0) / 100,
                        "iv30":    _parse_float(item.get("implied-volatility-30-day"), 20.0) / 100,
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
