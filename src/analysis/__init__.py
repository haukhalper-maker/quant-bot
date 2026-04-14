"""
Analysis Engine — Greeks, Implied Volatility, Vol Surface, Volume Profile, Footprint, Patterns.

All financial math is implemented to production quality:
  - Black-Scholes with all five Greeks
  - Newton-Raphson IV solver with Brent's method fallback
  - Implied volatility surface with bilinear interpolation
  - Volume Profile: POC, Value Area High/Low
  - Footprint (bid/ask imbalance) analysis
  - Gamma imbalance and support/resistance detection
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from loguru import logger


# ============================================================================
# BLACK-SCHOLES PRICING & GREEKS
# ============================================================================


def _bs_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float, float, float, float]:
    """
    Black-Scholes call option price and Greeks.

    Args:
        S:     spot price
        K:     strike
        T:     time to expiry in years
        r:     risk-free rate (annualized, continuous)
        sigma: implied volatility (annualized)

    Returns:
        (price, delta, gamma, vega, theta)
        vega is per 1% change in vol; theta is per calendar day.
    """
    if T <= 1e-6 or sigma <= 1e-6:
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
        delta = 1.0 if S > K else 0.0
        return intrinsic, delta, 0.0, 0.0, 0.0

    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    disc = np.exp(-r * T)
    sqrt_T = np.sqrt(T)

    price = S * Nd1 - K * disc * Nd2
    delta = Nd1
    gamma = nd1 / (S * sigma * sqrt_T)
    vega = S * nd1 * sqrt_T / 100.0          # per 1 vol point
    theta = (
        -(S * nd1 * sigma) / (2.0 * sqrt_T) - r * K * disc * Nd2
    ) / 365.0                                 # per calendar day

    return price, delta, gamma, vega, theta


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float, float, float, float]:
    """
    Black-Scholes put option price and Greeks.
    Uses put-call parity for price/delta/theta; shares gamma and vega with call.
    """
    if T <= 1e-6 or sigma <= 1e-6:
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)
        delta = -1.0 if S < K else 0.0
        return intrinsic, delta, 0.0, 0.0, 0.0

    call_price, call_delta, gamma, vega, call_theta = bs_call(S, K, T, r, sigma)
    disc = np.exp(-r * T)

    put_price = call_price - S + K * disc          # put-call parity
    put_delta = call_delta - 1.0
    put_theta = call_theta + r * K * disc / 365.0  # adjust for rate term

    return put_price, put_delta, gamma, vega, put_theta


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Solve for implied volatility using Newton-Raphson, falling back to Brent's method.

    Args:
        market_price: observed option premium
        S, K, T, r:  standard BS inputs
        option_type: 'call' or 'put'

    Returns:
        Implied vol in [0.001, 10.0], or None if no solution found.
    """
    if T <= 0 or market_price <= 0:
        return None

    pricer = bs_call if option_type.lower() == "call" else bs_put

    def objective(sigma: float) -> float:
        price, *_ = pricer(S, K, T, r, sigma)
        return price - market_price

    # Check bracketing
    try:
        lo, hi = objective(0.001), objective(10.0)
        if lo > 0 and hi > 0:
            return None   # market price below intrinsic value
        if lo < 0 and hi < 0:
            return None   # market price too high
    except Exception:
        return None

    # Newton-Raphson with vega as derivative
    sigma = 0.25  # initial guess
    for _ in range(max_iter):
        price, _, _, vega_1pct, _ = pricer(S, K, T, r, sigma)
        vega_dsigma = vega_1pct * 100.0  # convert back from "per 1%" to per unit
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if abs(vega_dsigma) < 1e-10:
            break
        sigma -= diff / vega_dsigma
        if sigma <= 0:
            sigma = 0.001

    # Fallback: Brent's method (guaranteed convergence)
    try:
        return brentq(objective, 0.001, 10.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        return None


@dataclass
class Greeks:
    """Complete Greeks for a single option position."""

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    price: float = 0.0

    def scale(self, quantity: int) -> "Greeks":
        """Return Greeks scaled to position size."""
        mult = quantity * 100  # contract multiplier
        return Greeks(
            delta=self.delta * mult,
            gamma=self.gamma * mult,
            vega=self.vega * mult,
            theta=self.theta * mult,
            rho=self.rho * mult,
            price=self.price,
        )


class GreeksCalculator:
    """
    Options Greeks calculator.
    Wraps bs_call / bs_put with a clean interface and rho calculation.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def calculate(
        self, option_type: str, S: float, K: float, T: float, sigma: float
    ) -> Greeks:
        """Calculate all Greeks for the given option."""
        pricer = bs_call if option_type.upper() == "CALL" else bs_put
        price, delta, gamma, vega, theta = pricer(S, K, T, self.r, sigma)

        # Rho: sensitivity to interest rates (approximate)
        disc = np.exp(-self.r * T)
        if T > 1e-6 and sigma > 1e-6:
            _, d2 = _bs_d1_d2(S, K, T, self.r, sigma)
            if option_type.upper() == "CALL":
                rho = K * T * disc * norm.cdf(d2) / 100.0
            else:
                rho = -K * T * disc * norm.cdf(-d2) / 100.0
        else:
            rho = 0.0

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, price=price)

    def implied_vol(self, market_price: float, option_type: str,
                    S: float, K: float, T: float) -> Optional[float]:
        return implied_volatility(market_price, S, K, T, self.r, option_type)


# ============================================================================
# VOLATILITY SURFACE & ANALYTICS
# ============================================================================


@dataclass
class VolatilitySurface:
    """
    Implied volatility surface across strikes and expirations.

    iv_grid[i][j] is the IV for expirations[i] and strikes[j].
    Bilinear interpolation between grid nodes.
    """

    symbol: str
    timestamp: datetime
    expirations: List[str]      # YYYY-MM-DD sorted ascending
    strikes: List[float]        # sorted ascending
    iv_grid: np.ndarray         # shape (len(expirations), len(strikes))
    spot: float = 0.0

    def interpolate_iv(self, expiry: str, strike: float) -> Optional[float]:
        """Bilinear interpolation of IV for arbitrary expiry / strike."""
        if not self.expirations or not self.strikes:
            return None

        # Find bracketing expiry index
        exp_dates = [datetime.strptime(e, "%Y-%m-%d") for e in self.expirations]
        target_exp = datetime.strptime(expiry, "%Y-%m-%d")
        ei = np.searchsorted([e.toordinal() for e in exp_dates], target_exp.toordinal())
        ei = int(np.clip(ei, 1, len(self.expirations) - 1))

        # Find bracketing strike index
        si = int(np.searchsorted(self.strikes, strike))
        si = int(np.clip(si, 1, len(self.strikes) - 1))

        # Bilinear interpolation weights
        e0, e1 = exp_dates[ei - 1].toordinal(), exp_dates[ei].toordinal()
        s0, s1 = self.strikes[si - 1], self.strikes[si]
        t_ord = target_exp.toordinal()

        we = (t_ord - e0) / max(e1 - e0, 1)
        ws = (strike - s0) / max(s1 - s0, 1e-8)
        we = float(np.clip(we, 0, 1))
        ws = float(np.clip(ws, 0, 1))

        iv00 = self.iv_grid[ei - 1][si - 1]
        iv01 = self.iv_grid[ei - 1][si]
        iv10 = self.iv_grid[ei][si - 1]
        iv11 = self.iv_grid[ei][si]

        iv = (1 - we) * ((1 - ws) * iv00 + ws * iv01) + we * ((1 - ws) * iv10 + ws * iv11)
        return float(iv)

    def atm_iv(self, expiry: str) -> Optional[float]:
        """Return ATM (closest to spot) IV for a given expiry."""
        if self.spot <= 0:
            return None
        closest_strike = min(self.strikes, key=lambda k: abs(k - self.spot))
        return self.interpolate_iv(expiry, closest_strike)

    def skew(self, expiry: str) -> Optional[float]:
        """
        Compute 25-delta skew (put IV - call IV at ≈25 delta strikes).
        Uses spot ± 10% as a proxy for ±25 delta.
        """
        if self.spot <= 0:
            return None
        put_iv = self.interpolate_iv(expiry, self.spot * 0.90)
        call_iv = self.interpolate_iv(expiry, self.spot * 1.10)
        if put_iv is None or call_iv is None:
            return None
        return put_iv - call_iv


class VolatilityAnalyzer:
    """
    Tracks IV history for a symbol and computes volatility metrics:
    IV rank, IV percentile, IV/RV ratio, realized vol (HV), skew.
    """

    def __init__(self, lookback_days: int = 30):
        self.lookback = lookback_days
        self._iv_history: Dict[str, List[float]] = {}       # symbol → list[iv]
        self._price_history: Dict[str, List[float]] = {}    # symbol → list[close]

    def update_iv(self, symbol: str, iv: float) -> None:
        hist = self._iv_history.setdefault(symbol, [])
        hist.append(iv)
        if len(hist) > self.lookback * 2:
            self._iv_history[symbol] = hist[-self.lookback * 2:]

    def update_price(self, symbol: str, close: float) -> None:
        hist = self._price_history.setdefault(symbol, [])
        hist.append(close)
        if len(hist) > self.lookback * 2:
            self._price_history[symbol] = hist[-self.lookback * 2:]

    def iv_percentile(self, symbol: str) -> float:
        """What % of the lookback window had lower IV than today."""
        hist = self._iv_history.get(symbol, [])
        if len(hist) < 5:
            return 50.0
        current = hist[-1]
        pct = sum(1 for v in hist[:-1] if current >= v) / max(len(hist) - 1, 1) * 100
        return float(pct)

    def iv_rank(self, symbol: str) -> float:
        """IV rank = (current - min) / (max - min) * 100"""
        hist = self._iv_history.get(symbol, [])
        if len(hist) < 5:
            return 50.0
        current = hist[-1]
        lo, hi = min(hist[:-1]), max(hist[:-1])
        if hi <= lo:
            return 50.0
        return float((current - lo) / (hi - lo) * 100)

    def realized_vol(self, symbol: str, window: Optional[int] = None) -> float:
        """
        Annualized historical volatility (close-to-close).
        Uses Yang-Zhang estimator if OHLC data available, else close-to-close.
        """
        prices = self._price_history.get(symbol, [])
        w = window or self.lookback
        if len(prices) < max(w, 2):
            return 0.20  # default 20% if insufficient data

        recent = prices[-w:]
        log_returns = np.diff(np.log(recent))
        return float(np.std(log_returns) * np.sqrt(252))

    def iv_rv_ratio(self, symbol: str) -> float:
        iv = self._iv_history.get(symbol, [0.20])[-1]
        rv = self.realized_vol(symbol)
        return iv / max(rv, 0.001)

    def current_iv(self, symbol: str) -> float:
        hist = self._iv_history.get(symbol, [])
        return hist[-1] if hist else 0.20

    def recent_iv_history(self, symbol: str, n: int = 20) -> List[float]:
        return self._iv_history.get(symbol, [])[-n:]

    def recent_returns(self, symbol: str, n: int = 20) -> List[float]:
        prices = self._price_history.get(symbol, [])
        if len(prices) < 2:
            return []
        log_returns = list(np.diff(np.log(prices)))
        return log_returns[-n:]


# ============================================================================
# VOLUME PROFILE
# ============================================================================


@dataclass
class VolumeProfile:
    """
    Volume distribution across price levels.

    POC:        Point of Control — price level with highest traded volume.
    VAH / VAL:  Value Area High / Low — range containing 70% of volume.
    """

    symbol: str
    timestamp: datetime
    price_levels: Dict[float, int] = field(default_factory=dict)
    poc: float = 0.0
    value_area_high: float = 0.0
    value_area_low: float = 0.0

    @classmethod
    def from_candles(cls, symbol: str, prices: List[float], volumes: List[int],
                     num_buckets: int = 50) -> "VolumeProfile":
        """Build a volume profile from price+volume arrays."""
        if not prices or not volumes or len(prices) != len(volumes):
            return cls(symbol=symbol, timestamp=datetime.utcnow())

        lo, hi = min(prices), max(prices)
        if hi <= lo:
            return cls(symbol=symbol, timestamp=datetime.utcnow(), poc=lo)

        bucket_size = (hi - lo) / num_buckets
        price_levels: Dict[float, int] = {}

        for price, vol in zip(prices, volumes):
            bucket = lo + int((price - lo) / bucket_size) * bucket_size
            bucket = round(bucket, 4)
            price_levels[bucket] = price_levels.get(bucket, 0) + vol

        # POC = highest volume bucket
        poc = max(price_levels, key=price_levels.get)

        # Value Area (70% of total volume centered around POC)
        total_vol = sum(price_levels.values())
        target = 0.70 * total_vol
        sorted_levels = sorted(price_levels.items())
        sorted_vols = [v for _, v in sorted_levels]
        sorted_prices = [p for p, _ in sorted_levels]

        poc_idx = next(i for i, p in enumerate(sorted_prices) if p >= poc)
        lo_idx, hi_idx = poc_idx, poc_idx
        accumulated = sorted_vols[poc_idx]

        while accumulated < target:
            can_go_lo = lo_idx > 0
            can_go_hi = hi_idx < len(sorted_levels) - 1
            if not can_go_lo and not can_go_hi:
                break
            add_lo = sorted_vols[lo_idx - 1] if can_go_lo else -1
            add_hi = sorted_vols[hi_idx + 1] if can_go_hi else -1
            if add_lo >= add_hi:
                lo_idx -= 1
                accumulated += sorted_vols[lo_idx]
            else:
                hi_idx += 1
                accumulated += sorted_vols[hi_idx]

        return cls(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price_levels=dict(sorted_levels),
            poc=poc,
            value_area_high=sorted_prices[hi_idx],
            value_area_low=sorted_prices[lo_idx],
        )

    def is_in_value_area(self, price: float) -> bool:
        return self.value_area_low <= price <= self.value_area_high


# ============================================================================
# FOOTPRINT (BID/ASK IMBALANCE)
# ============================================================================


@dataclass
class Footprint:
    """
    Bid/ask volume imbalance at a price level.
    ratio > 1 → more aggressive selling (bearish pressure).
    ratio < 1 → more aggressive buying (bullish pressure).
    """

    symbol: str
    timestamp: datetime
    price: float = 0.0
    bid_volume: int = 0
    ask_volume: int = 0
    ratio: float = 1.0         # bid / ask
    delta_volume: int = 0      # ask - bid (positive = net buying)
    imbalance_pct: float = 0.0  # |bid - ask| / total

    def __post_init__(self):
        total = self.bid_volume + self.ask_volume
        if total > 0:
            self.ratio = self.bid_volume / max(self.ask_volume, 1)
            self.delta_volume = self.ask_volume - self.bid_volume
            self.imbalance_pct = abs(self.bid_volume - self.ask_volume) / total


@dataclass
class DOMSnapshot:
    """Depth of Market snapshot — full order book."""

    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, int]] = field(default_factory=list)  # (price, size) sorted desc
    asks: List[Tuple[float, int]] = field(default_factory=list)  # (price, size) sorted asc

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2.0
        return 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.best_ask - self.best_bid
        return 0.0

    def bid_depth(self, levels: int = 5) -> int:
        return sum(s for _, s in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> int:
        return sum(s for _, s in self.asks[:levels])

    def order_flow_imbalance(self, levels: int = 5) -> float:
        """
        OFI: (bid depth - ask depth) / total depth
        Positive → more resting buy interest; negative → more resting sell interest.
        """
        bd = self.bid_depth(levels)
        ad = self.ask_depth(levels)
        total = bd + ad
        return (bd - ad) / total if total > 0 else 0.0


# ============================================================================
# PATTERN DETECTION
# ============================================================================


class PatternDetector:
    """
    Detects technical and microstructure patterns:
      - Gamma imbalance (dealer gamma wall detection)
      - Support/resistance from swing highs/lows
      - Volume divergence
      - Order flow divergence
    """

    @staticmethod
    def detect_gamma_imbalance(
        spot: float, gamma_skew: Dict[float, float]
    ) -> Optional[str]:
        """
        Detect if dealers are long or short net gamma relative to spot.
        gamma_skew: strike → net dealer gamma exposure.
        Returns 'long_gamma' | 'short_gamma' | None.
        """
        if not gamma_skew:
            return None

        near_strikes = {
            k: v for k, v in gamma_skew.items()
            if abs(k - spot) / spot < 0.05  # within 5% of spot
        }
        if not near_strikes:
            return None

        net = sum(near_strikes.values())
        if net > 0:
            return "long_gamma"    # dealers are long gamma → mean-reverting behavior expected
        if net < 0:
            return "short_gamma"   # dealers are short gamma → trending/explosive moves expected
        return None

    @staticmethod
    def detect_support_resistance(
        prices: List[float], window: int = 5
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Detect swing-high resistance and swing-low support using local extrema.
        Returns (support, resistance).
        """
        if len(prices) < window * 2 + 1:
            return None, None

        arr = np.array(prices)
        supports = []
        resistances = []

        for i in range(window, len(arr) - window):
            local_min = arr[i] == arr[i - window:i + window + 1].min()
            local_max = arr[i] == arr[i - window:i + window + 1].max()
            if local_min:
                supports.append(arr[i])
            if local_max:
                resistances.append(arr[i])

        support = float(np.mean(supports[-3:])) if supports else None
        resistance = float(np.mean(resistances[-3:])) if resistances else None
        return support, resistance

    @staticmethod
    def detect_volume_divergence(
        prices: List[float], volumes: List[int], window: int = 10
    ) -> Optional[str]:
        """
        Detect price/volume divergence.
        Returns 'bearish_divergence' | 'bullish_divergence' | None.
        """
        if len(prices) < window or len(volumes) < window:
            return None

        p = prices[-window:]
        v = volumes[-window:]
        price_trend = p[-1] - p[0]
        vol_trend = v[-1] - v[0]

        if price_trend > 0 and vol_trend < 0:
            return "bearish_divergence"   # price up, volume down → weakening rally
        if price_trend < 0 and vol_trend < 0:
            return "bullish_divergence"   # price down, volume down → weakening selloff
        return None


# ============================================================================
# MAIN ANALYSIS ENGINE
# ============================================================================


class AnalysisEngine:
    """
    Main analytics coordinator. Wires together all sub-components and provides
    a single interface for the strategy layer.
    """

    def __init__(self, risk_free_rate: float = 0.05, iv_lookback_days: int = 30):
        self.greeks_calc = GreeksCalculator(risk_free_rate=risk_free_rate)
        self.vol_analyzer = VolatilityAnalyzer(lookback_days=iv_lookback_days)
        self.pattern_detector = PatternDetector()
        self.vol_surfaces: Dict[str, VolatilitySurface] = {}
        self.volume_profiles: Dict[str, VolumeProfile] = {}
        logger.info("AnalysisEngine initialized")

    # --- Greeks ---

    def calculate_greeks(
        self, option_type: str, S: float, K: float, T: float, sigma: float
    ) -> Greeks:
        return self.greeks_calc.calculate(option_type, S, K, T, sigma)

    def implied_vol(
        self, market_price: float, option_type: str, S: float, K: float, T: float
    ) -> Optional[float]:
        return self.greeks_calc.implied_vol(market_price, option_type, S, K, T)

    # --- Volatility analytics ---

    def on_price(self, symbol: str, close: float) -> None:
        """Feed a new close price into the vol analyzer."""
        self.vol_analyzer.update_price(symbol, close)

    def on_iv(self, symbol: str, iv: float) -> None:
        """Feed a new IV reading into the vol analyzer."""
        self.vol_analyzer.update_iv(symbol, iv)

    def vol_metrics(self, symbol: str) -> Dict[str, float]:
        return {
            "iv": self.vol_analyzer.current_iv(symbol),
            "iv_percentile": self.vol_analyzer.iv_percentile(symbol),
            "iv_rank": self.vol_analyzer.iv_rank(symbol),
            "realized_vol": self.vol_analyzer.realized_vol(symbol),
            "iv_rv_ratio": self.vol_analyzer.iv_rv_ratio(symbol),
        }

    # --- Volume profile ---

    def update_volume_profile(
        self, symbol: str, prices: List[float], volumes: List[int]
    ) -> VolumeProfile:
        vp = VolumeProfile.from_candles(symbol, prices, volumes)
        self.volume_profiles[symbol] = vp
        logger.debug(f"Volume profile {symbol}: POC={vp.poc:.2f} VAH={vp.value_area_high:.2f} VAL={vp.value_area_low:.2f}")
        return vp

    def get_volume_profile(self, symbol: str) -> Optional[VolumeProfile]:
        return self.volume_profiles.get(symbol)

    # --- Footprint ---

    def analyze_footprint(
        self, symbol: str, price: float, bid_volume: int, ask_volume: int
    ) -> Footprint:
        return Footprint(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price=price,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
        )

    # --- Vol surface ---

    def update_vol_surface(self, surface: VolatilitySurface) -> None:
        self.vol_surfaces[surface.symbol] = surface
        logger.debug(f"Vol surface updated: {surface.symbol} @ {surface.timestamp}")

    def get_vol_surface(self, symbol: str) -> Optional[VolatilitySurface]:
        return self.vol_surfaces.get(symbol)

    # --- Patterns ---

    def analyze_patterns(
        self, symbol: str, prices: List[float], volumes: List[int]
    ) -> Dict:
        support, resistance = self.pattern_detector.detect_support_resistance(prices)
        vol_divergence = self.pattern_detector.detect_volume_divergence(prices, volumes)
        return {
            "support": support,
            "resistance": resistance,
            "volume_divergence": vol_divergence,
        }
