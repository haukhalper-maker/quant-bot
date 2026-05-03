"""
ZeroDTE Backtester — real Polygon bars, synthetic GEX + IV from VIX.

Pipeline per day:
  1. Pull real SPY 1-min bars from Polygon (rate-limited, 0.4s between calls)
  2. Pull VIX daily closes → IV + regime proxy
  3. Every 5-min bar: synthesize GEX walls + DTEAnalysis from BSM + VIX
  4. Run ZeroDTEStrategy.on_bar() → first signal is the trade for that day
  5. P&L = actual BSM entry premium vs intrinsic value at EOD close

Synthetic GEX regime (calibrated to real VIX/GEX correlation):
  VIX < 16   → strong positive GEX  (pinning — dealers long gamma)
  VIX 16-20  → mild positive GEX    (slight pin)
  VIX 20-23  → near-neutral         (transition zone)
  VIX 23-28  → negative GEX         (explosive — dealers short gamma)
  VIX > 28   → strong negative GEX  (crisis/crash mode)
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

import httpx
import numpy as np
from loguru import logger
from scipy.stats import norm as _norm

from src.analysis import GammaWall, DTEAnalysis
from src.market_data import Candle


# ── BSM helpers ──────────────────────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def bsm_call(S: float, K: float, T: float, r: float = 0.05, sigma: float = 0.20) -> float:
    if T <= 1e-6:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return max(S * _norm.cdf(d1) - K * math.exp(-r * T) * _norm.cdf(d2), 0.0)

def bsm_put(S: float, K: float, T: float, r: float = 0.05, sigma: float = 0.20) -> float:
    c = bsm_call(S, K, T, r, sigma)
    return max(c - S + K * math.exp(-r * T), 0.0)

def bsm_gamma(S: float, K: float, T: float, r: float = 0.05, sigma: float = 0.20) -> float:
    if T <= 1e-6 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return _norm.pdf(d1) / (S * sigma * math.sqrt(T))


# ── Synthetic market builder ──────────────────────────────────────────────────

class SyntheticMarketBuilder:
    R = 0.05

    # VIX → net GEX bias: negative = explosive, positive = pinning
    # Calibrated: real GEX turns negative around VIX 20-22 (SPX options dealers)
    _VIX_GEX_BIAS = [
        (16,  1.00),   # VIX < 16 → strong pin
        (20,  0.40),   # VIX 16-20 → mild pin
        (23,  -0.15),  # VIX 20-23 → slight explosive
        (28,  -0.65),  # VIX 23-28 → explosive
        (999, -1.00),  # VIX > 28 → strong explosive
    ]

    def _gex_bias(self, vix: float) -> float:
        for threshold, bias in self._VIX_GEX_BIAS:
            if vix < threshold:
                return bias
        return -1.0

    def _put_oi_mult(self, vix: float) -> float:
        # Put/call OI skew: more puts bought as VIX rises (protective positioning)
        # VIX 15 → ~0.85×  (call-heavy, bullish positioning)
        # VIX 20 → ~1.00×  (balanced)
        # VIX 30 → ~1.30×  (put-heavy, defensive)
        return 0.85 + (vix - 15) * 0.030

    def build_walls(
        self,
        spot: float,
        vix: float,
        price_history: List[float],
        trade_date: str,
    ) -> List[GammaWall]:
        iv      = vix / 100.0
        T_gamma = 1.0 / 365.0
        bias    = self._gex_bias(vix)
        put_mult = self._put_oi_mult(vix)

        # Real price momentum shifts GEX further
        momentum = 0.0
        if len(price_history) >= 10:
            momentum = (price_history[-1] - price_history[-10]) / price_history[-10]
        # Strong trend amplifies explosive regime
        if abs(momentum) > 0.003:
            bias -= np.sign(momentum) * 0.3

        walls: List[GammaWall] = []
        for offset in range(-6, 7):
            K = round(spot) + offset
            dist_pct = abs(K - spot) / spot
            if dist_pct > 0.05:
                continue

            g = bsm_gamma(spot, float(K), T_gamma, self.R, iv)
            if g <= 0:
                continue

            # OI: gaussian bell centered at ATM
            oi_peak = 80_000
            sigma_oi = 2.5
            oi_base  = max(int(oi_peak * math.exp(-0.5 * ((K - spot) / sigma_oi) ** 2)), 500)

            call_oi = int(oi_base * 0.50)
            put_oi  = int(oi_base * 0.50 * put_mult)

            call_gex = call_oi * g * 100 * spot
            put_gex  = put_oi  * g * 100 * spot

            # Apply bias: positive bias → boost calls (pin), negative → boost puts (explosive)
            if bias > 0:
                call_gex *= (1.0 + 0.6 * bias)
            else:
                put_gex  *= (1.0 - 0.6 * bias)  # bias is negative, so this increases put_gex

            net_gex = call_gex - put_gex

            proximity   = math.exp(-0.5 * ((K - spot) / 2.0) ** 2)
            gex_s       = min(abs(net_gex) / 2_000_000, 1.0)
            vol_signal  = min(abs(vix - 18) / 20.0, 1.0)
            confluence  = 0.40 * gex_s + 0.35 * proximity + 0.25 * vol_signal

            walls.append(GammaWall(
                strike=float(K),
                net_gex_dollars=net_gex,
                call_gex=call_gex,
                put_gex=put_gex,
                total_oi=call_oi + put_oi,
                total_volume=int((call_oi + put_oi) * 0.15),
                call_volume=int(call_oi * 0.15),
                put_volume=int(put_oi * 0.15),
                confluence_score=float(np.clip(confluence, 0.0, 1.0)),
                distance_pct=dist_pct,
                wall_type="pin" if net_gex >= 0 else "explosive",
            ))

        walls.sort(key=lambda w: abs(w.net_gex_dollars) * w.confluence_score, reverse=True)
        return walls[:6]

    def build_term(self, spot: float, vix: float, trade_date: str) -> List[DTEAnalysis]:
        iv_base = vix / 100.0
        # 0DTE options trade at ~15% IV premium over 30-day VIX
        iv_mults    = {0: 1.15, 1: 1.07, 2: 1.00}
        next_mults  = {0: 1.07, 1: 1.00, 2: 0.95}
        analyses    = []
        for dte in [0, 1, 2]:
            atm_iv      = iv_base * iv_mults[dte]
            next_iv     = iv_base * next_mults[dte]
            T           = max(dte, 0.25) / 365.0
            expected_mv = atm_iv * spot * math.sqrt(T)
            iv_premium  = atm_iv / max(next_iv, 0.01)
            crush_prob  = float(np.clip((iv_premium - 1.0) * 3.0, 0.0, 1.0))
            expiry_dt   = datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=dte)
            analyses.append(DTEAnalysis(
                dte=dte,
                expiry=expiry_dt.strftime("%Y-%m-%d"),
                atm_iv=atm_iv,
                expected_move_1sd=expected_mv,
                iv_premium_vs_next=iv_premium,
                crush_probability=crush_prob,
                kelly_sell_fraction=0.05,
            ))
        return analyses


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    date: str
    entry_time: str
    signal_type: str
    play_type: str
    strike: float
    entry_spot: float
    exit_spot: float
    entry_premium: float   # per share
    exit_value: float      # per share at expiry
    n_contracts: int
    pnl: float
    score: float
    vix: float
    time_regime: str

    @property
    def win(self) -> bool:
        return self.pnl > 0

    @property
    def cost_basis(self) -> float:
        return abs(self.entry_premium) * 100 * self.n_contracts

    @property
    def return_pct(self) -> float:
        b = self.cost_basis
        return (self.pnl / b * 100) if b > 0 else 0.0


# ── Backtest result ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    symbol: str
    start: str
    end: str
    initial_capital: float
    trades: List[BacktestTrade] = field(default_factory=list)
    daily_pnl: Dict[str, float] = field(default_factory=dict)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.win) / len(self.trades)

    @property
    def sharpe(self) -> float:
        vals = list(self.daily_pnl.values())
        if len(vals) < 5:
            return 0.0
        mu, std = np.mean(vals), np.std(vals)
        return float((mu / std) * math.sqrt(252)) if std > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        equity, peak, max_dd = self.initial_capital, self.initial_capital, 0.0
        for d in sorted(self.daily_pnl):
            equity += self.daily_pnl[d]
            peak    = max(peak, equity)
            max_dd  = max(max_dd, (peak - equity) / peak)
        return max_dd

    @property
    def profit_factor(self) -> float:
        gw = sum(t.pnl for t in self.trades if t.win)
        gl = abs(sum(t.pnl for t in self.trades if not t.win))
        return (gw / gl) if gl > 0 else float("inf")


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    SAMPLE_EVERY = 5   # bars between strategy calls (5-min sampling)
    RATE_LIMIT_S = 0.4 # seconds between Polygon requests (stays under 5/min free tier)
    R = 0.05

    def __init__(self, polygon, initial_capital: float = 2500.0, defined_risk: bool = False):
        self._polygon = polygon
        self._capital = initial_capital
        self._defined_risk = defined_risk
        self._builder = SyntheticMarketBuilder()

    async def run(self, symbol: str, start: str, end: str) -> BacktestResult:
        from src.strategy import (ZeroDTEStrategy, ORBStrategy, VWAPStrategy,
                                   GapFillStrategy, MomentumStrategy, IVPercentileStrategy)

        result = BacktestResult(symbol=symbol, start=start, end=end,
                                initial_capital=self._capital)

        vix_by_date  = await self._fetch_vix(start, end)
        trading_days = self._date_range(start, end)
        logger.info(
            f"[Backtest] {symbol}  {start} -> {end}  "
            f"({len(trading_days)} days, VIX loaded for {len(vix_by_date)} days)"
        )

        strategies = {
            "ZeroDTE":  ZeroDTEStrategy(account_bp=self._capital, risk_pct=0.18,
                                        defined_risk=self._defined_risk),
            "ORB":      ORBStrategy(account_bp=self._capital, risk_pct=0.12),
            "VWAP":     VWAPStrategy(account_bp=self._capital, risk_pct=0.12),
            "GapFill":  GapFillStrategy(account_bp=self._capital, risk_pct=0.12),
            "Momentum": MomentumStrategy(account_bp=self._capital, risk_pct=0.12),
            "IVPct":    IVPercentileStrategy(account_bp=self._capital, risk_pct=0.10),
        }
        prior_close = 0.0

        for trade_date in trading_days:
            await asyncio.sleep(self.RATE_LIMIT_S)
            bars = await self._fetch_bars_with_retry(symbol, trade_date)
            if not bars:
                continue

            market_bars = [b for b in bars if self._is_market_hours(b.timestamp)]
            if len(market_bars) < 30:
                continue

            today_open = market_bars[0].open
            gap_pct    = (today_open - prior_close) / prior_close if prior_close > 0 else 0.0
            vix        = vix_by_date.get(trade_date) or self._estimate_vix(market_bars)
            ivr        = min(int(vix * 2.2), 99)  # rough IVR proxy from VIX level

            # Set daily context on all strategies
            s = strategies
            s["ZeroDTE"].set_daily_context(gap_pct=gap_pct, prior_close=prior_close, vix=vix)
            s["ORB"].set_daily_context(trade_date)
            s["VWAP"].set_daily_context(trade_date)
            s["GapFill"].set_daily_context(gap_pct, prior_close, trade_date)
            s["Momentum"].set_daily_context(trade_date)
            s["IVPct"].set_daily_context(ivr, trade_date)

            trade = await self._run_day(strategies, symbol, trade_date, market_bars, vix,
                                         gap_pct, prior_close, ivr)
            prior_close = market_bars[-1].close

            result.daily_pnl[trade_date] = trade.pnl if trade else 0.0
            if trade:
                result.trades.append(trade)
                logger.info(
                    f"  {trade_date}  VIX={vix:.1f}  {trade.play_type:10s}  "
                    f"{trade.signal_type:16s}  K={trade.strike:.0f}  "
                    f"entry=${trade.entry_spot:.1f}  exit=${trade.exit_spot:.1f}  "
                    f"n={trade.n_contracts}  P&L=${trade.pnl:+.0f}  "
                    f"({'WIN' if trade.win else 'LOS'})"
                )
            else:
                logger.info(f"  {trade_date}  VIX={vix:.1f}  no signal")

        return result

    async def _run_day(
        self,
        strategies: dict,
        symbol: str,
        trade_date: str,
        bars: List[Candle],
        vix: float,
        gap_pct: float = 0.0,
        prior_close: float = 0.0,
        ivr: float = 50.0,
    ) -> Optional[BacktestTrade]:
        # PIN-only mode: skip high-vol days entirely. VIX ≥ 28 means dealers are
        # short gamma and SPY won't pin — sell-strangle risk/reward breaks down.
        if vix >= 28:
            return None

        eod_spot = bars[-1].close
        iv_0dte  = (vix / 100.0) * 1.15

        for i in range(0, len(bars) - 1, self.SAMPLE_EVERY):
            bar  = bars[i]
            spot = bar.close

            if self._et_hour(bar.timestamp) < 10.0:
                continue

            zdte  = strategies["ZeroDTE"]
            walls = self._builder.build_walls(spot, vix, list(zdte._price_history), trade_date)
            term  = self._builder.build_term(spot, vix, trade_date)

            # Run all strategies, take highest-confidence signal
            candidates = []
            for name, strat in strategies.items():
                try:
                    if name == "ZeroDTE":
                        sig = await strat.on_bar(bar, walls, term, self._capital)
                    elif name in ("ORB", "VWAP", "Momentum"):
                        sig = await strat.on_bar(bar, walls, term, self._capital, trade_date)
                    elif name == "GapFill":
                        sig = await strat.on_bar(bar, walls, term, self._capital,
                                                  trade_date, gap_pct, prior_close)
                    else:
                        sig = await strat.on_bar(bar, walls, term, self._capital, trade_date, ivr)
                    if sig:
                        candidates.append((name, sig))
                except Exception:
                    pass

            if not candidates:
                continue

            strat_name, signal = max(candidates, key=lambda x: x[1].confidence)

            sig_name  = signal.signal_type.value

            # Drop low-conviction premium-buying signals
            if sig_name in ("buy_call", "buy_put", "straddle") and signal.confidence < 0.75:
                continue

            play_type = signal.metadata.get("play_type", strat_name.lower())
            strike    = float(signal.strike)
            n_raw     = max(1, int(signal.position_size))
            score     = float(signal.metadata.get("condition_score", signal.confidence))
            regime    = signal.metadata.get("time_regime", "?")

            # Time remaining in trading day (in years for BSM).
            # 6.5 trading hours/day × 252 days/year = 1638 trading hours/year.
            bar_et_h = self._et_hour(bar.timestamp)
            T_entry  = max((16.0 - bar_et_h) / 6.5 / 252.0, 0.25 / 365.0)

            # Compute actual BSM entry premium — this is what we ACTUALLY pay/collect
            entry_premium = self._entry_premium(sig_name, spot, strike, T_entry, iv_0dte,
                                                 signal.metadata)

            # Cap contracts so max loss ≤ 18% of capital per trade ($450 on $2500).
            # Iron condor max loss = wing_width - net_credit (defined risk).
            max_risk     = self._capital * 0.18
            if sig_name == "straddle":
                max_loss_per = abs(entry_premium) * 100 * 0.75  # stop at 25% remaining
            elif sig_name == "iron_condor":
                wc = signal.metadata.get("wing_call_strike")
                wp = signal.metadata.get("wing_put_strike")
                ck = float(signal.metadata.get("call_strike", strike + 3))
                pk = float(signal.metadata.get("put_strike",  strike - 3))
                wing_width = (float(wc) - ck if wc else 5.0)
                max_loss_per = max(wing_width * 100 - abs(entry_premium) * 100, 10.0)
            else:
                max_loss_per = abs(entry_premium) * 100          # full premium at risk
            n = max(1, min(n_raw, int(max_risk / max_loss_per))) if max_loss_per > 0 else 1

            # Intraday stop-loss / take-profit.
            # buy_call / buy_put : stop at 20% remaining (after 30-bar min hold).
            # straddle           : take profit 2×; stop at 25% (after 30-bar hold).
            # sell_strangle (PIN): touch-stop when spot crosses either short strike;
            #                      take-profit when cost-to-close decays to 30% of entry.
            exit_spot = eod_spot
            early_exit_value: Optional[float] = None
            early_sell_pnl: Optional[float] = None   # sell-side uses inverse P&L formula
            if sig_name in ("buy_call", "buy_put"):
                stop_floor = entry_premium * 0.20
                _fn = bsm_call if sig_name == "buy_call" else bsm_put
                for j in range(i + 30 * self.SAMPLE_EVERY, len(bars) - 1, self.SAMPLE_EVERY):
                    nb = bars[j]
                    T_now = max((16.0 - self._et_hour(nb.timestamp)) / 6.5 / 252.0, 0.5 / 365.0)
                    cv = _fn(nb.close, strike, T_now, self.R, iv_0dte)
                    if cv <= stop_floor:
                        exit_spot = nb.close
                        early_exit_value = cv
                        break
            elif sig_name == "straddle":
                stop_floor  = entry_premium * 0.25
                take_profit = entry_premium * 2.0
                min_stop_j  = i + 30 * self.SAMPLE_EVERY
                for j in range(i + self.SAMPLE_EVERY, len(bars) - 1, self.SAMPLE_EVERY):
                    nb = bars[j]
                    T_now = max((16.0 - self._et_hour(nb.timestamp)) / 6.5 / 252.0, 0.5 / 365.0)
                    cv = bsm_call(nb.close, strike, T_now, self.R, iv_0dte) + \
                         bsm_put(nb.close, strike, T_now, self.R, iv_0dte)
                    if cv >= take_profit:
                        exit_spot = nb.close
                        early_exit_value = cv
                        break
                    if j >= min_stop_j and cv <= stop_floor:
                        exit_spot = nb.close
                        early_exit_value = cv
                        break
            elif sig_name in ("sell_strangle", "iron_condor"):
                _ck = float(signal.metadata.get("call_strike", strike + 3))
                _pk = float(signal.metadata.get("put_strike",  strike - 3))
                _wc = signal.metadata.get("wing_call_strike")
                _wp = signal.metadata.get("wing_put_strike")
                # Take-profit only: close when 60% of max profit is captured.
                tp_threshold = entry_premium * 0.40
                for j in range(i + self.SAMPLE_EVERY, len(bars) - 1, self.SAMPLE_EVERY):
                    nb = bars[j]
                    T_now = max((16.0 - self._et_hour(nb.timestamp)) / 6.5 / 252.0, 0.5 / 365.0)
                    cv = bsm_call(nb.close, _ck, T_now, self.R, iv_0dte) + \
                         bsm_put(nb.close, _pk, T_now, self.R, iv_0dte)
                    if _wc and _wp:
                        wing_cv = bsm_call(nb.close, float(_wc), T_now, self.R, iv_0dte) + \
                                  bsm_put(nb.close, float(_wp), T_now, self.R, iv_0dte)
                        cv = max(cv - wing_cv, 0.0)
                    if cv <= tp_threshold:
                        exit_spot = nb.close
                        early_sell_pnl = (entry_premium - cv) * 100 * n
                        break

            if early_sell_pnl is not None:
                pnl = early_sell_pnl
                exit_value = entry_premium   # notional; actual P&L already in pnl
            elif early_exit_value is not None:
                exit_value = early_exit_value
                pnl = (exit_value - entry_premium) * 100 * n
            else:
                exit_value, pnl = self._expiry_pnl(sig_name, spot, strike, exit_spot,
                                                    T_entry, iv_0dte, entry_premium, n,
                                                    signal.metadata)

            et_h = int(bar_et_h)
            et_m = int((bar_et_h - et_h) * 60)
            return BacktestTrade(
                date=trade_date,
                entry_time=f"{et_h:02d}:{et_m:02d}",
                signal_type=sig_name,
                play_type=f"{strat_name}:{play_type}",
                strike=strike,
                entry_spot=spot,
                exit_spot=exit_spot,
                entry_premium=entry_premium,
                exit_value=exit_value,
                n_contracts=n,
                pnl=pnl,
                score=score,
                vix=vix,
                time_regime=regime,
            )

        return None

    def _entry_premium(
        self,
        sig_name: str,
        spot: float,
        strike: float,
        T: float,
        iv: float,
        meta: dict,
    ) -> float:
        if sig_name == "buy_call":
            return bsm_call(spot, strike, T, self.R, iv)
        if sig_name == "buy_put":
            return bsm_put(spot, strike, T, self.R, iv)
        if sig_name in ("sell_straddle", "straddle"):
            return bsm_call(spot, strike, T, self.R, iv) + bsm_put(spot, strike, T, self.R, iv)
        if sig_name in ("sell_strangle", "iron_condor"):
            ck = float(meta.get("call_strike", strike + 3))
            pk = float(meta.get("put_strike",  strike - 3))
            gross = bsm_call(spot, ck, T, self.R, iv) + bsm_put(spot, pk, T, self.R, iv)
            # Iron condor: subtract wing cost to get net credit
            wc = meta.get("wing_call_strike")
            wp = meta.get("wing_put_strike")
            if wc and wp:
                wing_cost = bsm_call(spot, float(wc), T, self.R, iv) + bsm_put(spot, float(wp), T, self.R, iv)
                return max(gross - wing_cost, 0.01)
            return gross
        return 0.0

    def _expiry_pnl(
        self,
        sig_name: str,
        spot_entry: float,
        strike: float,
        spot_exit: float,
        T: float,
        iv: float,
        entry_premium: float,
        n: int,
        meta: dict,
    ) -> Tuple[float, float]:
        """Returns (exit_value_per_share, total_pnl)."""
        if sig_name == "buy_call":
            exit_  = max(spot_exit - strike, 0.0)
            pnl    = (exit_ - entry_premium) * 100 * n

        elif sig_name == "buy_put":
            exit_  = max(strike - spot_exit, 0.0)
            pnl    = (exit_ - entry_premium) * 100 * n

        elif sig_name == "sell_straddle":
            exit_  = abs(spot_exit - strike)
            pnl    = (entry_premium - exit_) * 100 * n

        elif sig_name in ("sell_strangle", "iron_condor"):
            ck = float(meta.get("call_strike", strike + 3))
            pk = float(meta.get("put_strike",  strike - 3))
            payout_short = max(spot_exit - ck, 0.0) + max(pk - spot_exit, 0.0)
            # Iron condor: wings offset the payout above their strikes
            wc = meta.get("wing_call_strike")
            wp = meta.get("wing_put_strike")
            if wc and wp:
                payout_long = max(spot_exit - float(wc), 0.0) + max(float(wp) - spot_exit, 0.0)
                exit_ = max(payout_short - payout_long, 0.0)
            else:
                exit_ = payout_short
            pnl = (entry_premium - exit_) * 100 * n

        elif sig_name == "straddle":  # buy straddle
            exit_ = abs(spot_exit - strike)
            pnl   = (exit_ - entry_premium) * 100 * n

        else:
            exit_, pnl = 0.0, 0.0

        return exit_, pnl

    # ── data helpers ──────────────────────────────────────────────────────────

    async def _fetch_bars_with_retry(self, symbol: str, date: str, retries: int = 3) -> List:
        for attempt in range(retries):
            bars = await self._polygon.get_bars(symbol, date, multiplier=1, timespan="minute")
            if bars:
                return bars
            if attempt < retries - 1:
                await asyncio.sleep(2.0 * (attempt + 1))  # backoff on empty/rate-limit
        return []

    async def _fetch_vix(self, start: str, end: str) -> Dict[str, float]:
        key = self._polygon.api_key
        url = f"https://api.polygon.io/v2/aggs/ticker/I:VIX/range/1/day/{start}/{end}"
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                resp = await c.get(url, params={"apiKey": key, "sort": "asc", "limit": 5000})
            if resp.status_code == 200:
                out = {}
                for r in resp.json().get("results", []):
                    dt = datetime.utcfromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d")
                    out[dt] = float(r["c"])
                logger.info(f"[Backtest] VIX: {len(out)} days loaded")
                return out
            logger.warning(f"[Backtest] VIX HTTP {resp.status_code} — will estimate from bars")
        except Exception as e:
            logger.warning(f"[Backtest] VIX fetch error: {e} — will estimate from bars")
        return {}

    @staticmethod
    def _estimate_vix(bars: List[Candle]) -> float:
        """Proxy VIX from intraday bar range when VIX data unavailable."""
        if len(bars) < 2:
            return 18.0
        highs  = [b.high  for b in bars]
        lows   = [b.low   for b in bars]
        closes = [b.close for b in bars]
        daily_range = (max(highs) - min(lows)) / closes[0]
        # Annualize: daily range ≈ 1-day realized vol; VIX ≈ realized × 1.1 × sqrt(252)
        return float(np.clip(daily_range * math.sqrt(252) * 1.1 * 100, 10.0, 80.0))

    @staticmethod
    def _date_range(start: str, end: str) -> List[str]:
        days, cur = [], datetime.strptime(start, "%Y-%m-%d")
        end_      = datetime.strptime(end,   "%Y-%m-%d")
        while cur <= end_:
            if cur.weekday() < 5:
                days.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)
        return days

    @staticmethod
    def _is_market_hours(ts: datetime) -> bool:
        et = ts.hour * 60 + ts.minute - 4 * 60  # UTC → EDT approx
        return 9 * 60 + 30 <= et < 15 * 60 + 55

    @staticmethod
    def _et_hour(ts: datetime) -> float:
        return ts.hour + ts.minute / 60.0 - 4.0


# ── Reporter ──────────────────────────────────────────────────────────────────

class BacktestReporter:

    @staticmethod
    def print(result: BacktestResult) -> None:
        t = result.trades
        n = len(t)

        print("\n" + "=" * 70)
        print(f"  BACKTEST  |  {result.symbol}  |  {result.start} -> {result.end}")
        print("=" * 70)

        if not t:
            print("  No trades generated.")
            return

        wins      = [x for x in t if x.win]
        losses    = [x for x in t if not x.win]
        avg_win   = np.mean([x.pnl for x in wins])   if wins   else 0.0
        avg_loss  = np.mean([x.pnl for x in losses]) if losses else 0.0
        best      = max(t, key=lambda x: x.pnl)
        worst     = min(t, key=lambda x: x.pnl)
        days      = len(result.daily_pnl)

        print(f"\n  Period         : {result.start} to {result.end}")
        print(f"  Days fetched   : {days}  |  Trades: {n}  ({n/max(days,1)*100:.0f}% signal rate)")
        print(f"  Capital        : ${result.initial_capital:,.0f}")
        print(f"\n  Total P&L      : ${result.total_pnl:+,.2f}")
        print(f"  Return         : {result.total_pnl/result.initial_capital*100:+.1f}%")
        print(f"  Win Rate       : {result.win_rate:.1%}  ({len(wins)}W / {len(losses)}L)")
        print(f"  Profit Factor  : {result.profit_factor:.2f}x")
        print(f"  Avg Win        : ${avg_win:+.0f}")
        print(f"  Avg Loss       : ${avg_loss:+.0f}")
        print(f"  Sharpe (ann.)  : {result.sharpe:.2f}")
        print(f"  Max Drawdown   : {result.max_drawdown:.1%}")
        print(f"\n  Best trade     : {best.date}  ${best.pnl:+.0f}  ({best.play_type}  {best.signal_type})")
        print(f"  Worst trade    : {worst.date}  ${worst.pnl:+.0f}  ({worst.play_type}  {worst.signal_type})")

        # by play type
        print("\n  By Play Type:")
        print(f"  {'Play':12s}  {'N':>4}  {'Win%':>5}  {'AvgW':>7}  {'AvgL':>7}  {'Total':>8}")
        print("  " + "-" * 52)
        all_sub_plays = sorted({x.play_type.split(":")[-1] for x in t})
        for play in all_sub_plays:
            pt = [x for x in t if x.play_type.split(":")[-1] == play]
            if not pt:
                continue
            w   = [x for x in pt if x.win]
            l   = [x for x in pt if not x.win]
            wr  = len(w) / len(pt)
            aw  = np.mean([x.pnl for x in w]) if w else 0.0
            al  = np.mean([x.pnl for x in l]) if l else 0.0
            tot = sum(x.pnl for x in pt)
            print(f"  {play:12s}  {len(pt):>4}  {wr:>4.0%}  {aw:>+7.0f}  {al:>+7.0f}  {tot:>+8.0f}")

        # by regime
        print("\n  By Time Regime:")
        print(f"  {'Regime':12s}  {'N':>4}  {'Win%':>5}  {'Avg P&L':>8}")
        print("  " + "-" * 36)
        for reg in ["open", "morning", "midday", "afternoon", "close"]:
            rt = [x for x in t if x.time_regime == reg]
            if not rt:
                continue
            wr  = sum(1 for x in rt if x.win) / len(rt)
            avg = np.mean([x.pnl for x in rt])
            print(f"  {reg:12s}  {len(rt):>4}  {wr:>4.0%}  {avg:>+8.0f}")

        # trade log
        print(f"\n  {'Date':12s}  {'Time':5s}  {'Play':10s}  {'Signal':16s}  "
              f"{'K':>5}  {'Entry':>6}  {'Exit':>6}  {'n':>2}  {'Score':>5}  {'VIX':>4}  {'P&L':>8}")
        print("  " + "-" * 95)
        for tr in sorted(t, key=lambda x: x.date)[-25:]:
            print(
                f"  {tr.date}  {tr.entry_time}  {tr.play_type:10s}  "
                f"{tr.signal_type:16s}  {tr.strike:>5.0f}  "
                f"{tr.entry_spot:>6.1f}  {tr.exit_spot:>6.1f}  "
                f"{tr.n_contracts:>2}  {tr.score:>4.0%}  {tr.vix:>4.1f}  "
                f"${tr.pnl:>+7.0f}  {'WIN' if tr.win else 'LOS'}"
            )

        print("=" * 70 + "\n")
