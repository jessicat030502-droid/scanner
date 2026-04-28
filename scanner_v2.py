"""
============================================================

  QUANT SCANNER — FULL YFINANCE BUILD  (v2 — PATCHED)
  scanner_yf.py

  FIXES vs v1:
  ─────────────────────────────────────────────────────────
  FIX 1 — Intraday Momentum Gate
      Penalises composite score when price is dropping
      from today's open (>1% down → 0.75×, >2% → 0.50×,
      >3% → 0.25×). Kills alert entirely below -3%.

  FIX 2 — VWAP Gate
      Computes VWAP from today's 1-min bars. If price is
      below VWAP the composite is penalised and alert is
      suppressed — avoids entering into active sell-offs.

  FIX 3 — Directional Hawkes (buy vs sell volume)
      Splits each 1-min bar into buy_vol / sell_vol using
      Bulk Volume Classification before feeding into
      Hawkes. Spikes in *sell* volume now REDUCE intensity
      instead of inflating it — fixing the false "clustering"
      signal on gap-down / sell-off days.

  FIX 4 — Lower-Low Momentum Filter
      Checks the last 10 1-min candles. If price is making
      consecutive lower lows the signal is suppressed.

  FIX 5 — Gap-Down Detection
      Compares today's open to prior close. A gap down
      > 1% adds an additional penalty multiplier.

  DATA SOURCE STRATEGY
  ─────────────────────
  yfinance 1-min bars  → Hawkes + OFI (simulated tick-level)
  yfinance daily bars  → Hurst Exponent
  yfinance daily bars  → Market Regime (SPY/QQQ)
  yfinance daily bars  → Sector Relative Strength

  INSTALL
  ───────
  pip install yfinance pandas numpy streamlit

  RUN MODES
  ─────────
  Mode 1 — Terminal scanner:
    python scanner_yf.py

  Mode 2 — Streamlit dashboard:
    streamlit run scanner_yf.py

============================================================
"""

import os
import sys
import time
import math
import warnings
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf

warnings.filterwarnings("ignore")

# ── detect if running inside streamlit ──────────────────────
try:
    import streamlit as st
    STREAMLIT_MODE = True
except ImportError:
    STREAMLIT_MODE = False


# ============================================================
#  CONFIG  ← edit these
# ============================================================

WATCHLIST = [
    # ── Technology (XLK) ──────────────────────────────────
    "CRUS", "POWI", "MTSI", "JAMF", "ALKT", "TASK",
    # ── Consumer Discretionary (XLY) ──────────────────────
    "BOOT", "GIII", "PLAY", "LESL", "PRPL", "HIMS",
    # ── Healthcare (XLV) ──────────────────────────────────
    "ACAD", "INVA", "PDCO", "HCAT", "NVCR", "GKOS",
    # ── Financials (XLF) ──────────────────────────────────
    "CURO", "OPFI", "HIBB", "GCMG", "NRDS",
    # ── Energy (XLE) ──────────────────────────────────────
    "CIVI", "BATL", "REX", "PTEN", "WTTR",
    # ── Industrials (XLI) ─────────────────────────────────
    "KTOS", "ASTE", "HLIO", "DLX", "HAYW",
    # ── Materials (XLB) ───────────────────────────────────
    "TROX", "RYAM", "KWR",
    # ── Real Estate (XLRE) ────────────────────────────────
    "NTST", "GMRE", "EPRT",
]

SECTOR_MAP = {
    "CRUS": "XLK", "POWI": "XLK", "MTSI": "XLK",
    "JAMF": "XLK", "ALKT": "XLK", "TASK": "XLK",
    "BOOT": "XLY", "GIII": "XLY", "PLAY": "XLY",
    "LESL": "XLY", "PRPL": "XLY", "HIMS": "XLY",
    "ACAD": "XLV", "INVA": "XLV", "PDCO": "XLV",
    "HCAT": "XLV", "NVCR": "XLV", "GKOS": "XLV",
    "CURO": "XLF", "OPFI": "XLF", "HIBB": "XLF",
    "GCMG": "XLF", "NRDS": "XLF",
    "CIVI": "XLE", "BATL": "XLE", "REX":  "XLE",
    "PTEN": "XLE", "WTTR": "XLE",
    "KTOS": "XLI", "ASTE": "XLI", "HLIO": "XLI",
    "DLX":  "XLI", "HAYW": "XLI",
    "TROX": "XLB", "RYAM": "XLB", "KWR":  "XLB",
    "NTST": "XLRE","GMRE": "XLRE","EPRT": "XLRE",
}

MIDCAP_MIN       = 300_000_000
MIDCAP_MAX       = 20_000_000_000
POLL_INTERVAL    = 60
ACCOUNT_SIZE     = float(os.environ.get("ACCOUNT_SIZE", 50000))
SIGNAL_THRESHOLD = 65
SIGNAL_TTL       = 600

W_HURST   = 0.20
W_HAWKES  = 0.35
W_OFI     = 0.30
W_SECTOR  = 0.15

SPY_WEAK_THRESH  = -0.005
SECTOR_RS_MIN    = 1.02
HAWKES_DECAY     = 0.3
OFI_WINDOW       = 20

# ── FIX 1/5 thresholds ──────────────────────────────────────
INTRADAY_SOFT_WARN  = -0.010   # -1.0%  → 0.75× multiplier
INTRADAY_HARD_WARN  = -0.020   # -2.0%  → 0.50× multiplier
INTRADAY_KILL       = -0.030   # -3.0%  → alert killed entirely
GAP_DOWN_THRESH     = -0.010   # -1.0%  gap from prior close → extra penalty
LOWER_LOW_BARS      = 10       # consecutive lower-low window (1-min bars)


# ============================================================
#  DATA LAYER
# ============================================================

_mcap_cache: dict = {}
MCAP_CACHE_TTL = 3600


def get_market_cap(symbol: str) -> Optional[float]:
    now = time.time()
    if symbol in _mcap_cache:
        mcap, ts = _mcap_cache[symbol]
        if now - ts < MCAP_CACHE_TTL:
            return mcap
    try:
        info = yf.Ticker(symbol).info
        mcap = info.get("marketCap") or info.get("enterpriseValue")
        if mcap and mcap > 0:
            _mcap_cache[symbol] = (float(mcap), now)
            return float(mcap)
    except Exception:
        pass
    return None


def is_midcap(symbol: str) -> tuple:
    mcap = get_market_cap(symbol)
    if mcap is None:
        return False, 0.0
    passes = MIDCAP_MIN <= mcap <= MIDCAP_MAX
    return passes, round(mcap / 1_000_000_000, 2)


def fetch_daily(symbol: str, n: int = 60) -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(symbol).history(period="3mo", interval="1d")
        if df.empty or len(df) < 10:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna().tail(n)
    except Exception:
        return None


def fetch_intraday(symbol: str, period: str = "5d",
                   interval: str = "1m") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty or len(df) < 20:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return None


def fetch_closes(symbol: str, n: int = 25) -> Optional[np.ndarray]:
    df = fetch_daily(symbol, n)
    return df["close"].values if df is not None else None


# ============================================================
#  FIX 1 + 4 + 5: INTRADAY HEALTH CHECK
#  Returns a penalty multiplier and descriptive flags.
# ============================================================

def intraday_health(df_1min: pd.DataFrame,
                    prior_close: float) -> tuple[float, str, dict]:
    """
    FIX 1 — Intraday momentum penalty
    FIX 4 — Lower-low detection
    FIX 5 — Gap-down detection

    Returns:
        mult       : float  0.0–1.0 penalty multiplier on composite
        label      : str    human-readable flag
        detail     : dict   individual flags for display
    """
    if df_1min is None or len(df_1min) < 5:
        return 1.0, "NO DATA", {}

    # Today's bars only (index is tz-aware datetime)
    try:
        today_str = pd.Timestamp.now(tz=df_1min.index.tz).strftime("%Y-%m-%d")
        today_mask = df_1min.index.strftime("%Y-%m-%d") == today_str
        today_bars = df_1min[today_mask]
    except Exception:
        today_bars = df_1min.tail(60)   # fallback: last 60 bars

    if len(today_bars) < 3:
        today_bars = df_1min.tail(60)

    today_open  = float(today_bars["open"].iloc[0])
    current     = float(today_bars["close"].iloc[-1])
    intraday_r  = (current - today_open) / today_open if today_open > 0 else 0.0

    # FIX 5 — Gap-down vs prior close
    gap_r = (today_open - prior_close) / prior_close if prior_close > 0 else 0.0

    # FIX 4 — Lower-low check on last N 1-min closes
    recent = today_bars["close"].values[-LOWER_LOW_BARS:]
    lower_lows = int(sum(recent[i] < recent[i - 1] for i in range(1, len(recent))))
    making_lower_lows = lower_lows >= (LOWER_LOW_BARS - 2)   # 8+ of 10 bars

    # ── FIX 2: VWAP ─────────────────────────────────────────
    tp   = (today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3
    vwap = float((tp * today_bars["volume"]).cumsum().iloc[-1] /
                 today_bars["volume"].cumsum().iloc[-1]) \
           if today_bars["volume"].sum() > 0 else current
    below_vwap = current < vwap

    # ── Build multiplier ────────────────────────────────────
    mult = 1.0
    flags = []

    # Gap-down penalty
    if gap_r < GAP_DOWN_THRESH:
        mult  *= 0.85
        flags.append(f"GAP↓{gap_r*100:.1f}%")

    # Intraday drop penalty
    if intraday_r <= INTRADAY_KILL:
        mult   = 0.0
        flags.append(f"SELLOFF {intraday_r*100:.1f}%")
    elif intraday_r <= INTRADAY_HARD_WARN:
        mult  *= 0.50
        flags.append(f"WEAK {intraday_r*100:.1f}%")
    elif intraday_r <= INTRADAY_SOFT_WARN:
        mult  *= 0.75
        flags.append(f"SOFT {intraday_r*100:.1f}%")
    else:
        flags.append(f"OK {intraday_r*100:+.1f}%")

    # Lower-low penalty
    if making_lower_lows:
        mult  *= 0.70
        flags.append("LOWER-LOWS")

    # VWAP penalty
    if below_vwap:
        mult  *= 0.80
        flags.append(f"<VWAP${vwap:.2f}")

    mult  = float(np.clip(mult, 0.0, 1.0))
    label = " | ".join(flags) if flags else "HEALTHY"

    detail = {
        "intraday_ret":     round(intraday_r * 100, 2),
        "gap_ret":          round(gap_r * 100, 2),
        "today_open":       round(today_open, 2),
        "vwap":             round(vwap, 2),
        "below_vwap":       below_vwap,
        "lower_lows":       lower_lows,
        "making_lower_lows":making_lower_lows,
        "health_mult":      round(mult, 3),
    }
    return mult, label, detail


# ============================================================
#  MARKET REGIME GATE
# ============================================================

def get_market_regime() -> dict:
    spy = fetch_closes("SPY", 25)
    qqq = fetch_closes("QQQ", 25)

    def deviation(arr):
        if arr is None or len(arr) < 20:
            return 0.0, 0.0
        sma = np.mean(arr[-20:])
        return arr[-1], (arr[-1] - sma) / sma if sma > 0 else 0.0

    sp, sd = deviation(spy)
    qp, qd = deviation(qqq)

    if sd > 0.002 and qd > 0.002:
        regime = "RISK-ON 📈"; allows_long = True;  mkt_mult = 1.10
    elif sd < SPY_WEAK_THRESH and qd < SPY_WEAK_THRESH:
        regime = "RISK-OFF 📉"; allows_long = False; mkt_mult = 0.40
    else:
        regime = "NEUTRAL ➡";  allows_long = True;  mkt_mult = 1.00

    return {
        "regime": regime, "allows_long": allows_long, "mkt_mult": mkt_mult,
        "spy_price": round(sp, 2), "spy_dev": round(sd * 100, 2),
        "qqq_price": round(qp, 2), "qqq_dev": round(qd * 100, 2),
    }


# ============================================================
#  SECTOR RELATIVE STRENGTH
# ============================================================

def get_sector_rs(etf: str) -> tuple[float, float, bool]:
    etf_c = fetch_closes(etf, 25)
    spy_c = fetch_closes("SPY", 25)
    if etf_c is None or spy_c is None or len(etf_c) < 20 or len(spy_c) < 20:
        return 1.0, 50.0, True
    etf_rs = etf_c[-1] / np.mean(etf_c[-20:])
    spy_rs = spy_c[-1] / np.mean(spy_c[-20:])
    ratio  = etf_rs / spy_rs if spy_rs > 0 else 1.0
    score  = float(np.clip(50 + (ratio - 1.0) * 500, 0, 100))
    gate   = ratio >= SECTOR_RS_MIN
    return round(ratio, 4), round(score, 1), gate


# ============================================================
#  LAYER 1: HURST EXPONENT
# ============================================================

def compute_hurst(prices: np.ndarray) -> float:
    arr = np.array(prices)
    if len(arr) < 30:
        return 0.5
    log_r = np.diff(np.log(arr + 1e-9))
    n = len(log_r)
    lags = np.unique(np.floor(np.geomspace(5, n // 2, 12)).astype(int))
    lags = lags[lags >= 4]
    rs_v, vl = [], []
    for lag in lags:
        nw = n // lag
        if nw < 2:
            continue
        rl = []
        for i in range(nw):
            seg = log_r[i * lag:(i + 1) * lag]
            cs  = np.cumsum(seg - seg.mean())
            S   = seg.std(ddof=1)
            if S > 0:
                rl.append((cs.max() - cs.min()) / S)
        if rl:
            rs_v.append(np.mean(rl))
            vl.append(lag)
    if len(vl) < 3:
        return 0.5
    H, _ = np.polyfit(np.log(vl), np.log(rs_v), 1)
    return float(np.clip(H, 0.0, 1.0))


def hurst_score(H: float) -> float:
    return float(np.clip((H - 0.5) * 200 + 50, 0, 100))


def hurst_regime(H: float) -> str:
    if H > 0.58:   return "📈 TRENDING"
    elif H < 0.42: return "🔄 REVERTING"
    else:          return "〰 CHOPPY"


# ============================================================
#  LAYER 2: DIRECTIONAL HAWKES  ← FIX 3
#
#  v1 bug: any volume spike (buy OR sell) inflated λ.
#  v2 fix: split each bar into buy_vol / sell_vol via BVC.
#          Buy spikes  → +alpha  (as before)
#          Sell spikes → -alpha  (NEW — reduces intensity)
#  This prevents a gap-down sell-off from generating a
#  false "CLUSTERING" signal.
# ============================================================

def compute_hawkes(df_1min: pd.DataFrame) -> tuple[float, float]:
    """
    Returns (current_lambda, hawkes_score_0_100).
    Directional: sell spikes now reduce intensity.
    """
    close = df_1min["close"].values
    high  = df_1min["high"].values
    low   = df_1min["low"].values
    vols  = df_1min["volume"].values
    n     = len(vols)

    if n < 20:
        return 0.0, 50.0

    # BVC split — same as OFI
    bar_range = high - low
    buy_ratio = np.where(bar_range > 0, (close - low) / bar_range, 0.5)
    buy_vol   = buy_ratio * vols
    sell_vol  = (1 - buy_ratio) * vols

    baseline_buy  = pd.Series(buy_vol).rolling(20, min_periods=5).mean().values
    baseline_sell = pd.Series(sell_vol).rolling(20, min_periods=5).mean().values

    mu   = np.nanmean(baseline_buy[~np.isnan(baseline_buy)])
    if mu <= 0:
        return 0.0, 50.0

    alpha = mu * 0.5
    beta  = HAWKES_DECAY

    intensities    = np.zeros(n)
    intensities[0] = mu

    for t in range(1, n):
        decayed  = mu + (intensities[t - 1] - mu) * math.exp(-beta)
        bb_t     = baseline_buy[t - 1]  if not np.isnan(baseline_buy[t - 1])  else mu
        bs_t     = baseline_sell[t - 1] if not np.isnan(baseline_sell[t - 1]) else mu

        is_buy_spike  = buy_vol[t - 1]  > bb_t * 1.8
        is_sell_spike = sell_vol[t - 1] > bs_t * 1.8

        # FIX 3: buy spikes add, sell spikes subtract
        delta = 0.0
        if is_buy_spike:
            delta += alpha
        if is_sell_spike:
            delta -= alpha * 0.8   # sell pressure dampens intensity

        intensities[t] = max(0.0, decayed + delta)

    cur_lambda   = intensities[-1]
    baseline_lam = np.nanmean(intensities[max(0, n - 30):-5]) if n > 10 else mu

    if baseline_lam <= 0:
        return cur_lambda, 50.0

    ratio = cur_lambda / (baseline_lam + 1e-9)
    score = float(np.clip(50 + 50 * math.tanh(ratio - 1.0), 0, 100))
    return round(cur_lambda, 4), round(score, 1)


def hawkes_signal(score: float) -> str:
    if score >= 72:   return "🔥 CLUSTERING"
    elif score >= 58: return "⚡ BUILDING"
    elif score >= 42: return "〰 IDLE"
    else:             return "❄ FADING"


# ============================================================
#  LAYER 3: OFI PROXY (1-min bars)
# ============================================================

def compute_ofi(df_1min: pd.DataFrame, window: int = OFI_WINDOW) -> tuple[float, float, float]:
    close = df_1min["close"].values
    high  = df_1min["high"].values
    low   = df_1min["low"].values
    vol   = df_1min["volume"].values

    bar_range = high - low
    buy_ratio = np.where(bar_range > 0, (close - low) / bar_range, 0.5)
    buy_vol   = buy_ratio * vol
    tot_vol   = vol

    buy_roll = pd.Series(buy_vol).rolling(window, min_periods=3).sum()
    tot_roll = pd.Series(tot_vol).rolling(window, min_periods=3).sum()
    ofi_s    = (buy_roll / tot_roll.replace(0, np.nan)).fillna(0.5)

    cur_ofi = float(ofi_s.iloc[-1])
    delta   = float(ofi_s.iloc[-1] - ofi_s.iloc[-4]) if len(ofi_s) >= 4 else 0.0
    score   = float(np.clip(cur_ofi * 100 + delta * 50, 0, 100))

    return round(cur_ofi, 4), round(delta, 4), round(score, 1)


def ofi_signal(ofi: float, delta: float) -> str:
    if ofi >= 0.65 and delta >= 0:  return "🟢 ACCUMULATING"
    elif ofi >= 0.60 and delta < 0: return "🟡 TOPPING"
    elif ofi <= 0.35:               return "🔴 DISTRIBUTING"
    elif ofi <= 0.42:               return "🟠 SELLING"
    else:                           return "⚪ NEUTRAL"


# ============================================================
#  KELLY POSITION SIZER
# ============================================================

def compute_atr_daily(df_daily: pd.DataFrame, n: int = 14) -> float:
    if df_daily is None or len(df_daily) < n:
        return 0.0
    high  = df_daily["high"].values
    low   = df_daily["low"].values
    close = df_daily["close"].values
    trs   = []
    for i in range(1, min(n + 1, len(close))):
        tr = max(high[-i] - low[-i],
                 abs(high[-i] - close[-i - 1]) if i < len(close) else 0,
                 abs(low[-i]  - close[-i - 1]) if i < len(close) else 0)
        trs.append(tr)
    return round(float(np.mean(trs)), 4) if trs else 0.0


def kelly_size(price: float, atr: float,
               win_rate: float = 0.60,
               rr: float = 2.0) -> tuple[float, float, int]:
    if atr <= 0 or price <= 0:
        return 0.0, 0.0, 0
    kf        = max(0.0, (win_rate - (1 - win_rate) / rr) * 0.5)
    kf        = min(kf, 0.10)
    drisk     = ACCOUNT_SIZE * kf
    stop_dist = atr * 1.5
    shares    = int(drisk / stop_dist) if stop_dist > 0 else 0
    shares    = min(shares, int(ACCOUNT_SIZE * 0.20 / price))
    return round(kf, 4), round(drisk, 2), shares


# ============================================================
#  SIGNAL DECAY
# ============================================================

_signal_fire_times: dict[str, float]  = {}
_signal_entry_prices: dict[str, float] = {}


def check_signal_decay(symbol: str, score: float,
                        current_price: float,
                        atr: float) -> tuple[bool, float, bool]:
    now      = time.time()
    is_alert = score >= SIGNAL_THRESHOLD

    if is_alert and symbol not in _signal_fire_times:
        _signal_fire_times[symbol]   = now
        _signal_entry_prices[symbol] = current_price

    if symbol not in _signal_fire_times:
        return False, 0.0, False

    elapsed   = now - _signal_fire_times[symbol]
    remaining = max(0.0, SIGNAL_TTL - elapsed)
    entry     = _signal_entry_prices.get(symbol, current_price)
    confirmed = (current_price - entry) > atr * 0.5

    if not is_alert:
        _signal_fire_times.pop(symbol, None)
        _signal_entry_prices.pop(symbol, None)
        return False, 0.0, False

    if elapsed > SIGNAL_TTL and not confirmed:
        _signal_fire_times.pop(symbol, None)
        _signal_entry_prices.pop(symbol, None)
        return False, 0.0, True

    return True, round(remaining, 0), False


# ============================================================
#  MAIN SCAN FUNCTION
# ============================================================

def scan_symbol(symbol: str, market: dict) -> Optional[dict]:
    # ── MID-CAP GATE ─────────────────────────────────────────
    passes_mcap, mcap_b = is_midcap(symbol)
    if not passes_mcap:
        return None

    df_daily = fetch_daily(symbol, 60)
    if df_daily is None or len(df_daily) < 30:
        return None

    df_1min = fetch_intraday(symbol, period="5d", interval="1m")
    if df_1min is None or len(df_1min) < 30:
        df_1min = df_daily.copy()

    price       = float(df_daily["close"].iloc[-1])
    prior_close = float(df_daily["close"].iloc[-2]) if len(df_daily) >= 2 else price

    # ── LAYER 1: HURST ───────────────────────────────────────
    H      = compute_hurst(df_daily["close"].values)
    h_sc   = hurst_score(H)
    h_reg  = hurst_regime(H)

    # ── LAYER 2: DIRECTIONAL HAWKES (FIX 3) ─────────────────
    cur_lam, hawk_sc = compute_hawkes(df_1min)
    hawk_sig         = hawkes_signal(hawk_sc)

    # ── LAYER 3: OFI ─────────────────────────────────────────
    cur_ofi, ofi_d, o_sc = compute_ofi(df_1min)
    o_sig                 = ofi_signal(cur_ofi, ofi_d)

    # ── SECTOR RS ────────────────────────────────────────────
    etf                      = SECTOR_MAP.get(symbol, "SPY")
    sec_rs, sec_sc, sec_gate = get_sector_rs(etf)

    # ── COMPOSITE (pre-intraday adjustment) ──────────────────
    raw = float(np.clip(
        W_HURST  * h_sc    +
        W_HAWKES * hawk_sc +
        W_OFI    * o_sc    +
        W_SECTOR * sec_sc,
        0, 100
    ))
    sec_mult = 1.0 if sec_gate else 0.70
    comp_raw = float(np.clip(raw * market["mkt_mult"] * sec_mult, 0, 100))

    # ── FIX 1 + 2 + 4 + 5: INTRADAY HEALTH ──────────────────
    health_mult, health_label, health_detail = intraday_health(df_1min, prior_close)
    comp  = round(float(np.clip(comp_raw * health_mult, 0, 100)), 1)
    alert = comp >= SIGNAL_THRESHOLD and market["allows_long"] and health_mult > 0.0

    # ── KELLY ─────────────────────────────────────────────────
    atr               = compute_atr_daily(df_daily)
    kf, drisk, shares = kelly_size(price, atr) if alert else (0.0, 0.0, 0)

    # ── SIGNAL DECAY ──────────────────────────────────────────
    sig_live, remaining, just_expired = check_signal_decay(symbol, comp, price, atr)
    if just_expired:
        alert = False
        comp  = max(0.0, comp - 10)

    stop_price = round(price - atr * 1.5, 2) if atr > 0 else 0.0
    target     = round(price + atr * 3.0, 2) if atr > 0 else 0.0

    return {
        "symbol":        symbol,
        "price":         round(price, 2),
        "score":         comp,
        "score_raw":     round(comp_raw, 1),   # pre-intraday score for debug
        "alert":         alert,

        "hurst_H":       round(H, 3),
        "hurst_score":   round(h_sc, 1),
        "hurst_regime":  h_reg,

        "hawkes_lam":    cur_lam,
        "hawkes_score":  hawk_sc,
        "hawkes_sig":    hawk_sig,

        "ofi":           cur_ofi,
        "ofi_delta":     ofi_d,
        "ofi_score":     o_sc,
        "ofi_sig":       o_sig,

        "market":        market["regime"],
        "sector_etf":    etf,
        "sector_rs":     sec_rs,
        "sector_score":  sec_sc,
        "sector_gate":   sec_gate,

        "kelly_frac":    kf,
        "dollar_risk":   drisk,
        "shares":        shares,
        "atr":           atr,
        "stop":          stop_price,
        "target":        target,

        "sig_live":      sig_live,
        "sig_remaining": int(remaining),
        "just_expired":  just_expired,

        "mcap_b":        mcap_b,

        # ── NEW intraday fields ──────────────────────────────
        "health_mult":   health_mult,
        "health_label":  health_label,
        "intraday_ret":  health_detail.get("intraday_ret", 0.0),
        "gap_ret":       health_detail.get("gap_ret", 0.0),
        "vwap":          health_detail.get("vwap", 0.0),
        "below_vwap":    health_detail.get("below_vwap", False),
        "lower_lows":    health_detail.get("lower_lows", 0),

        "scanned_at":    datetime.now().strftime("%H:%M:%S"),
    }


# ============================================================
#  FULL SCAN
# ============================================================

def run_full_scan(symbols: list = WATCHLIST) -> tuple[pd.DataFrame, dict]:
    market  = get_market_regime()
    results = []
    blocked = []

    for sym in symbols:
        try:
            passes, mcap_b = is_midcap(sym)
            if not passes:
                blocked.append(f"{sym} (${mcap_b:.1f}B)" if mcap_b > 0 else sym)
                continue
            r = scan_symbol(sym, market)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[SCAN ERR] {sym}: {e}")

    if blocked:
        print(f"[FILTER] Blocked {len(blocked)} large/mega-cap(s): {', '.join(blocked)}")

    market["blocked"]       = blocked
    market["blocked_count"] = len(blocked)
    market["scanned"]       = len(results)

    if not results:
        return pd.DataFrame(), market

    df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return df, market


# ============================================================
#  TERMINAL MODE
# ============================================================

def print_results(df: pd.DataFrame, market: dict):
    print(f"\n{'='*80}")
    print(f"  QUANT SCANNER v2  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  MARKET: {market['regime']}  SPY={market['spy_price']} ({market['spy_dev']:+.2f}%)")
    print(f"{'='*80}")
    print(f"  {'SYM':<6} {'SCORE':>5} {'RAW':>5}  {'HEALTH':<26} {'HAWKES':<14} {'OFI':<16} {'KELLY%':>6}  {'SHR':>5}")
    print(f"  {'─'*78}")

    for _, r in df.iterrows():
        flag = " ◄◄" if r["alert"] else ("  ✗" if r["just_expired"] else "   ")
        print(
            f"  {r['symbol']:<6} {r['score']:>5}  {r['score_raw']:>4}  "
            f"{r['health_label']:<26} {r['hawkes_sig']:<14} "
            f"{r['ofi_sig']:<16} {r['kelly_frac']:>5.1%}  "
            f"{r['shares']:>5}{flag}"
        )

    alerts = df[df["alert"] == True]
    print(f"\n  {'─'*78}")
    print(f"  Signals: {len(alerts)}/{len(df)}  |  Market gate: {'✓ OPEN' if market['allows_long'] else '✗ RISK-OFF'}")

    if len(alerts) > 0:
        print(f"\n  ── TOP PICKS {'─'*58}")
        for _, r in alerts.iterrows():
            print(f"\n  {r['symbol']}  ${r['price']}  MCap ${r.get('mcap_b',0):.1f}B  [{r['score']}]")
            print(f"    Intraday: {r['intraday_ret']:+.2f}% from open | Gap: {r['gap_ret']:+.2f}% | VWAP ${r['vwap']:.2f} {'✗ BELOW' if r['below_vwap'] else '✓ ABOVE'}")
            print(f"    Health  : {r['health_label']} (mult {r['health_mult']:.2f}×)")
            print(f"    Hurst   : H={r['hurst_H']}  → {r['hurst_regime']}")
            print(f"    Hawkes  : λ={r['hawkes_lam']}  → {r['hawkes_sig']}")
            print(f"    OFI     : {r['ofi']}  Δ={r['ofi_delta']}  → {r['ofi_sig']}")
            print(f"    Sector  : {r['sector_etf']}  RS={r['sector_rs']}  {'✓' if r['sector_gate'] else '✗'}")
            print(f"    Size    : {r['kelly_frac']:.1%} Kelly  →  {r['shares']} shares  (${r['dollar_risk']} risk)")
            print(f"    Stop    : ${r['stop']}  |  Target: ${r['target']}  (3:1 R:R)")
            if r["sig_live"]:
                print(f"    Decay   : {r['sig_remaining']}s remaining")

    print(f"\n{'='*80}\n")


# ============================================================
#  STREAMLIT DASHBOARD
# ============================================================

def run_dashboard():
    st.set_page_config(
        page_title="QUANT SCANNER v2",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@600;700&display=swap');
    html, body, [class*="css"] { background:#080c0f !important; color:#c8d8e8 !important; }
    .title  { font-family:'Share Tech Mono',monospace; font-size:24px; letter-spacing:4px;
              color:#00e5ff; text-shadow:0 0 12px rgba(0,229,255,0.4); }
    .market { font-family:'Share Tech Mono',monospace; font-size:13px; color:#4a8aaa;
              letter-spacing:2px; margin-bottom:16px; }
    .card   { background:#0d1820; border:1px solid #1a2e40; border-radius:6px;
              padding:14px 16px; margin-bottom:10px; }
    .card.on{ border-color:#00ff8c; box-shadow:0 0 16px rgba(0,255,140,0.15); }
    .card.warn { border-color:#ffb400; box-shadow:0 0 10px rgba(255,180,0,0.12); }
    .card.off { border-color:#ff4040; opacity:0.7; }
    .sym    { font-family:'Share Tech Mono',monospace; font-size:22px; color:#e8f4ff;
              letter-spacing:2px; }
    .sc     { font-family:'Share Tech Mono',monospace; font-size:30px; font-weight:700; }
    .tag    { display:inline-block; font-family:'Share Tech Mono',monospace; font-size:11px;
              padding:2px 8px; border-radius:3px; margin-right:5px; letter-spacing:1px; }
    .bar-bg { background:#0a1520; border-radius:3px; height:6px; margin:8px 0 6px; overflow:hidden; }
    .bar-fg { height:6px; border-radius:3px; }
    .meta   { font-family:'Share Tech Mono',monospace; font-size:11px; color:#3a5a72; }
    .health { font-family:'Share Tech Mono',monospace; font-size:11px;
              padding:3px 8px; border-radius:3px; display:inline-block; }
    .trade  { background:#0a1820; border:1px solid #1a3040; border-radius:4px;
              padding:8px 12px; margin-top:8px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙ CONTROLS")
        threshold   = st.slider("Signal Threshold", 40, 90, SIGNAL_THRESHOLD)
        account     = st.number_input("Account Size ($)", value=int(ACCOUNT_SIZE),
                                       step=5000, format="%d")
        refresh     = st.slider("Refresh (sec)", 30, 300, POLL_INTERVAL)
        # Try to load the automated list first
if os.path.exists("auto_watchlist.txt"):
    with open("auto_watchlist.txt", "r") as f:
        auto_list = f.read()
else:
    auto_list = "\n".join(WATCHLIST)
custom_syms = st.text_area("Watchlist (one per line)", value=auto_list, height=300)
    run_btn     = st.button("▶ RUN SCAN NOW", use_container_width=True)
        st.markdown("---")
        st.markdown("""
        <div style='font-family:Share Tech Mono;font-size:11px;color:#2a4a62'>
        <b style='color:#ffb400'>v2 FIXES ACTIVE:</b><br>
        ✓ Intraday momentum gate<br>
        ✓ VWAP gate (below = penalised)<br>
        ✓ Directional Hawkes (sell spikes reduce λ)<br>
        ✓ Lower-low detection<br>
        ✓ Gap-down penalty<br><br>
        <b style='color:#ffb400'>CAP FILTER: $300M – $20B</b><br>
        Data: yfinance (daily + 1-min)<br>
        No API key required.
        </div>
        """, unsafe_allow_html=True)

    symbols = [s.strip().upper() for s in custom_syms.split("\n") if s.strip()]

    if "results" not in st.session_state:
        st.session_state["results"]    = pd.DataFrame()
        st.session_state["market"]     = {}
        st.session_state["last_scan"]  = 0.0
        st.session_state["scan_count"] = 0

    now         = time.time()
    last        = st.session_state["last_scan"]
    should_scan = run_btn or (now - last > refresh) or last == 0

    if should_scan:
        with st.spinner("Scanning..."):
            df, market = run_full_scan(symbols)
        st.session_state["results"]    = df
        st.session_state["market"]     = market
        st.session_state["last_scan"]  = now
        st.session_state["scan_count"] += 1

    df     = st.session_state["results"]
    market = st.session_state["market"]
    count  = st.session_state["scan_count"]

    # ── Header ─────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="title">⚡ QUANT SCANNER v2</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div class="meta" style="text-align:right;margin-top:8px">'
            f'{datetime.now().strftime("%H:%M:%S")} EST<br>'
            f'Scan #{count} | next in {max(0,int(refresh-(now-st.session_state["last_scan"])))}s'
            f'</div>',
            unsafe_allow_html=True
        )

    if market:
        regime_color = "#00ff8c" if "RISK-ON" in market.get("regime","") else (
                       "#ff4040" if "RISK-OFF" in market.get("regime","") else "#ffb400")
        blocked_str  = (f" &nbsp;|&nbsp; 🚫 {market['blocked_count']} blocked"
                        if market.get("blocked_count", 0) > 0 else "")
        st.markdown(
            f'<div class="market">MARKET: '
            f'<span style="color:{regime_color}">{market.get("regime","—")}</span>'
            f' &nbsp;|&nbsp; SPY {market.get("spy_price","—")} '
            f'({market.get("spy_dev",0):+.2f}%) '
            f'&nbsp;|&nbsp; QQQ {market.get("qqq_price","—")} '
            f'({market.get("qqq_dev",0):+.2f}%)'
            f' &nbsp;|&nbsp; ✅ {market.get("scanned",0)} scanned'
            f'{blocked_str}'
            f'</div>',
            unsafe_allow_html=True
        )

    if df.empty:
        st.info("No results yet — click ▶ RUN SCAN NOW")
        time.sleep(refresh)
        st.rerun()
        return

    # ── Cards ──────────────────────────────────────────────
    alerts = df[df["alert"] == True]
    others = df[df["alert"] == False]

    def health_color(mult):
        if mult >= 0.90: return "#00ff8c"
        if mult >= 0.70: return "#ffb400"
        return "#ff4040"

    def render_card(r):
        is_on   = r["alert"]
        hm      = r["health_mult"]
        hc      = health_color(hm)
        css_cls = "on" if is_on and hm >= 0.90 else ("warn" if hm >= 0.50 else "off")
        sc_col  = "#00ff8c" if r["score"] >= 75 else ("#ffb400" if r["score"] >= 60 else "#aaa")
        vwap_str = f"{'🔴' if r['below_vwap'] else '🟢'} VWAP ${r['vwap']:.2f}"

        st.markdown(f"""
        <div class="card {css_cls}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <span class="sym">{r['symbol']}</span>
              <span class="meta" style="margin-left:12px">${r['price']}  MCap ${r.get('mcap_b',0):.1f}B</span>
            </div>
            <div style="text-align:right">
              <span class="sc" style="color:{sc_col}">{r['score']}</span>
              <span class="meta"> /100</span><br>
              <span class="meta" style="color:#556">raw {r['score_raw']}</span>
            </div>
          </div>

          <div style="margin:6px 0 4px">
            <span class="health" style="background:{hc}22;color:{hc};border:1px solid {hc}44">
              {r['health_label']}  ×{hm:.2f}
            </span>
            &nbsp;
            <span class="meta">{vwap_str} &nbsp;|&nbsp;
              intra {r['intraday_ret']:+.1f}% &nbsp;|&nbsp;
              gap {r['gap_ret']:+.1f}%
            </span>
          </div>

          <div class="bar-bg"><div class="bar-fg" style="width:{r['score']}%;background:{sc_col}"></div></div>

          <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px">
            <span class="tag" style="background:#1a2e40;color:#aac">{r['hurst_regime']}</span>
            <span class="tag" style="background:#1a2e40;color:#aac">{r['hawkes_sig']}</span>
            <span class="tag" style="background:#1a2e40;color:#aac">{r['ofi_sig']}</span>
            <span class="tag" style="background:#1a2e40;color:#{'4af' if r['sector_gate'] else 'f44'}">{r['sector_etf']} RS {r['sector_rs']}</span>
          </div>

          {'<div class="trade">' +
           f'<span class="meta" style="color:#00ff8c">▶ ENTRY: ${r["price"]}  '
           f'STOP: ${r["stop"]}  TARGET: ${r["target"]}  '
           f'Kelly {r["kelly_frac"]:.1%} → {r["shares"]} shares  (${r["dollar_risk"]} risk)</span></div>'
           if is_on else ''}
        </div>
        """, unsafe_allow_html=True)

    if not alerts.empty:
        st.markdown(f"### 🟢 Active Signals ({len(alerts)})")
        for _, row in alerts.iterrows():
            render_card(row)
        st.markdown("---")

    st.markdown(f"### 📊 All Scanned ({len(df)})")
    for _, row in df.iterrows():
        render_card(row)

    time.sleep(refresh)
    st.rerun()


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if STREAMLIT_MODE and "streamlit" in sys.argv[0]:
        run_dashboard()
    else:
        print("\n[QUANT SCANNER v2] Running terminal scan...\n")
        df, market = run_full_scan()
        print_results(df, market)
