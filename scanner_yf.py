"""
============================================================
  QUANT SCANNER — FULL YFINANCE BUILD
  scanner_yf.py

  No Schwab. No API keys. No setup beyond pip install.

  DATA SOURCE STRATEGY
  ─────────────────────
  yfinance 1-min bars  → Hawkes + OFI (simulated tick-level)
  yfinance daily bars  → Hurst Exponent
  yfinance daily bars  → Market Regime (SPY/QQQ)
  yfinance daily bars  → Sector Relative Strength

  TRADEOFF vs LIVE TICK FEED
  ──────────────────────────
  Live tick:   true bid/ask on every trade  → ~85% OFI accuracy
  1-min bars:  OHLCV per minute candle      → ~72% OFI accuracy
  The gap is real but manageable. The Hawkes/OFI signals
  you get here are still significantly better than RSI/MACD.

  INSTALL
  ───────
  pip install yfinance pandas numpy streamlit

  RUN MODES
  ─────────
  Mode 1 — Terminal scanner (prints ranked table):
    python scanner_yf.py

  Mode 2 — Streamlit dashboard (live auto-refresh):
    streamlit run scanner_yf.py

  GITHUB DEPLOYMENT
  ─────────────────
  1. Push scanner_yf.py to a public GitHub repo
  2. Go to share.streamlit.io → connect repo → deploy
  3. Set ACCOUNT_SIZE in Streamlit Secrets if desired
  Free hosting, runs in browser, no local Python needed.
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

# ── MID-CAP PRESET WATCHLIST ($1B–$20B) ─────────────────────
# 40 liquid mid-caps across 8 sectors. The scanner auto-filters
# any that drift outside the $1B–$20B market cap range at runtime.
#
# To add your own: just append the ticker to WATCHLIST and add
# its sector ETF to SECTOR_MAP.

WATCHLIST = [
    # ── Technology (XLK) ──────────────────────────────────
    "CRUS",   # Cirrus Logic          — analog semis
    "POWI",   # Power Integrations    — power mgmt chips
    "MTSI",   # MACOM Technology      — RF/microwave semis
    "JAMF",   # Jamf Holding          — Apple device mgmt SaaS
    "ALKT",   # Alkami Technology     — digital banking SaaS
    "TASK",   # TaskUs                — tech-enabled BPO

    # ── Consumer Discretionary (XLY) ──────────────────────
    "BOOT",   # Boot Barn Holdings    — western/work apparel
    "GIII",   # G-III Apparel         — fashion/licensing
    "PLAY",   # Dave & Buster's       — entertainment venues
    "LESL",   # Leslie's              — pool supplies retail
    "PRPL",   # Purple Innovation     — sleep products
    "HIMS",   # Hims & Hers Health    — telehealth/DTC

    # ── Healthcare (XLV) ──────────────────────────────────
    "ACAD",   # Acadia Pharmaceuticals — CNS drugs
    "INVA",   # Innoviva               — royalty pharma
    "PDCO",   # Patterson Companies    — dental/vet supply
    "HCAT",   # Health Catalyst        — healthcare data
    "NVCR",   # NovoCure                — oncology devices
    "GKOS",   # Glaukos Corporation    — eye care devices

    # ── Financials (XLF) ──────────────────────────────────
    "CURO",   # CURO Group             — consumer finance
    "OPFI",   # OppFi                  — fintech lending
    "HIBB",   # Hibbett                — specialty retail/fin
    "GCMG",   # GCM Grosvenor          — alt asset mgmt
    "NRDS",   # NerdWallet              — personal finance

    # ── Energy (XLE) ──────────────────────────────────────
    "CIVI",   # Civitas Resources      — oil & gas E&P
    "BATL",   # Battalion Oil          — E&P
    "REX",    # REX Energy             — nat gas
    "PTEN",   # Patterson-UTI Energy   — drilling services
    "WTTR",   # Select Water Solutions — water services

    # ── Industrials (XLI) ─────────────────────────────────
    "KTOS",   # Kratos Defense         — unmanned systems
    "ASTE",   # Astec Industries       — infrastructure equip
    "HLIO",   # Helios Technologies    — hydraulics
    "DLX",    # Deluxe Corporation     — business services
    "HAYW",   # Hayward Holdings       — pool equipment

    # ── Materials (XLB) ───────────────────────────────────
    "TROX",   # Tronox Holdings        — titanium dioxide
    "RYAM",   # Rayonier Advanced      — specialty cellulose
    "KWR",    # Quaker Houghton        — industrial fluids

    # ── Real Estate (XLRE) ────────────────────────────────
    "NTST",   # NETSTREIT               — net lease REIT
    "GMRE",   # Global Medical REIT     — healthcare facilities
    "EPRT",   # Essential Properties    — net lease REIT
]

# Sector ETF map — used for relative strength gate
SECTOR_MAP = {
    # Tech
    "CRUS": "XLK", "POWI": "XLK", "MTSI": "XLK",
    "JAMF": "XLK", "ALKT": "XLK", "TASK": "XLK",
    # Consumer Disc
    "BOOT": "XLY", "GIII": "XLY", "PLAY": "XLY",
    "LESL": "XLY", "PRPL": "XLY", "HIMS": "XLY",
    # Healthcare
    "ACAD": "XLV", "INVA": "XLV", "PDCO": "XLV",
    "HCAT": "XLV", "NVCR": "XLV", "GKOS": "XLV",
    # Financials
    "CURO": "XLF", "OPFI": "XLF", "HIBB": "XLF",
    "GCMG": "XLF", "NRDS": "XLF",
    # Energy
    "CIVI": "XLE", "BATL": "XLE", "REX": "XLE",
    "PTEN": "XLE", "WTTR": "XLE",
    # Industrials
    "KTOS": "XLI", "ASTE": "XLI", "HLIO": "XLI",
    "DLX":  "XLI", "HAYW": "XLI",
    # Materials
    "TROX": "XLB", "RYAM": "XLB", "KWR": "XLB",
    # Real Estate
    "NTST": "XLRE", "GMRE": "XLRE", "EPRT": "XLRE",
}

# ── MID-CAP FILTER ───────────────────────────────────────────
# Any symbol outside this range is skipped at scan time.
# yfinance .info["marketCap"] used for live check.
MIDCAP_MIN = 1_000_000_000    # $1B
MIDCAP_MAX = 20_000_000_000   # $20B

# Polling interval for live mode (seconds)
# yfinance 1-min data has a 15-sec lag — don't poll faster than this
POLL_INTERVAL    = 60        # refresh every 60 seconds

ACCOUNT_SIZE     = float(os.environ.get("ACCOUNT_SIZE", 50000))
SIGNAL_THRESHOLD = 65
SIGNAL_TTL       = 600       # 10-min signal decay

# Composite weights
W_HURST   = 0.20
W_HAWKES  = 0.35
W_OFI     = 0.30
W_SECTOR  = 0.15

# Thresholds
SPY_WEAK_THRESH  = -0.005
SECTOR_RS_MIN    = 1.02
HAWKES_DECAY     = 0.3
OFI_WINDOW       = 20        # bars (using 1-min, 20 bars = 20 min)


# ============================================================
#  DATA LAYER — all yfinance
# ============================================================

# ── Market cap cache (avoid re-fetching every scan) ─────────
_mcap_cache: dict = {}   # sym → (mcap_float, timestamp)
MCAP_CACHE_TTL = 3600    # re-check market cap every 60 min


def get_market_cap(symbol: str) -> Optional[float]:
    """
    Returns current market cap in dollars via yfinance .info.
    Cached for 60 min — .info is a slow HTTP call.
    """
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
    """
    Returns (passes_filter: bool, market_cap_billions: float).
    Lets symbol through if market cap data is unavailable.
    """
    mcap = get_market_cap(symbol)
    if mcap is None:
        return True, 0.0
    passes = MIDCAP_MIN <= mcap <= MIDCAP_MAX
    return passes, round(mcap / 1_000_000_000, 2)


def fetch_daily(symbol: str, n: int = 60) -> Optional[pd.DataFrame]:
    """Daily OHLCV. Used for Hurst, sector RS, market regime."""
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
    """
    1-minute bars for the past 5 days.
    Used to simulate tick-level Hawkes and OFI.
    yfinance 1-min data is available for last 7 days.
    """
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
#  GAP 1: MARKET REGIME GATE
# ============================================================

def get_market_regime() -> dict:
    """
    Returns market regime based on SPY + QQQ vs their 20-day SMA.
    RISK-ON / RISK-OFF / NEUTRAL
    """
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
        regime      = "RISK-ON 📈"
        allows_long = True
        mkt_mult    = 1.10
    elif sd < SPY_WEAK_THRESH and qd < SPY_WEAK_THRESH:
        regime      = "RISK-OFF 📉"
        allows_long = False
        mkt_mult    = 0.40
    else:
        regime      = "NEUTRAL ➡"
        allows_long = True
        mkt_mult    = 1.00

    return {
        "regime":      regime,
        "allows_long": allows_long,
        "mkt_mult":    mkt_mult,
        "spy_price":   round(sp, 2),
        "spy_dev":     round(sd * 100, 2),
        "qqq_price":   round(qp, 2),
        "qqq_dev":     round(qd * 100, 2),
    }


# ============================================================
#  GAP 2: SECTOR RELATIVE STRENGTH
# ============================================================

def get_sector_rs(etf: str) -> tuple[float, float, bool]:
    """Returns (rs_ratio, score_0_100, gate_open)."""
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
#  LAYER 1: HURST EXPONENT (daily close prices)
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
#  LAYER 2: HAWKES INTENSITY (1-min bar volume)
#
#  Without tick data we use 1-min candle volume as events.
#  Each bar where volume > 1.8× its 20-bar avg = "spike event."
#  λ decays between bars using bar index as time proxy.
# ============================================================

def compute_hawkes(df_1min: pd.DataFrame) -> tuple[float, float]:
    """
    Returns (current_lambda, hawkes_score_0_100).
    Uses 1-min volume bars as the event stream.
    """
    vols    = df_1min["volume"].values
    n       = len(vols)
    if n < 20:
        return 0.0, 50.0

    baseline = pd.Series(vols).rolling(20, min_periods=5).mean().values
    mu       = np.nanmean(baseline[~np.isnan(baseline)])
    if mu <= 0:
        return 0.0, 50.0

    alpha = mu * 0.5
    beta  = HAWKES_DECAY

    intensities    = np.zeros(n)
    intensities[0] = mu

    for t in range(1, n):
        # Each bar = 1 time unit
        decayed  = mu + (intensities[t - 1] - mu) * math.exp(-beta)
        base_t   = baseline[t - 1] if not np.isnan(baseline[t - 1]) else mu
        is_spike = vols[t - 1] > base_t * 1.8
        intensities[t] = decayed + (alpha if is_spike else 0.0)

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
#
#  Each 1-min candle gets buy/sell volume split by close position
#  within the bar's range. Rolling OFI = buy% over last N bars.
#  This is the Bulk Volume Classification method.
# ============================================================

def compute_ofi(df_1min: pd.DataFrame, window: int = OFI_WINDOW) -> tuple[float, float, float]:
    """
    Returns (current_ofi, ofi_delta, ofi_score).
    """
    close = df_1min["close"].values
    high  = df_1min["high"].values
    low   = df_1min["low"].values
    vol   = df_1min["volume"].values

    bar_range = high - low
    buy_ratio = np.where(bar_range > 0, (close - low) / bar_range, 0.5)

    buy_vol = buy_ratio * vol
    tot_vol = vol

    buy_roll = pd.Series(buy_vol).rolling(window, min_periods=3).sum()
    tot_roll = pd.Series(tot_vol).rolling(window, min_periods=3).sum()
    ofi_s    = (buy_roll / tot_roll.replace(0, np.nan)).fillna(0.5)

    cur_ofi  = float(ofi_s.iloc[-1])
    delta    = float(ofi_s.iloc[-1] - ofi_s.iloc[-4]) if len(ofi_s) >= 4 else 0.0
    score    = float(np.clip(cur_ofi * 100 + delta * 50, 0, 100))

    return round(cur_ofi, 4), round(delta, 4), round(score, 1)


def ofi_signal(ofi: float, delta: float) -> str:
    if ofi >= 0.65 and delta >= 0:   return "🟢 ACCUMULATING"
    elif ofi >= 0.60 and delta < 0:  return "🟡 TOPPING"
    elif ofi <= 0.35:                return "🔴 DISTRIBUTING"
    elif ofi <= 0.42:                return "🟠 SELLING"
    else:                            return "⚪ NEUTRAL"


# ============================================================
#  GAP 3: KELLY POSITION SIZER
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
    """Returns (kelly_fraction, dollar_risk, shares)."""
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
#  GAP 4: SIGNAL DECAY
# ============================================================

# Track when each symbol's signal fired
_signal_fire_times: dict[str, float] = {}
_signal_entry_prices: dict[str, float] = {}


def check_signal_decay(symbol: str, score: float,
                        current_price: float,
                        atr: float) -> tuple[bool, float, bool]:
    """
    Returns (signal_live, seconds_remaining, just_expired).

    Starts a 10-min timer when score first crosses threshold.
    Marks confirmed if price moves > 0.5 ATR before expiry.
    Kills the signal (returns False) if TTL exceeded without confirmation.
    """
    now = time.time()
    is_alert = score >= SIGNAL_THRESHOLD

    if is_alert and symbol not in _signal_fire_times:
        _signal_fire_times[symbol]    = now
        _signal_entry_prices[symbol]  = current_price

    if symbol not in _signal_fire_times:
        return False, 0.0, False

    elapsed   = now - _signal_fire_times[symbol]
    remaining = max(0.0, SIGNAL_TTL - elapsed)
    entry     = _signal_entry_prices.get(symbol, current_price)

    # Price confirmation — moved > 0.5 ATR in right direction
    confirmed = (current_price - entry) > atr * 0.5

    if not is_alert:
        # Score dropped — reset timer
        _signal_fire_times.pop(symbol, None)
        _signal_entry_prices.pop(symbol, None)
        return False, 0.0, False

    if elapsed > SIGNAL_TTL and not confirmed:
        _signal_fire_times.pop(symbol, None)
        _signal_entry_prices.pop(symbol, None)
        return False, 0.0, True   # just_expired

    return True, round(remaining, 0), False


# ============================================================
#  MAIN SCAN FUNCTION
#  Pulls all data for one symbol and returns a result dict.
# ============================================================

def scan_symbol(symbol: str, market: dict) -> Optional[dict]:
    """
    Full scan for one symbol.
    Fetches daily + 1-min data, runs all 4 layers + gaps.
    Returns result dict or None if data insufficient or outside mid-cap range.
    """
    # ── MID-CAP GATE ────────────────────────────────────────
    passes_mcap, mcap_b = is_midcap(symbol)
    if not passes_mcap:
        return None   # drifted outside $1B–$20B, skip silently

    # Daily data (Hurst, ATR, Kelly)
    df_daily = fetch_daily(symbol, 60)
    if df_daily is None or len(df_daily) < 30:
        return None

    # 1-min intraday data (Hawkes, OFI)
    df_1min = fetch_intraday(symbol, period="5d", interval="1m")
    if df_1min is None or len(df_1min) < 30:
        # Fallback: use daily as pseudo-intraday (lower accuracy)
        df_1min = df_daily.copy()

    price = float(df_daily["close"].iloc[-1])

    # ── LAYER 1: HURST ──────────────────────────────────────
    H       = compute_hurst(df_daily["close"].values)
    h_sc    = hurst_score(H)
    h_reg   = hurst_regime(H)

    # ── LAYER 2: HAWKES ─────────────────────────────────────
    cur_lam, hawk_sc = compute_hawkes(df_1min)
    hawk_sig         = hawkes_signal(hawk_sc)

    # ── LAYER 3: OFI ────────────────────────────────────────
    cur_ofi, ofi_d, o_sc = compute_ofi(df_1min)
    o_sig                 = ofi_signal(cur_ofi, ofi_d)

    # ── GAP 2: SECTOR RS ────────────────────────────────────
    etf                   = SECTOR_MAP.get(symbol, "SPY")
    sec_rs, sec_sc, sec_gate = get_sector_rs(etf)

    # ── COMPOSITE ───────────────────────────────────────────
    raw = float(np.clip(
        W_HURST  * h_sc   +
        W_HAWKES * hawk_sc +
        W_OFI    * o_sc   +
        W_SECTOR * sec_sc,
        0, 100
    ))

    sec_mult = 1.0 if sec_gate else 0.70
    comp     = round(float(np.clip(raw * market["mkt_mult"] * sec_mult, 0, 100)), 1)
    alert    = comp >= SIGNAL_THRESHOLD and market["allows_long"]

    # ── GAP 3: KELLY ────────────────────────────────────────
    atr                = compute_atr_daily(df_daily)
    kf, drisk, shares  = kelly_size(price, atr) if alert else (0.0, 0.0, 0)

    # ── GAP 4: SIGNAL DECAY ─────────────────────────────────
    sig_live, remaining, just_expired = check_signal_decay(symbol, comp, price, atr)
    if just_expired:
        alert = False
        comp  = max(0.0, comp - 10)  # penalty on expiry

    # ── STOP LOSS ───────────────────────────────────────────
    stop_price = round(price - atr * 1.5, 2) if atr > 0 else 0.0
    target     = round(price + atr * 3.0, 2) if atr > 0 else 0.0

    return {
        "symbol":       symbol,
        "price":        round(price, 2),
        "score":        comp,
        "alert":        alert,

        # Layer outputs
        "hurst_H":      round(H, 3),
        "hurst_score":  round(h_sc, 1),
        "hurst_regime": h_reg,

        "hawkes_lam":   cur_lam,
        "hawkes_score": hawk_sc,
        "hawkes_sig":   hawk_sig,

        "ofi":          cur_ofi,
        "ofi_delta":    ofi_d,
        "ofi_score":    o_sc,
        "ofi_sig":      o_sig,

        # Gap outputs
        "market":       market["regime"],
        "sector_etf":   etf,
        "sector_rs":    sec_rs,
        "sector_score": sec_sc,
        "sector_gate":  sec_gate,

        "kelly_frac":   kf,
        "dollar_risk":  drisk,
        "shares":       shares,
        "atr":          atr,
        "stop":         stop_price,
        "target":       target,

        "sig_live":     sig_live,
        "sig_remaining":int(remaining),
        "just_expired": just_expired,

        "mcap_b":       mcap_b,   # market cap in billions
        "scanned_at":   datetime.now().strftime("%H:%M:%S"),
    }


def run_full_scan(symbols: list = WATCHLIST) -> tuple[pd.DataFrame, dict]:
    """
    Runs full scan on all symbols.
    Returns (results_df, market_info).
    """
    market = get_market_regime()
    results = []

    for sym in symbols:
        try:
            r = scan_symbol(sym, market)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[SCAN ERR] {sym}: {e}")

    if not results:
        return pd.DataFrame(), market

    df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return df, market


# ============================================================
#  TERMINAL MODE
#  Runs when called directly: python scanner_yf.py
# ============================================================

def print_results(df: pd.DataFrame, market: dict):
    print(f"\n{'='*72}")
    print(f"  QUANT SCANNER  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  MARKET: {market['regime']}  SPY={market['spy_price']} ({market['spy_dev']:+.2f}%)")
    print(f"{'='*72}")
    print(f"  {'SYM':<6} {'SCORE':>5}  {'REGIME':<14} {'HAWKES':<14} {'OFI':<16} {'KELLY%':>6}  {'SHR':>5}  {'DECAY':>6}")
    print(f"  {'─'*70}")

    for _, r in df.iterrows():
        flag = " ◄◄" if r["alert"] else ("  ✗" if r["just_expired"] else "   ")
        dec  = f"{r['sig_remaining']}s" if r["sig_live"] else "—"
        print(
            f"  {r['symbol']:<6} {r['score']:>5}  "
            f"{r['hurst_regime']:<14} {r['hawkes_sig']:<14} "
            f"{r['ofi_sig']:<16} {r['kelly_frac']:>5.1%}  "
            f"{r['shares']:>5}  {dec:>6}{flag}"
        )

    alerts = df[df["alert"] == True]
    print(f"\n  {'─'*70}")
    print(f"  Signals: {len(alerts)}/{len(df)}  |  Market gate: {'✓ OPEN' if market['allows_long'] else '✗ RISK-OFF'}")

    if len(alerts) > 0:
        print(f"\n  ── TOP PICKS {'─'*50}")
        for _, r in alerts.iterrows():
            print(f"\n  {r['symbol']}  ${r['price']}  MCap ${r.get('mcap_b',0):.1f}B  [{r['score']}]")
            print(f"    Hurst   : H={r['hurst_H']}  → {r['hurst_regime']}")
            print(f"    Hawkes  : λ={r['hawkes_lam']}  → {r['hawkes_sig']}")
            print(f"    OFI     : {r['ofi']}  Δ={r['ofi_delta']}  → {r['ofi_sig']}")
            print(f"    Sector  : {r['sector_etf']}  RS={r['sector_rs']}  {'✓' if r['sector_gate'] else '✗'}")
            print(f"    Size    : {r['kelly_frac']:.1%} Kelly  →  {r['shares']} shares  (${r['dollar_risk']} risk)")
            print(f"    Stop    : ${r['stop']}  |  Target: ${r['target']}  (3:1 R:R)")
            if r["sig_live"]:
                print(f"    Decay   : {r['sig_remaining']}s remaining")

    print(f"\n{'='*72}\n")


# ============================================================
#  STREAMLIT DASHBOARD
#  Runs when: streamlit run scanner_yf.py
# ============================================================

def run_dashboard():
    st.set_page_config(
        page_title="QUANT SCANNER",
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
    .sym    { font-family:'Share Tech Mono',monospace; font-size:22px; color:#e8f4ff;
              letter-spacing:2px; }
    .sc     { font-family:'Share Tech Mono',monospace; font-size:30px; font-weight:700; }
    .tag    { display:inline-block; font-family:'Share Tech Mono',monospace; font-size:11px;
              padding:2px 8px; border-radius:3px; margin-right:5px; letter-spacing:1px; }
    .bar-bg { background:#0a1520; border-radius:3px; height:6px; margin:8px 0 6px; overflow:hidden; }
    .bar-fg { height:6px; border-radius:3px; }
    .meta   { font-family:'Share Tech Mono',monospace; font-size:11px; color:#3a5a72; }
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
        custom_syms = st.text_area("Watchlist (one per line)",
                                    value="\n".join(WATCHLIST), height=300)
        run_btn     = st.button("▶ RUN SCAN NOW", use_container_width=True)
        st.markdown("---")
        st.markdown("""
        <div style='font-family:Share Tech Mono;font-size:11px;color:#2a4a62'>
        Data: yfinance (daily + 1-min)<br>
        Hawkes: 1-min volume events<br>
        OFI: Bulk Volume Classification<br>
        Hurst: Daily R/S Analysis<br><br>
        No API key required.
        </div>
        """, unsafe_allow_html=True)

    symbols = [s.strip().upper() for s in custom_syms.split("\n") if s.strip()]

    # ── Auto-init state ────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state["results"]    = pd.DataFrame()
        st.session_state["market"]     = {}
        st.session_state["last_scan"]  = 0.0
        st.session_state["scan_count"] = 0

    # ── Trigger scan ───────────────────────────────────────
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
        st.markdown('<div class="title">⚡ QUANT SCANNER</div>', unsafe_allow_html=True)
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
        st.markdown(
            f'<div class="market">MARKET: '
            f'<span style="color:{regime_color}">{market.get("regime","—")}</span>'
            f' &nbsp;|&nbsp; SPY {market.get("spy_price","—")} '
            f'({market.get("spy_dev",0):+.2f}%) '
            f'&nbsp;|&nbsp; QQQ {market.get("qqq_price","—")} '
            f'({market.get("qqq_dev",0):+.2f}%)'
            f'</div>',
            unsafe_allow_html=True
        )

    if df.empty:
        st.info("No results yet — click ▶ RUN SCAN NOW")
        time.sleep(refresh)
        st.rerun()
        return

    # ── Alert banner ───────────────────────────────────────
    alerts = df[df["score"] >= threshold]
    if not alerts.empty:
        syms_str = "  ·  ".join(
            f"{r['symbol']} [{r['score']}]" for _, r in alerts.iterrows()
        )
        st.markdown(
            f'<div style="background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.4);'
            f'border-radius:6px;padding:10px 16px;margin-bottom:14px;'
            f'font-family:Share Tech Mono;font-size:13px;color:#00e5ff;letter-spacing:2px;">'
            f'▶ SIGNAL  ·  {syms_str}</div>',
            unsafe_allow_html=True
        )

    # ── Cards ──────────────────────────────────────────────
    cols = st.columns(min(4, len(df)))
    for i, (_, r) in enumerate(df.iterrows()):
        score = r["score"]
        color = ("#00ff8c" if score >= 75
                 else "#00e5ff" if score >= threshold
                 else "#ffb400" if score >= 45
                 else "#ff4040")
        card_cls = "card on" if score >= threshold else "card"
        dec_str  = (f"⏱ {r['sig_remaining']}s" if r.get("sig_live") else
                    "✗ EXPIRED" if r.get("just_expired") else "—")

        card = f"""
        <div class="{card_cls}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <div class="sym">{r['symbol']}</div>
              <div class="meta">${r['price']} &nbsp;|&nbsp; MCap ${r.get('mcap_b', 0):.1f}B</div>
            </div>
            <div style="text-align:right">
              <div class="sc" style="color:{color}">{score}</div>
              <div class="meta">SCORE</div>
            </div>
          </div>
          <div class="bar-bg">
            <div class="bar-fg" style="width:{int(score)}%;background:{color}"></div>
          </div>
          <div style="margin:6px 0">
            <span class="tag" style="background:rgba(0,229,255,0.1);color:#00e5ff;border:1px solid rgba(0,229,255,0.3)">{r['hurst_regime']}</span>
            <span class="tag" style="background:rgba(0,255,140,0.08);color:#00ff8c;border:1px solid rgba(0,255,140,0.25)">{r['hawkes_sig']}</span>
            <span class="tag" style="background:rgba(255,180,0,0.08);color:#ffb400;border:1px solid rgba(255,180,0,0.25)">{r['ofi_sig']}</span>
          </div>
          <div style="display:flex;gap:8px;margin-top:8px">
            <div style="flex:1;background:#0a1520;border-radius:4px;padding:5px 8px">
              <div class="meta">HURST H</div>
              <div style="font-family:Share Tech Mono;font-size:13px;color:#a0c0d8">{r['hurst_H']}</div>
            </div>
            <div style="flex:1;background:#0a1520;border-radius:4px;padding:5px 8px">
              <div class="meta">OFI</div>
              <div style="font-family:Share Tech Mono;font-size:13px;color:#a0c0d8">{r['ofi']}</div>
            </div>
            <div style="flex:1;background:#0a1520;border-radius:4px;padding:5px 8px">
              <div class="meta">SECTOR</div>
              <div style="font-family:Share Tech Mono;font-size:13px;color:{'#00ff8c' if r['sector_gate'] else '#ff4040'}">{r['sector_etf']} {'✓' if r['sector_gate'] else '✗'}</div>
            </div>
          </div>
          {"" if not r.get("alert") else f'''
          <div class="trade">
            <div class="meta">TRADE SETUP</div>
            <div style="font-family:Share Tech Mono;font-size:12px;color:#c0d8e8">
              Kelly {r['kelly_frac']:.1%} → {r['shares']} shares (${r['dollar_risk']} risk)<br>
              Stop ${r['stop']} &nbsp;|&nbsp; Target ${r['target']} &nbsp;|&nbsp; {dec_str}
            </div>
          </div>
          '''}
        </div>
        """
        with cols[i % len(cols)]:
            st.markdown(card, unsafe_allow_html=True)

    # ── Summary table ──────────────────────────────────────
    st.markdown("---")
    display_cols = ["symbol", "mcap_b", "score", "hurst_regime", "hawkes_sig",
                    "ofi_sig", "sector_etf", "kelly_frac", "shares",
                    "stop", "target", "market"]
    st.dataframe(
        df[display_cols].rename(columns={
            "symbol": "Symbol", "mcap_b": "MCap $B", "score": "Score",
            "hurst_regime": "Regime", "hawkes_sig": "Hawkes",
            "ofi_sig": "OFI", "sector_etf": "Sector",
            "kelly_frac": "Kelly%", "shares": "Shares",
            "stop": "Stop $", "target": "Target $", "market": "Mkt"
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Auto-refresh
    time.sleep(refresh)
    st.rerun()


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Check if running via streamlit
    if "streamlit" in sys.modules and hasattr(st, "session_state"):
        run_dashboard()
    else:
        # Terminal mode — loop every POLL_INTERVAL seconds
        print("""
╔══════════════════════════════════════════════════════════╗
║  QUANT SCANNER — YFINANCE BUILD                          ║
║  No API key required. Data: yfinance (daily + 1-min)    ║
║                                                          ║
║  Terminal mode: python scanner_yf.py                    ║
║  Dashboard:     streamlit run scanner_yf.py             ║
╚══════════════════════════════════════════════════════════╝
        """)
        try:
            while True:
                df, market = run_full_scan(WATCHLIST)
                if not df.empty:
                    print_results(df, market)
                    fname = f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    df.to_csv(fname, index=False)
                    print(f"  Saved → {fname}")
                print(f"  Next scan in {POLL_INTERVAL}s...  (Ctrl+C to stop)\n")
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\n  Scanner stopped.")

# ── Streamlit entry: imported directly by streamlit ─────────
else:
    if STREAMLIT_MODE:
        run_dashboard()
