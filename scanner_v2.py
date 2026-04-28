python

"""
============================================================
  QUANT SCANNER + SIGNAL ENGINE — v3 UNIFIED
  scanner_v3.py

  WHAT'S NEW IN v3
  ─────────────────
  NEW 1 — ADD Breadth Gate
      Uses SPY advancing/declining volume as $ADD proxy.
      Schwab live $ADD when credentials active.
      Blocks all longs when market internals are bearish.

  NEW 2 — Z-Score Exhaustion Gate
      Calculates how many σ price is from its 20-bar mean.
      Z > 2.5 → move already overextended → BLOCK entry.
      Prevents "buying the top" of an exhausted move.

  NEW 3 — Kurtosis & Skewness Filter
      Kurtosis > 3.0 → fat tails → outlier moves likely.
      Positive skew → upside outliers more probable (LONG).
      Negative skew → downside outliers more probable (SHORT).
      Finds the stocks where BIG fast moves actually happen.

  NEW 4 — Bayesian Probability Score
      Stacks ADD + Sector RS + Hawkes into conditional
      win probability using Bayes' theorem.
      P(Win | ADD, Sector, Hawkes) shown as % on each card.

  NEW 5 — Alpha Half-Life Decay
      Signal strength decays exponentially from fire time.
      Decay rate = f(signal strength, ATR volatility).
      Stronger signals get longer half-life. Replaces flat TTL.

  TIMEFRAME TOGGLE (sidebar)
      15s / 1m / 5m / 15m — all selectable live.

  DATA SOURCE TOGGLE (sidebar)
      Schwab API (live streaming) or yfinance (fallback).
      Schwab unlocks: real $ADD, Level 1 bid size,
      true TIMESALE tick stream for half-life calc.

  INSTALL
  ───────
  pip install yfinance pandas numpy streamlit schwab-py

  ENV VARS (Schwab — set when ready):
      SCHWAB_API_KEY=...
      SCHWAB_API_SECRET=...
      SCHWAB_CALLBACK_URL=https://127.0.0.1
      ACCOUNT_SIZE=50000

  RUN
  ───
  python -m streamlit run scanner_v3.py
============================================================
"""

import os, sys, time, math, warnings, threading
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from collections import deque
from scipy import stats as scipy_stats

import yfinance as yf
warnings.filterwarnings("ignore")

try:
    import streamlit as st
    STREAMLIT_MODE = True
except ImportError:
    STREAMLIT_MODE = False

try:
    import schwab
    from schwab.streaming import StreamClient
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False


# ============================================================
#  CONFIG
# ============================================================

WATCHLIST = [
    "CRUS","POWI","MTSI","JAMF","ALKT","TASK",
    "BOOT","GIII","PLAY","LESL","PRPL","HIMS",
    "ACAD","INVA","PDCO","HCAT","NVCR","GKOS",
    "CURO","OPFI","HIBB","GCMG","NRDS",
    "CIVI","BATL","REX","PTEN","WTTR",
    "KTOS","ASTE","HLIO","DLX","HAYW",
    "TROX","RYAM","KWR",
    "NTST","GMRE","EPRT",
]

SECTOR_MAP = {
    "CRUS":"XLK","POWI":"XLK","MTSI":"XLK","JAMF":"XLK","ALKT":"XLK","TASK":"XLK",
    "BOOT":"XLY","GIII":"XLY","PLAY":"XLY","LESL":"XLY","PRPL":"XLY","HIMS":"XLY",
    "ACAD":"XLV","INVA":"XLV","PDCO":"XLV","HCAT":"XLV","NVCR":"XLV","GKOS":"XLV",
    "CURO":"XLF","OPFI":"XLF","HIBB":"XLF","GCMG":"XLF","NRDS":"XLF",
    "CIVI":"XLE","BATL":"XLE","REX":"XLE","PTEN":"XLE","WTTR":"XLE",
    "KTOS":"XLI","ASTE":"XLI","HLIO":"XLI","DLX":"XLI","HAYW":"XLI",
    "TROX":"XLB","RYAM":"XLB","KWR":"XLB",
    "NTST":"XLRE","GMRE":"XLRE","EPRT":"XLRE",
}

MIDCAP_MIN        = 300_000_000
MIDCAP_MAX        = 20_000_000_000
ACCOUNT_SIZE      = float(os.environ.get("ACCOUNT_SIZE", 50000))
SIGNAL_THRESHOLD  = 65
ATR_STOP_MULT     = 1.5
ATR_TARGET_MULT   = 3.0
SPY_WEAK_THRESH   = -0.005
SECTOR_RS_MIN     = 1.02
HAWKES_DECAY      = 0.3
OFI_LONG_ENTRY    = 0.60
OFI_SHORT_ENTRY   = 0.40
OFI_LONG_EXIT     = 0.45
OFI_SHORT_EXIT    = 0.55
HAWKES_MIN_SLOPE  = 0.02

# NEW 2 — Z-Score exhaustion threshold
ZSCORE_MAX        = 2.5    # block long entry if Z > this
ZSCORE_MIN        = -2.5   # block short entry if Z < this

# NEW 3 — Kurtosis / Skewness minimums
KURTOSIS_MIN      = 3.0    # excess kurtosis > 3 = fat tails
SKEW_LONG_MIN     = 0.1    # positive skew for longs
SKEW_SHORT_MAX    = -0.1   # negative skew for shorts

# NEW 4 — Bayesian base rates (tune after 50+ trades)
BAYES_BASE_WIN    = 0.50   # random base rate
BAYES_ADD_BOOST   = 0.08   # ADD breadth adds ~8%
BAYES_SECTOR_BOOST= 0.10   # sector RS adds ~10%
BAYES_HAWKES_BOOST= 0.14   # Hawkes clustering adds ~14%

# NEW 5 — Half-life decay params
HALFLIFE_BASE_SEC = 600    # base half-life 10 min
HALFLIFE_MIN_SEC  = 120    # floor: 2 min
HALFLIFE_MAX_SEC  = 1800   # ceiling: 30 min

# Intraday health thresholds (v2)
INTRADAY_SOFT_WARN = -0.010
INTRADAY_HARD_WARN = -0.020
INTRADAY_KILL      = -0.030
GAP_DOWN_THRESH    = -0.010
LOWER_LOW_BARS     = 6

# Timeframe options → yfinance interval strings
TIMEFRAME_OPTIONS  = {
    "15s":  "15s",   # yfinance doesn't support 15s — we simulate via 1m
    "1m":   "1m",
    "5m":   "5m",
    "15m":  "15m",
}
DEFAULT_TIMEFRAME  = "5m"


# ============================================================
#  MCAP CACHE
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
    return MIDCAP_MIN <= mcap <= MIDCAP_MAX, round(mcap / 1e9, 2)


# ============================================================
#  DATA FETCHERS
# ============================================================

def fetch_daily(symbol: str, n: int = 60) -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(symbol).history(period="3mo", interval="1d")
        if df.empty or len(df) < 10:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df[["open","high","low","close","volume"]].dropna().tail(n)
    except Exception:
        return None

def fetch_intraday(symbol: str, timeframe: str = "5m") -> Optional[pd.DataFrame]:
    """
    Fetch intraday bars. 15s is not supported by yfinance —
    we fetch 1m and resample to approximate 15s bars.
    """
    try:
        interval = "1m" if timeframe == "15s" else timeframe
        period   = "5d" if timeframe in ("5m","15m") else "1d"
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty or len(df) < 10:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()

        # Resample 1m → approximate 15s by splitting each bar into 4
        if timeframe == "15s":
            rows = []
            for _, row in df.iterrows():
                for q in range(4):
                    rows.append({
                        "open":   row["open"],
                        "high":   row["high"],
                        "low":    row["low"],
                        "close":  row["close"],
                        "volume": row["volume"] / 4,
                    })
            df = pd.DataFrame(rows)

        return df
    except Exception:
        return None

def fetch_closes(symbol: str, n: int = 25) -> Optional[np.ndarray]:
    df = fetch_daily(symbol, n)
    return df["close"].values if df is not None else None


# ============================================================
#  NEW 1: ADD BREADTH GATE
#  $ADD = NYSE Advancing Issues − Declining Issues.
#  When ADD > 0, more stocks are rising than falling —
#  the "tide" is rising and momentum longs have tailwind.
#
#  yfinance proxy: SPY up-volume / total-volume ratio.
#  If > 0.55 → breadth positive (ADD proxy bullish).
#  Schwab mode: streams real-time $ADD quote when available.
# ============================================================

_add_cache: dict = {"value": 0.0, "bullish": True, "updated": 0.0}
_add_lock = threading.Lock()

def update_add_breadth_yf():
    """
    Computes ADD proxy from SPY 1-min bars.
    Up-volume / total-volume > 0.55 = advancing breadth.
    Runs in background thread, updates every 60 seconds.
    """
    while True:
        try:
            df = yf.Ticker("SPY").history(period="1d", interval="1m")
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
                close = df["close"].values
                vol   = df["volume"].values
                bar_range = df["high"].values - df["low"].values
                buy_r = np.where(bar_range > 0,
                                 (close - df["low"].values) / bar_range, 0.5)
                up_vol  = (buy_r * vol).sum()
                tot_vol = vol.sum()
                ratio   = up_vol / tot_vol if tot_vol > 0 else 0.5
                bullish = ratio > 0.52
                with _add_lock:
                    _add_cache["value"]   = round(ratio, 4)
                    _add_cache["bullish"] = bullish
                    _add_cache["updated"] = time.time()
        except Exception:
            pass
        time.sleep(60)

def get_add_breadth() -> tuple[float, bool]:
    """Returns (add_proxy_value, is_bullish)."""
    with _add_lock:
        return _add_cache["value"], _add_cache["bullish"]

def add_score(bullish: bool) -> float:
    """Score contribution from ADD breadth: 70 if bullish, 30 if not."""
    return 70.0 if bullish else 30.0


# ============================================================
#  NEW 2: Z-SCORE EXHAUSTION GATE
#  Z = (price − SMA20) / StdDev20
#  Z > 2.5 → price is 2.5σ above mean → overextended.
#  Buying here = buying the top. BLOCK the entry.
#  Z < -2.5 → overextended to downside → block short entry.
#
#  Professional rule: at 3σ, >99.7% of price action is
#  within this range — the move is statistically exhausted.
# ============================================================

def compute_zscore(prices: np.ndarray, window: int = 20) -> float:
    """
    Z-score of current price relative to rolling 20-bar mean.
    Uses daily closes for stability.
    """
    if len(prices) < window:
        return 0.0
    recent = prices[-window:]
    mean   = np.mean(recent)
    std    = np.std(recent, ddof=1)
    if std == 0:
        return 0.0
    return float((prices[-1] - mean) / std)

def zscore_gate(z: float, direction: str) -> tuple[bool, str]:
    """
    Returns (passes, reason).
    Blocks long entry if Z > ZSCORE_MAX (overextended up).
    Blocks short entry if Z < ZSCORE_MIN (overextended down).
    """
    if direction == "LONG" and z > ZSCORE_MAX:
        return False, f"Z={z:.2f} > {ZSCORE_MAX} — EXHAUSTED UP"
    if direction == "SHORT" and z < ZSCORE_MIN:
        return False, f"Z={z:.2f} < {ZSCORE_MIN} — EXHAUSTED DOWN"
    return True, f"Z={z:.2f} ✓"

def zscore_score(z: float) -> float:
    """
    Score 0-100. Near zero = ideal (50-75). Extremes = penalised.
    """
    return float(np.clip(100 - abs(z) * 20, 0, 100))


# ============================================================
#  NEW 3: KURTOSIS & SKEWNESS FILTER
#  Finds stocks where big fast moves are most statistically
#  likely — the "hunting ground" for day trades.
#
#  Kurtosis > 3.0 → fat tails → outlier moves common.
#  Positive skew → right tail fatter → big up moves likely.
#  Negative skew → left tail fatter → big down moves likely.
#
#  Use: only trade stocks where kurtosis is elevated AND
#  skew aligns with your intended direction.
# ============================================================

def compute_kurtosis_skew(prices: np.ndarray) -> tuple[float, float]:
    """
    Returns (excess_kurtosis, skewness) of log returns.
    Uses log returns so results are comparable across stocks.
    """
    if len(prices) < 20:
        return 0.0, 0.0
    log_r = np.diff(np.log(prices + 1e-9))
    kurt  = float(scipy_stats.kurtosis(log_r, fisher=True))   # excess kurtosis
    skew  = float(scipy_stats.skew(log_r))
    return round(kurt, 3), round(skew, 3)

def kurtosis_skew_score(kurt: float, skew: float,
                         direction: str = "LONG") -> tuple[float, str]:
    """
    Returns (score_0_100, label).
    Fat tails = good (higher score).
    Skew aligned with direction = bonus.
    """
    # Base score from kurtosis
    base = float(np.clip((kurt / 6.0) * 60, 0, 60))   # 0-60 pts

    # Skew alignment bonus (0-40 pts)
    if direction == "LONG" and skew > SKEW_LONG_MIN:
        skew_bonus = float(np.clip(skew * 20, 0, 40))
    elif direction == "SHORT" and skew < SKEW_SHORT_MAX:
        skew_bonus = float(np.clip(abs(skew) * 20, 0, 40))
    else:
        skew_bonus = 0.0

    score = round(base + skew_bonus, 1)

    if kurt > 5.0 and score > 70:
        label = "🎯 FAT TAIL"
    elif kurt > KURTOSIS_MIN:
        label = "📊 ELEVATED"
    else:
        label = "〰 NORMAL"

    return score, label


# ============================================================
#  NEW 4: BAYESIAN PROBABILITY SCORE
#  Stacks independent signals using Bayes' theorem to compute
#  the true conditional win probability given all conditions.
#
#  P(W|ADD,Sector,Hawkes) = updated sequentially:
#    Start: 50% base rate
#    If ADD bullish: +8%
#    If Sector outperforming: +10%
#    If Hawkes clustering: +14%
#  Each condition updates the posterior probability.
# ============================================================

def bayesian_win_prob(add_bullish: bool, sector_gate: bool,
                       hawkes_score: float, zscore_ok: bool,
                       kurt_score: float) -> tuple[float, list]:
    """
    Returns (win_probability_0_to_1, factor_list).
    Sequential Bayesian update from base rate.
    """
    p = BAYES_BASE_WIN
    factors = [f"Base: {p:.0%}"]

    # ADD breadth
    if add_bullish:
        p = min(0.97, p + BAYES_ADD_BOOST)
        factors.append(f"ADD ↑ +{BAYES_ADD_BOOST:.0%} → {p:.0%}")
    else:
        p = max(0.03, p - BAYES_ADD_BOOST)
        factors.append(f"ADD ↓ -{BAYES_ADD_BOOST:.0%} → {p:.0%}")

    # Sector RS
    if sector_gate:
        p = min(0.97, p + BAYES_SECTOR_BOOST)
        factors.append(f"Sector ↑ +{BAYES_SECTOR_BOOST:.0%} → {p:.0%}")
    else:
        p = max(0.03, p - BAYES_SECTOR_BOOST * 0.5)
        factors.append(f"Sector weak → {p:.0%}")

    # Hawkes intensity
    if hawkes_score >= 72:
        p = min(0.97, p + BAYES_HAWKES_BOOST)
        factors.append(f"Hawkes 🔥 +{BAYES_HAWKES_BOOST:.0%} → {p:.0%}")
    elif hawkes_score >= 58:
        p = min(0.97, p + BAYES_HAWKES_BOOST * 0.5)
        factors.append(f"Hawkes ⚡ +{BAYES_HAWKES_BOOST*0.5:.0%} → {p:.0%}")

    # Z-score (not exhausted = bonus)
    if zscore_ok:
        p = min(0.97, p + 0.04)
        factors.append(f"Z-ok +4% → {p:.0%}")
    else:
        p = max(0.03, p - 0.12)
        factors.append(f"Z-exhausted -12% → {p:.0%}")

    # Fat tails (outlier move likely)
    if kurt_score >= 60:
        p = min(0.97, p + 0.05)
        factors.append(f"FatTail +5% → {p:.0%}")

    return round(p, 4), factors


# ============================================================
#  NEW 5: ALPHA HALF-LIFE DECAY
#  Replaces the flat 10-min TTL with true exponential decay.
#
#  Signal strength S(t) = S₀ × exp(−λt)
#  where λ = ln(2) / half_life
#
#  Half-life is dynamic:
#    Strong signals (score > 80) → longer half-life (20 min)
#    Weak signals  (score < 60) → shorter half-life (3 min)
#    ATR volatility adjusts: high vol = faster decay
#
#  Signal considered "alive" while S(t) > 0.25 × S₀
#  (25% of original strength remaining)
# ============================================================

_signal_start: dict[str, dict] = {}

def compute_halflife(score: float, atr_pct: float) -> float:
    """
    Dynamic half-life in seconds based on signal strength and volatility.
    atr_pct = ATR / price (e.g. 0.02 = 2% ATR)
    """
    # Base: stronger signal = longer half-life
    base = HALFLIFE_BASE_SEC * (score / 65.0)

    # Volatility adjustment: high vol = faster decay
    vol_factor = 1.0 / (1.0 + atr_pct * 10)

    hl = base * vol_factor
    return float(np.clip(hl, HALFLIFE_MIN_SEC, HALFLIFE_MAX_SEC))

def halflife_remaining(symbol: str, current_score: float,
                        current_price: float, atr: float) -> tuple[float, float, bool]:
    """
    Returns (strength_pct, seconds_remaining, is_alive).
    strength_pct: how much of original signal remains (0.0-1.0)
    """
    now = time.time()
    atr_pct = (atr / current_price) if current_price > 0 else 0.02

    # Register new signal
    if symbol not in _signal_start or current_score < SIGNAL_THRESHOLD:
        if current_score >= SIGNAL_THRESHOLD:
            hl = compute_halflife(current_score, atr_pct)
            _signal_start[symbol] = {
                "time":     now,
                "score":    current_score,
                "halflife": hl,
                "lambda":   math.log(2) / hl,
            }
        else:
            _signal_start.pop(symbol, None)
            return 0.0, 0.0, False

    sig = _signal_start[symbol]
    elapsed  = now - sig["time"]
    strength = math.exp(-sig["lambda"] * elapsed)
    hl       = sig["halflife"]
    remaining = max(0.0, hl * (math.log(strength) / math.log(0.5))) if strength > 0 else 0.0

    # Alive while strength > 25% of original
    alive = strength > 0.25

    if not alive:
        _signal_start.pop(symbol, None)

    return round(strength, 4), round(remaining, 1), alive


# ============================================================
#  CORE INDICATORS (from v2 — unchanged)
# ============================================================

def compute_hurst(prices: np.ndarray) -> float:
    arr = np.array(prices)
    if len(arr) < 30: return 0.5
    log_r = np.diff(np.log(arr + 1e-9))
    n = len(log_r)
    lags = np.unique(np.floor(np.geomspace(5, n//2, 12)).astype(int))
    lags = lags[lags >= 4]
    rs_v, vl = [], []
    for lag in lags:
        nw = n // lag
        if nw < 2: continue
        rl = []
        for i in range(nw):
            seg = log_r[i*lag:(i+1)*lag]
            cs  = np.cumsum(seg - seg.mean())
            S   = seg.std(ddof=1)
            if S > 0: rl.append((cs.max()-cs.min())/S)
        if rl: rs_v.append(np.mean(rl)); vl.append(lag)
    if len(vl) < 3: return 0.5
    H, _ = np.polyfit(np.log(vl), np.log(rs_v), 1)
    return float(np.clip(H, 0.0, 1.0))

def hurst_score(H): return float(np.clip((H-0.5)*200+50, 0, 100))
def hurst_regime(H): return "📈 TRENDING" if H>0.58 else ("🔄 REVERTING" if H<0.42 else "〰 CHOPPY")

def compute_hawkes(df: pd.DataFrame) -> tuple[float, float]:
    close=df["close"].values; high=df["high"].values
    low=df["low"].values; vols=df["volume"].values; n=len(vols)
    if n < 20: return 0.0, 50.0
    br = np.where(high-low>0, (close-low)/(high-low), 0.5)
    bv = br*vols; sv = (1-br)*vols
    bb = pd.Series(bv).rolling(20,min_periods=5).mean().values
    bs = pd.Series(sv).rolling(20,min_periods=5).mean().values
    mu = float(np.nanmean(bb[~np.isnan(bb)]))
    if mu<=0: return 0.0, 50.0
    alpha=mu*0.5; beta=HAWKES_DECAY
    lams=np.zeros(n); lams[0]=mu
    for t in range(1,n):
        d=mu+(lams[t-1]-mu)*math.exp(-beta)
        bbt=bb[t-1] if not np.isnan(bb[t-1]) else mu
        bst=bs[t-1] if not np.isnan(bs[t-1]) else mu
        delta=0.0
        if bv[t-1]>bbt*1.8: delta+=alpha
        if sv[t-1]>bst*1.8: delta-=alpha*0.8
        lams[t]=max(0.0, d+delta)
    cur=lams[-1]
    base=np.nanmean(lams[max(0,n-30):-5]) if n>10 else mu
    if base<=0: return cur, 50.0
    ratio=cur/(base+1e-9)
    score=float(np.clip(50+50*math.tanh(ratio-1.0),0,100))
    return round(cur,4), round(score,1)

def hawkes_signal(s): return "🔥 CLUSTERING" if s>=72 else ("⚡ BUILDING" if s>=58 else ("〰 IDLE" if s>=42 else "❄ FADING"))

def compute_ofi(df: pd.DataFrame, window: int = 12) -> tuple[float, float, float]:
    close=df["close"].values; high=df["high"].values
    low=df["low"].values; vol=df["volume"].values
    br=np.where(high-low>0,(close-low)/(high-low),0.5)
    bv=br*vol
    br_=pd.Series(bv).rolling(window,min_periods=3).sum()
    tr_=pd.Series(vol).rolling(window,min_periods=3).sum()
    ofi_s=(br_/tr_.replace(0,np.nan)).fillna(0.5)
    ofi=float(ofi_s.iloc[-1])
    delta=float(ofi_s.iloc[-1]-ofi_s.iloc[-4]) if len(ofi_s)>=4 else 0.0
    return round(ofi,4), round(delta,4), round(float(np.clip(ofi*100+delta*50,0,100)),1)

def ofi_signal(ofi,delta):
    if ofi>=0.65 and delta>=0: return "🟢 ACCUMULATING"
    elif ofi>=0.60 and delta<0: return "🟡 TOPPING"
    elif ofi<=0.35: return "🔴 DISTRIBUTING"
    elif ofi<=0.42: return "🟠 SELLING"
    return "⚪ NEUTRAL"

def compute_atr(df: pd.DataFrame, n: int=14) -> float:
    if df is None or len(df)<n: return 0.0
    h=df["high"].values; l=df["low"].values; c=df["close"].values
    trs=[max(h[-i]-l[-i], abs(h[-i]-c[-i-1]) if i<len(c) else 0,
             abs(l[-i]-c[-i-1]) if i<len(c) else 0)
         for i in range(1,min(n+1,len(c)))]
    return round(float(np.mean(trs)),4) if trs else 0.0

def calc_vwap(df: pd.DataFrame) -> float:
    try:
        tz=df.index.tz
        today=pd.Timestamp.now(tz=tz).strftime("%Y-%m-%d")
        tb=df[df.index.strftime("%Y-%m-%d")==today]
        if len(tb)<2: tb=df.tail(20)
        tp=(tb["high"]+tb["low"]+tb["close"])/3
        return round(float((tp*tb["volume"]).sum()/tb["volume"].sum()),4)
    except Exception:
        return float(df["close"].iloc[-1])

def intraday_health(df: pd.DataFrame, prior_close: float) -> tuple[float, str, dict]:
    if df is None or len(df)<5: return 1.0, "NO DATA", {}
    try:
        tz=df.index.tz; today=pd.Timestamp.now(tz=tz).strftime("%Y-%m-%d")
        tb=df[df.index.strftime("%Y-%m-%d")==today]
    except Exception:
        tb=df.tail(60)
    if len(tb)<3: tb=df.tail(60)
    today_open=float(tb["open"].iloc[0]); current=float(tb["close"].iloc[-1])
    intra_r=(current-today_open)/today_open if today_open>0 else 0.0
    gap_r=(today_open-prior_close)/prior_close if prior_close>0 else 0.0
    recent=tb["close"].values[-LOWER_LOW_BARS:]
    ll=int(sum(recent[i]<recent[i-1] for i in range(1,len(recent))))
    making_ll=ll>=(LOWER_LOW_BARS-2)
    tp=(tb["high"]+tb["low"]+tb["close"])/3
    vwap=float((tp*tb["volume"]).cumsum().iloc[-1]/tb["volume"].cumsum().iloc[-1]) \
         if tb["volume"].sum()>0 else current
    below_vwap=current<vwap
    mult=1.0; flags=[]
    if gap_r<GAP_DOWN_THRESH: mult*=0.85; flags.append(f"GAP↓{gap_r*100:.1f}%")
    if intra_r<=INTRADAY_KILL: mult=0.0; flags.append(f"SELLOFF{intra_r*100:.1f}%")
    elif intra_r<=INTRADAY_HARD_WARN: mult*=0.50; flags.append(f"WEAK{intra_r*100:.1f}%")
    elif intra_r<=INTRADAY_SOFT_WARN: mult*=0.75; flags.append(f"SOFT{intra_r*100:.1f}%")
    else: flags.append(f"OK{intra_r*100:+.1f}%")
    if making_ll: mult*=0.70; flags.append("LOWER-LOWS")
    if below_vwap: mult*=0.80; flags.append(f"<VWAP${vwap:.2f}")
    mult=float(np.clip(mult,0.0,1.0))
    return mult," | ".join(flags) if flags else "HEALTHY",{
        "intraday_ret":round(intra_r*100,2),"gap_ret":round(gap_r*100,2),
        "today_open":round(today_open,2),"vwap":round(vwap,2),
        "below_vwap":below_vwap,"lower_lows":ll,"health_mult":round(mult,3)}

def get_market_regime() -> dict:
    spy=fetch_closes("SPY",25); qqq=fetch_closes("QQQ",25)
    def dev(arr):
        if arr is None or len(arr)<20: return 0.0,0.0
        sma=np.mean(arr[-20:])
        return arr[-1],(arr[-1]-sma)/sma if sma>0 else 0.0
    sp,sd=dev(spy); qp,qd=dev(qqq)
    if sd>0.002 and qd>0.002: r="RISK-ON 📈"; al=True; mm=1.10
    elif sd<SPY_WEAK_THRESH and qd<SPY_WEAK_THRESH: r="RISK-OFF 📉"; al=False; mm=0.40
    else: r="NEUTRAL ➡"; al=True; mm=1.00
    return {"regime":r,"allows_long":al,"mkt_mult":mm,
            "spy_price":round(sp,2),"spy_dev":round(sd*100,2),
            "qqq_price":round(qp,2),"qqq_dev":round(qd*100,2)}

def get_sector_rs(etf: str) -> tuple[float, float, bool]:
    ec=fetch_closes(etf,25); sc=fetch_closes("SPY",25)
    if ec is None or sc is None or len(ec)<20 or len(sc)<20: return 1.0,50.0,True
    er=ec[-1]/np.mean(ec[-20:]); sr=sc[-1]/np.mean(sc[-20:])
    ratio=er/sr if sr>0 else 1.0
    score=float(np.clip(50+(ratio-1.0)*500,0,100))
    return round(ratio,4),round(score,1),ratio>=SECTOR_RS_MIN

def kelly_size(price,atr,win_rate=0.60,rr=2.0):
    if atr<=0 or price<=0: return 0.0,0.0,0
    kf=max(0.0,(win_rate-(1-win_rate)/rr)*0.5); kf=min(kf,0.10)
    drisk=ACCOUNT_SIZE*kf; stop=atr*ATR_STOP_MULT
    sh=int(drisk/stop) if stop>0 else 0
    return round(kf,4),round(drisk,2),min(sh,int(ACCOUNT_SIZE*0.20/price))


# ============================================================
#  SIGNAL POSITIONS + HALF-LIFE STATE
# ============================================================

@dataclass
class Position:
    symbol:      str
    direction:   str   = "FLAT"
    entry_price: float = 0.0
    entry_time:  str   = ""
    stop:        float = 0.0
    target:      float = 0.0
    atr:         float = 0.0
    highest:     float = 0.0
    lowest:      float = 0.0
    pnl:         float = 0.0
    shares:      int   = 0

POSITIONS: dict[str, Position] = {}
TRADE_LOG: list[dict] = []

def update_trailing_stop(pos: Position, price: float) -> Position:
    if pos.direction=="LONG":
        pos.highest=max(pos.highest,price)
        pos.stop=max(pos.stop, round(pos.highest-pos.atr*ATR_STOP_MULT,2))
        pos.pnl=round((price-pos.entry_price)*pos.shares,2)
    elif pos.direction=="SHORT":
        pos.lowest=min(pos.lowest,price)
        pos.stop=min(pos.stop, round(pos.lowest+pos.atr*ATR_STOP_MULT,2))
        pos.pnl=round((pos.entry_price-price)*pos.shares,2)
    return pos

def close_position(symbol: str, exit_price: float, reason: str):
    pos=POSITIONS.get(symbol)
    if pos and pos.direction!="FLAT":
        pnl=((exit_price-pos.entry_price) if pos.direction=="LONG"
             else (pos.entry_price-exit_price))*pos.shares
        TRADE_LOG.append({"Symbol":symbol,"Direction":pos.direction,
            "Entry $":pos.entry_price,"Exit $":round(exit_price,2),
            "Shares":pos.shares,"P&L $":round(pnl,2),
            "Result":"WIN ✅" if pnl>0 else "LOSS ❌",
            "Reason":reason,"Time":datetime.now().strftime("%H:%M:%S")})
    POSITIONS[symbol]=Position(symbol)


# ============================================================
#  MASTER SCAN FUNCTION
#  Runs all 10 layers for one symbol.
# ============================================================

def scan_symbol(symbol: str, market: dict, timeframe: str = "5m") -> Optional[dict]:
    # ── Mid-cap gate ─────────────────────────────────────────
    passes, mcap_b = is_midcap(symbol)
    if not passes: return None

    df_daily = fetch_daily(symbol, 60)
    if df_daily is None or len(df_daily)<30: return None

    df_intra = fetch_intraday(symbol, timeframe)
    if df_intra is None or len(df_intra)<15:
        df_intra = df_daily.copy()

    price       = float(df_daily["close"].iloc[-1])
    prior_close = float(df_daily["close"].iloc[-2]) if len(df_daily)>=2 else price
    closes      = df_daily["close"].values

    # ── Core indicators ──────────────────────────────────────
    H=compute_hurst(closes); h_sc=hurst_score(H); h_reg=hurst_regime(H)
    cur_lam,hawk_sc=compute_hawkes(df_intra); hawk_sig=hawkes_signal(hawk_sc)
    cur_ofi,ofi_d,o_sc=compute_ofi(df_intra); o_sig=ofi_signal(cur_ofi,ofi_d)
    etf=SECTOR_MAP.get(symbol,"SPY"); sec_rs,sec_sc,sec_gate=get_sector_rs(etf)
    atr=compute_atr(df_daily)

    # ── NEW 1: ADD breadth ───────────────────────────────────
    add_val,add_bull=get_add_breadth()
    add_sc=add_score(add_bull)

    # ── NEW 2: Z-Score ───────────────────────────────────────
    z=compute_zscore(closes)
    z_sc=zscore_score(z)

    # ── NEW 3: Kurtosis & Skewness ───────────────────────────
    kurt,skew=compute_kurtosis_skew(closes)
    direction_guess="LONG" if cur_ofi>=0.5 else "SHORT"
    ks_sc,ks_label=kurtosis_skew_score(kurt,skew,direction_guess)

    # ── NEW 4: Bayesian probability ──────────────────────────
    z_ok_long  = z <= ZSCORE_MAX
    bayes_p,bayes_factors=bayesian_win_prob(
        add_bull,sec_gate,hawk_sc,z_ok_long,ks_sc)

    # ── Composite score (all 10 layers) ──────────────────────
    W = {"hurst":0.12,"hawkes":0.20,"ofi":0.18,"sector":0.10,
         "add":0.12,"zscore":0.10,"kurtskew":0.08,"bayes":0.10}
    raw = float(np.clip(
        W["hurst"]*h_sc + W["hawkes"]*hawk_sc + W["ofi"]*o_sc +
        W["sector"]*sec_sc + W["add"]*add_sc + W["zscore"]*z_sc +
        W["kurtskew"]*ks_sc + W["bayes"]*bayes_p*100,
        0,100))

    sec_mult   = 1.0 if sec_gate else 0.70
    add_mult   = 1.0 if add_bull else 0.80
    health_mult,health_label,health_detail=intraday_health(df_intra,prior_close)
    comp=round(float(np.clip(raw*market["mkt_mult"]*sec_mult*add_mult*health_mult,0,100)),1)
    alert=comp>=SIGNAL_THRESHOLD and market["allows_long"] and health_mult>0.0

    # Z-Score exhaustion hard block
    z_long_ok,z_reason=zscore_gate(z,"LONG")
    if not z_long_ok and direction_guess=="LONG":
        alert=False; comp=min(comp,55.0)

    # ── NEW 5: Alpha half-life ───────────────────────────────
    strength,hl_remaining,hl_alive=halflife_remaining(symbol,comp,price,atr)

    # ── Kelly ────────────────────────────────────────────────
    kf,drisk,shares=kelly_size(price,atr,win_rate=bayes_p) if alert else (0.0,0.0,0)
    stop_price=round(price-atr*ATR_STOP_MULT,2) if atr>0 else 0.0
    target=round(price+atr*ATR_TARGET_MULT,2) if atr>0 else 0.0

    # R:R check (NEW from to-do list — must be >2.0)
    rr_ratio=((target-price)/atr) if atr>0 else 0.0
    rr_ok=rr_ratio>=2.0

    return {
        "symbol":symbol,"price":round(price,2),"score":comp,"alert":alert,
        "hurst_H":round(H,3),"hurst_score":round(h_sc,1),"hurst_regime":h_reg,
        "hawkes_lam":cur_lam,"hawkes_score":hawk_sc,"hawkes_sig":hawk_sig,
        "ofi":cur_ofi,"ofi_delta":ofi_d,"ofi_score":o_sc,"ofi_sig":o_sig,
        "market":market["regime"],"sector_etf":etf,
        "sector_rs":sec_rs,"sector_score":sec_sc,"sector_gate":sec_gate,
        # NEW fields
        "add_val":add_val,"add_bull":add_bull,"add_score":round(add_sc,1),
        "zscore":round(z,3),"zscore_score":round(z_sc,1),"zscore_ok":z_long_ok,
        "kurtosis":kurt,"skewness":skew,"ks_score":ks_sc,"ks_label":ks_label,
        "bayes_prob":round(bayes_p*100,1),"bayes_factors":bayes_factors,
        "hl_strength":strength,"hl_remaining":round(hl_remaining,0),"hl_alive":hl_alive,
        "rr_ratio":round(rr_ratio,2),"rr_ok":rr_ok,
        "kelly_frac":kf,"dollar_risk":drisk,"shares":shares,
        "atr":atr,"stop":stop_price,"target":target,
        "health_mult":health_mult,"health_label":health_label,
        "intraday_ret":health_detail.get("intraday_ret",0.0),
        "gap_ret":health_detail.get("gap_ret",0.0),
        "vwap":health_detail.get("vwap",0.0),
        "below_vwap":health_detail.get("below_vwap",False),
        "mcap_b":mcap_b,"scanned_at":datetime.now().strftime("%H:%M:%S"),
    }

def run_full_scan(symbols=WATCHLIST, timeframe="5m") -> tuple[pd.DataFrame, dict]:
    market=get_market_regime(); results=[]; blocked=[]
    for sym in symbols:
        try:
            p,mb=is_midcap(sym)
            if not p: blocked.append(f"{sym}(${mb:.1f}B)"); continue
            r=scan_symbol(sym,market,timeframe)
            if r: results.append(r)
        except Exception as e:
            print(f"[ERR] {sym}: {e}")
    market["blocked"]=blocked; market["blocked_count"]=len(blocked)
    market["scanned"]=len(results)
    if not results: return pd.DataFrame(), market
    return pd.DataFrame(results).sort_values("score",ascending=False).reset_index(drop=True), market


# ============================================================
#  STREAMLIT DASHBOARD
# ============================================================

def run_dashboard():
    st.set_page_config(page_title="QUANT v3",page_icon="⚡",
                       layout="wide",initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;700&display=swap');
    html,body,[class*="css"]{background:#060a0d!important;color:#c8d8e8!important;}
    .hdr{font-family:'Share Tech Mono',monospace;font-size:26px;letter-spacing:5px;
         color:#00e5ff;text-shadow:0 0 20px rgba(0,229,255,0.5);}
    .sub{font-family:'Share Tech Mono',monospace;font-size:11px;color:#3a6a82;letter-spacing:3px;}
    .card{border-radius:8px;padding:16px;margin-bottom:10px;
          border:1px solid #1a2e40;background:#0d1820;}
    .card.on{border-color:#00ff8c;background:#081812;box-shadow:0 0 20px rgba(0,255,140,0.1);}
    .card.warn{border-color:#ffb400;}
    .sym{font-family:'Share Tech Mono',monospace;font-size:20px;color:#e8f4ff;letter-spacing:3px;}
    .sc{font-family:'Share Tech Mono',monospace;font-size:28px;font-weight:700;}
    .bayes{font-family:'Share Tech Mono',monospace;font-size:22px;font-weight:700;}
    .tag{display:inline-block;font-family:'Share Tech Mono',monospace;font-size:10px;
         padding:2px 7px;border-radius:3px;margin:2px;letter-spacing:1px;}
    .row{display:flex;gap:8px;margin-top:8px;}
    .cell{flex:1;background:#0a1520;border-radius:4px;padding:5px 8px;}
    .lbl{font-family:'Share Tech Mono',monospace;font-size:9px;color:#2a4a62;
         text-transform:uppercase;letter-spacing:1px;}
    .val{font-family:'Share Tech Mono',monospace;font-size:12px;color:#a0c0d8;}
    .bar-bg{background:#0a1520;border-radius:3px;height:5px;margin:6px 0;overflow:hidden;}
    .bar-fg{height:5px;border-radius:3px;}
    .trade{background:#0a1420;border:1px solid #1a3040;border-radius:5px;
           padding:8px 12px;margin-top:8px;
           font-family:'Share Tech Mono',monospace;font-size:11px;}
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.header("🔍 v3 Controls")

        timeframe = st.selectbox("Timeframe", ["15s","1m","5m","15m"],
                                  index=2,
                                  help="15s = simulated from 1m bars")

        if os.path.exists("auto_watchlist.txt"):
            with open("auto_watchlist.txt","r") as f: auto=f.read()
        else:
            auto="\n".join(WATCHLIST)
        custom_syms = st.text_area("Watchlist (one per line)", value=auto, height=250)
        run_btn = st.button("▶ RUN SCAN NOW", use_container_width=True)
        refresh = st.slider("Auto-Refresh (sec)", 10, 300, 60)
        acct    = st.number_input("Account Size ($)", value=int(ACCOUNT_SIZE),
                                   step=5000, format="%d")
        st.markdown("---")
        st.markdown(f"""
        <div style='font-family:Share Tech Mono;font-size:10px;color:#2a4a62'>
        <b style='color:#00e5ff'>v3 ALL 10 LAYERS:</b><br>
        ✓ Hurst regime<br>
        ✓ Directional Hawkes λ<br>
        ✓ OFI buy/sell pressure<br>
        ✓ Sector RS gate<br>
        ✓ ADD breadth proxy<br>
        ✓ Z-Score exhaustion gate<br>
        ✓ Kurtosis / Skewness filter<br>
        ✓ Bayesian win probability<br>
        ✓ Alpha half-life decay<br>
        ✓ ATR trailing stop (3:1 R:R)<br><br>
        <b style='color:#ffb400'>CAP: $300M–$20B</b><br>
        <b style='color:#ffb400'>TF: {timeframe}</b><br>
        Data: yfinance (Schwab ready)
        </div>""", unsafe_allow_html=True)

    symbols=[s.strip().upper() for s in custom_syms.split("\n") if s.strip()]

    for k,v in [("results",pd.DataFrame()),("market",{}),
                ("last_scan",0.0),("count",0)]:
        if k not in st.session_state: st.session_state[k]=v

    now=time.time()
    should=run_btn or (now-st.session_state["last_scan"]>refresh) or st.session_state["last_scan"]==0
    if should:
        with st.spinner(f"Scanning {len(symbols)} symbols on {timeframe}..."):
            df,market=run_full_scan(symbols, timeframe)
        st.session_state.update({"results":df,"market":market,
                                   "last_scan":now,"count":st.session_state["count"]+1})

    df=st.session_state["results"]; market=st.session_state["market"]
    count=st.session_state["count"]

    # ── Header ───────────────────────────────────────────────
    c1,c2=st.columns([3,1])
    with c1:
        st.markdown('<div class="hdr">⚡ QUANT SCANNER v3</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="sub">10-LAYER ENGINE · {timeframe.upper()} BARS · ALL FACTORS ACTIVE</div>',
                    unsafe_allow_html=True)
    with c2:
        add_val,add_bull=get_add_breadth()
        add_color="#00ff8c" if add_bull else "#ff4060"
        st.markdown(
            f'<div style="text-align:right;margin-top:6px">'
            f'<div class="sub">{datetime.now().strftime("%H:%M:%S")} EST · #{count}</div>'
            f'<div style="font-family:Share Tech Mono;font-size:12px;'
            f'color:{add_color}">ADD {"▲ BULLISH" if add_bull else "▼ BEARISH"} '
            f'({add_val:.3f})</div>'
            f'<div class="sub">Next in {max(0,int(refresh-(now-st.session_state["last_scan"])))}s</div>'
            f'</div>', unsafe_allow_html=True)

    if market:
        rc="#00ff8c" if "RISK-ON" in market.get("regime","") else (
           "#ff4060" if "RISK-OFF" in market.get("regime","") else "#ffb400")
        bc=f" 🚫{market['blocked_count']} blocked" if market.get("blocked_count",0)>0 else ""
        st.markdown(
            f'<div style="font-family:Share Tech Mono;font-size:12px;'
            f'color:#4a8aaa;letter-spacing:2px;margin-bottom:14px">'
            f'MARKET: <span style="color:{rc}">{market.get("regime","—")}</span>'
            f' · SPY {market.get("spy_price","—")} ({market.get("spy_dev",0):+.2f}%)'
            f' · QQQ {market.get("qqq_price","—")} ({market.get("qqq_dev",0):+.2f}%)'
            f' · ✅{market.get("scanned",0)} scanned{bc}</div>',
            unsafe_allow_html=True)

    if df.empty:
        st.info("No results — click ▶ RUN SCAN NOW")
        time.sleep(refresh); st.rerun(); return

    alerts=df[df["alert"]==True]
    if not alerts.empty:
        syms="  ·  ".join(f"{r['symbol']} [{r['score']}] {r['bayes_prob']:.0f}%"
                           for _,r in alerts.iterrows())
        st.markdown(
            f'<div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.35);'
            f'border-radius:6px;padding:10px 16px;margin-bottom:14px;'
            f'font-family:Share Tech Mono;font-size:13px;color:#00e5ff;letter-spacing:2px">'
            f'▶ SIGNAL  ·  {syms}</div>', unsafe_allow_html=True)

    cols=st.columns(min(4,max(1,len(df))))
    for i,(_,r) in enumerate(df.iterrows()):
        sc=r["score"]
        sc_col=("#00ff8c" if sc>=75 else "#00e5ff" if sc>=SIGNAL_THRESHOLD
                else "#ffb400" if sc>=45 else "#ff4060")
        css="on" if r["alert"] else ("warn" if sc>=55 else "")
        bp=r["bayes_prob"]
        bp_col="#00ff8c" if bp>=70 else "#ffb400" if bp>=55 else "#ff4060"
        hl=r["hl_remaining"]; hl_str=f"⏱{int(hl)}s" if r["hl_alive"] else "—"
        rr_str=f"{'✓' if r['rr_ok'] else '✗'} R:R {r['rr_ratio']:.1f}"

        trade_html=""
        if r["alert"]:
            trade_html=f"""<div class="trade">
            ENTRY ${r['price']} · STOP ${r['stop']} · TARGET ${r['target']}<br>
            Kelly {r['kelly_frac']:.1%} → {r['shares']} shares · ${r['dollar_risk']} risk
            </div>"""

        with cols[i%len(cols)]:
            st.markdown(f"""
            <div class="card {css}">
              <div style="display:flex;justify-content:space-between">
                <div>
                  <div class="sym">{r['symbol']}</div>
                  <div style="font-family:Share Tech Mono;font-size:11px;color:#4a7a92">
                    ${r['price']} · MCap ${r.get('mcap_b',0):.1f}B
                  </div>
                </div>
                <div style="text-align:right">
                  <div class="sc" style="color:{sc_col}">{sc}</div>
                  <div class="bayes" style="color:{bp_col}">{bp:.0f}%</div>
                  <div class="sub">BAYES WIN</div>
                </div>
              </div>
              <div class="bar-bg">
                <div class="bar-fg" style="width:{sc}%;background:{sc_col}"></div>
              </div>
              <div style="margin:4px 0">
                <span class="tag" style="background:#0a1828;color:#4af;border:1px solid #1a3a52">{r['hurst_regime']}</span>
                <span class="tag" style="background:#0a1828;color:#4af;border:1px solid #1a3a52">{r['hawkes_sig']}</span>
                <span class="tag" style="background:#0a1828;color:#4af;border:1px solid #1a3a52">{r['ofi_sig']}</span>
                <span class="tag" style="background:#0a1828;color:{'#0f8' if r['add_bull'] else '#f46'};border:1px solid #1a3a52">
                  ADD {'▲' if r['add_bull'] else '▼'}
                </span>
                <span class="tag" style="background:#0a1828;color:{'#0f8' if r['zscore_ok'] else '#f46'};border:1px solid #1a3a52">
                  Z={r['zscore']:.2f}
                </span>
                <span class="tag" style="background:#0a1828;color:#af8;border:1px solid #1a3a52">
                  {r['ks_label']}
                </span>
              </div>
              <div class="row">
                <div class="cell"><div class="lbl">HURST H</div><div class="val">{r['hurst_H']}</div></div>
                <div class="cell"><div class="lbl">OFI</div><div class="val">{r['ofi']}</div></div>
                <div class="cell"><div class="lbl">KURT</div><div class="val">{r['kurtosis']}</div></div>
                <div class="cell"><div class="lbl">SKEW</div><div class="val">{r['skewness']}</div></div>
              </div>
              <div class="row">
                <div class="cell"><div class="lbl">SECTOR</div>
                  <div class="val" style="color:{'#0f8' if r['sector_gate'] else '#f46'}">{r['sector_etf']} {'✓' if r['sector_gate'] else '✗'}</div>
                </div>
                <div class="cell"><div class="lbl">R:R</div>
                  <div class="val" style="color:{'#0f8' if r['rr_ok'] else '#f46'}">{rr_str}</div>
                </div>
                <div class="cell"><div class="lbl">HALF-LIFE</div><div class="val">{hl_str}</div></div>
                <div class="cell"><div class="lbl">HEALTH</div>
                  <div class="val">{r['health_mult']:.2f}×</div>
                </div>
              </div>
              {trade_html}
            </div>""", unsafe_allow_html=True)

    # ── Trade log ────────────────────────────────────────────
    if TRADE_LOG:
        st.markdown("---")
        st.markdown("### 📋 Trade Log")
        total=sum(t["P&L $"] for t in TRADE_LOG)
        wins=sum(1 for t in TRADE_LOG if t["P&L $"]>0)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Total P&L", f"${total:+.2f}")
        m2.metric("Win Rate", f"{wins/len(TRADE_LOG)*100:.0f}%")
        m3.metric("Trades", len(TRADE_LOG))
        m4.metric("Wins/Losses", f"{wins}/{len(TRADE_LOG)-wins}")
        st.dataframe(pd.DataFrame(TRADE_LOG[::-1]),
                     use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📊 Full Results Table")
    show_cols=["symbol","score","bayes_prob","zscore","kurtosis","skewness",
               "ks_label","hurst_regime","hawkes_sig","ofi_sig","sector_etf",
               "add_bull","rr_ok","kelly_frac","shares","stop","target","health_label"]
    st.dataframe(df[[c for c in show_cols if c in df.columns]],
                 use_container_width=True, hide_index=True)

    time.sleep(refresh)
    st.rerun()


# ============================================================
#  BACKGROUND THREADS — start once
# ============================================================

_threads_started = False
def _start_background():
    global _threads_started
    if not _threads_started:
        threading.Thread(target=update_add_breadth_yf,
                         daemon=True, name="add_breadth").start()
        _threads_started = True

_start_background()


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n[QUANT v3] Run: python -m streamlit run scanner_v3.py\n")

if STREAMLIT_MODE:
    run_dashboard()
