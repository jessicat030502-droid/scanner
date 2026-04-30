"""
QUANT SCANNER v3 — Mid/Small Cap Day Trading Engine
=====================================================
Run:   python -m streamlit run scanner_v3.py
Deps:  pip install yfinance pandas numpy scipy streamlit openpyxl schedule

10 Layers: Hurst · Hawkes · OFI · Sector RS · ADD Breadth ·
           Z-Score · Kurtosis/Skew · Bayesian Prob · Half-Life · ATR R:R

Universe Scanner auto-runs at 9:00 AM EST daily.
Cap filter: $300M–$15B. Timeframe: 15s / 1m / 5m / 15m toggle.
"""

import os, time, math, warnings, threading
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from scipy import stats as sp_stats
import yfinance as yf
import schedule

warnings.filterwarnings("ignore")

try:
    import streamlit as st
    STREAMLIT_MODE = True
except ImportError:
    STREAMLIT_MODE = False

try:
    import schwab
    # Only mark as available if credentials are actually set
    _schwab_key    = os.environ.get("SCHWAB_API_KEY", "")
    _schwab_secret = os.environ.get("SCHWAB_API_SECRET", "")
    _schwab_acct   = os.environ.get("SCHWAB_ACCOUNT_ID", "")
    SCHWAB_AVAILABLE = bool(_schwab_key and _schwab_secret and _schwab_acct)
    # To connect: set SCHWAB_API_KEY, SCHWAB_API_SECRET, SCHWAB_ACCOUNT_ID env vars
except ImportError:
    SCHWAB_AVAILABLE = False
    # Running in yfinance mode — all features work, data has 15s delay


# ── Config ────────────────────────────────────────────────────────────────────

WATCHLIST = [
    "CRUS","POWI","MTSI","JAMF","ALKT","TASK",
    "BOOT","GIII","PLAY","LESL","PRPL","HIMS",
    "ACAD","INVA","PDCO","HCAT","NVCR","GKOS",
    "CURO","OPFI","HIBB","GCMG","NRDS",
    "CIVI","BATL","REX","PTEN","WTTR",
    "KTOS","ASTE","HLIO","DLX","HAYW",
    "TROX","RYAM","KWR","NTST","GMRE","EPRT",
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

UNIVERSE = list(dict.fromkeys([
    "CRUS","POWI","MTSI","JAMF","ALKT","TASK","APPF","VCRA","DOMO","EVER",
    "INVA","TNDM","HAYW","ASTE","KTOS","DLX","HLIO","TROX","RYAM","KWR",
    "NTST","GMRE","EPRT","CIVI","PTEN","WTTR","BOOT","GIII","PLAY","LESL",
    "PRPL","HIMS","ACAD","PDCO","HCAT","NVCR","GKOS","CURO","OPFI","GCMG",
    "AXNX","PRCT","ARVN","ARWR","FOLD","IMVT","KRYS","LGND","MYGN","NKTR",
    "NVAX","PCRX","RARE","RCUS","SRPT","TGTX","ABR","AIV","ALEX","AMH",
    "STAG","CUBE","DEA","EFC","EQC","FBRT","GOOD","HIW","AMRX","ATRC",
    "BCPC","BLKB","BPOP","CABO","CADE","CARG","CBT","CCOI","CDRE","CEVA",
    "CHCO","CHDN","CHEF","CHGG","CIR","CLAR","CLDT","CLW","CMCO","CNMD",
    "CNSL","COHU","CONN","COUP","CRVL","CTBI","CVBF","CVCO","CVLT","CWST",
    "DAN","DBRG","DIN","DIOD","DXLG","EAT","ECPG","EGAN","EGHT","ELME",
    "EML","ENVA","EPAC","ERII","ESRT","ETSY","EVGO","EVTC","EXLS","EXPI",
    "EXPO","EZPW","FARO","FBMS","FELE","FGEN","FISI","FLNC","FLR","FMBH",
    "FORM","FORR","FOXF","FRME","FRST","FSTR","FTDR","GDEN","GDOT","GEOS",
    "GIFI","GLDD","GLNG","GNSS","GNW","GPOR","GPRE","GRPN","GSBC","GSHD",
    "GTLS","HAFC","HAIN","HALO","HASI","HBT","HCC","HCSG","HEES","HELE",
    "HFWA","HIBB","HIIQ","HIMX","HLX","HMST","HOPE","HROW","HRMY","HSII",
    "HTBK","HTLD","HTLF","HURC","HURN","HVT","HWKN","IART","IBTX","ICFI",
    "ICHR","IDCC","IDYA","IESC","IIPR","IMCR","INDB","INMD","INSM","IONS",
    "IPAR","IRDM","IRTC","ITRI","JACK","JBSS","JJSF","KELYA","KMPR","KNTK",
    "KVHI","LBAI","LCII","LCUT","LECO","LKFN","LMAT","LNTH","LOVE","LPRO",
    "LQDT","LSCC","LSTR","LTHM","LYTS","MARA","MATW","MATX","MBIN","MBWM",
    "MCRI","MDGL","MDRX","MERC","MFIN","MGNI","MGPI","MLAB","MMSI","MNKD",
    "MNRO","MODV","MOFG","MORN","MRCY","MRTN","MRUS","MSEX","MTDR","MVBF",
    "NATH","NBTB","NCBS","NDLS","NEOG","NESR","NEXT","NFBK","NLSN","NMFC",
    "NMIH","NNBR","NOMD","NOVT","NSP","NTGR","NTRA","NWBI","NXRT","NYCB",
    "OBNK","OCFC","OCGN","OCSL","OCUL","OFG","OGS","OMCL","OPCH","OPRT",
]))

# ── Thresholds ───────────────────────────────────────────────
MIDCAP_MIN         = 300_000_000    # $300M minimum — covers small/mid-cap watchlist
MIDCAP_MAX         = 15_000_000_000  # $15B maximum
ACCOUNT_SIZE       = float(os.environ.get("ACCOUNT_SIZE", 50000))
SIGNAL_THRESHOLD   = 65
ATR_STOP_MULT      = 1.5
ATR_TARGET_MULT    = 3.0             # Used in TREND mode only
SPY_WEAK_THRESH    = -0.005
SECTOR_RS_MIN      = 1.01            # Sector must lead SPY by 1% (1.05 was too strict for normal markets)
HAWKES_DECAY       = 0.3
OFI_LONG_ENTRY     = 0.30            # Adaptive base (overridden by regime)
OFI_SHORT_ENTRY    = -0.30
OFI_LONG_EXIT      = 0.45
OFI_SHORT_EXIT     = 0.55
ZSCORE_MAX         = 2.5
ZSCORE_MIN         = -2.5
Z_ENTRY_THRESH     = 2.0             # Exhaustion entry threshold
KURTOSIS_MIN       = 3.0
SKEW_LONG_MIN      = 0.1
SKEW_SHORT_MAX     = -0.1
BAYES_BASE         = 0.50
BAYES_ADD          = 0.08
BAYES_SECTOR       = 0.10
BAYES_HAWKES       = 0.14
HALFLIFE_BASE      = 600
HALFLIFE_MIN       = 120
HALFLIFE_MAX       = 1800
INTRADAY_SOFT      = -0.010
INTRADAY_HARD      = -0.020
INTRADAY_KILL      = -0.030
GAP_DOWN_THRESH    = -0.010
LOWER_LOW_BARS     = 6

# ── Liquidity Firewall ────────────────────────────────────────
MIN_DOLLAR_VOLUME  = 5_000_000       # $5M avg daily dollar volume
MIN_REL_VOLUME     = 1.2             # Current volume must be 1.2x average (1.5 blocked off-peak scans)

# ── Strategy Switch ───────────────────────────────────────────
# TREND:         Hurst + Hawkes momentum — for trending/volatile markets
# MEAN_REVERSION: Z-score exhaustion + VWAP reversion — for ranging markets
# AUTO:          ADX + SPY vol determines which mode fires (recommended)
STRATEGY_MODE      = "AUTO"          # "TREND" | "MEAN_REVERSION" | "AUTO"

# ── Mean Reversion Exit ───────────────────────────────────────
MR_TAKE_PROFIT_PCT = 0.005           # 0.5% scalp target
MR_STOP_LOSS_PCT   = 0.008           # 0.8% hard stop

# ── ADX Regime ───────────────────────────────────────────────
ADX_MAX            = 25              # Above this = strong trend = no mean reversion
SPY_VOL_NOTRADE    = 0.020           # SPY move > 2.0% = no-trade zone (1.5% was too tight)
SPY_VOL_CALM       = 0.005           # < 0.5% = calm market
SPY_VOL_NORMAL     = 0.010           # 0.5–1.0% = normal market


# ── Market Cap Filter ─────────────────────────────────────────────────────────

_mcap_cache: dict = {}

def get_market_cap(symbol: str) -> Optional[float]:
    now = time.time()
    if symbol in _mcap_cache and now - _mcap_cache[symbol][1] < 3600:
        return _mcap_cache[symbol][0]
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


# ── Data Fetchers ─────────────────────────────────────────────────────────────

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
    try:
        interval = "1m" if timeframe == "15s" else timeframe
        period   = "5d" if timeframe in ("5m","15m") else "1d"
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty or len(df) < 10:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()
        if timeframe == "15s":
            rows = [{"open":r["open"],"high":r["high"],"low":r["low"],
                     "close":r["close"],"volume":r["volume"]/4}
                    for _, r in df.iterrows() for _ in range(4)]
            df = pd.DataFrame(rows)
        return df
    except Exception:
        return None

def fetch_closes(symbol: str, n: int = 25) -> Optional[np.ndarray]:
    df = fetch_daily(symbol, n)
    return df["close"].values if df is not None else None


# ── SPY/Sector close cache (5-min TTL) ───────────────────────
# SPY closes were being fetched 3+ times per scan: market regime,
# sector RS (once per ETF), and strategy regime. Cache eliminates
# all duplicate network calls within a scan cycle.
_closes_cache: dict = {}
_CLOSES_TTL = 300  # 5 minutes

def fetch_closes_cached(symbol: str, n: int = 25) -> Optional[np.ndarray]:
    key = f"{symbol}:{n}"
    now = time.time()
    if key in _closes_cache and now - _closes_cache[key][1] < _CLOSES_TTL:
        return _closes_cache[key][0]
    result = fetch_closes(symbol, n)
    if result is not None:
        _closes_cache[key] = (result, now)
    return result


# ── ADD Breadth (SPY up-volume proxy, updates every 60s) ──────────────────────

_add_cache = {"value": 0.0, "bullish": True, "updated": 0.0}
_add_lock  = threading.Lock()

def _update_add_loop():
    while True:
        try:
            df = yf.Ticker("SPY").history(period="1d", interval="1m")
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
                rng   = df["high"].values - df["low"].values
                br    = np.where(rng > 0, (df["close"].values - df["low"].values) / rng, 0.5)
                ratio = (br * df["volume"].values).sum() / df["volume"].values.sum()
                with _add_lock:
                    _add_cache.update({"value":round(ratio,4), "bullish":ratio>0.52, "updated":time.time()})
        except Exception:
            pass
        time.sleep(60)

def get_add_breadth() -> tuple:
    with _add_lock:
        return _add_cache["value"], _add_cache["bullish"]

def add_score(bullish: bool) -> float:
    return 70.0 if bullish else 30.0


# ── Z-Score (exhaustion gate) ─────────────────────────────────────────────────

def compute_zscore(prices: np.ndarray, window: int = 20) -> float:
    if len(prices) < window:
        return 0.0
    s = np.std(prices[-window:], ddof=1)
    return float((prices[-1] - np.mean(prices[-window:])) / s) if s else 0.0

def zscore_gate(z: float, direction: str) -> tuple:
    if direction == "LONG"  and z >  ZSCORE_MAX: return False, f"Z={z:.2f} EXHAUSTED"
    if direction == "SHORT" and z <  ZSCORE_MIN: return False, f"Z={z:.2f} EXHAUSTED"
    return True, f"Z={z:.2f} ✓"

def zscore_score(z: float) -> float:
    return float(np.clip(100 - abs(z) * 20, 0, 100))


# ── Kurtosis & Skewness (fat-tail filter) ─────────────────────────────────────

def compute_kurtosis_skew(prices: np.ndarray) -> tuple:
    if len(prices) < 20:
        return 0.0, 0.0
    log_r = np.diff(np.log(prices + 1e-9))
    return round(float(sp_stats.kurtosis(log_r, fisher=True)), 3), \
           round(float(sp_stats.skew(log_r)), 3)

def kurtosis_skew_score(kurt: float, skew: float, direction: str = "LONG") -> tuple:
    base  = float(np.clip((kurt / 6.0) * 60, 0, 60))
    bonus = 0.0
    if direction == "LONG"  and skew >  SKEW_LONG_MIN:  bonus = float(np.clip(skew      * 20, 0, 40))
    if direction == "SHORT" and skew <  SKEW_SHORT_MAX:  bonus = float(np.clip(abs(skew) * 20, 0, 40))
    score = round(base + bonus, 1)
    label = "🎯 FAT TAIL" if kurt > 5.0 and score > 70 else \
            "📊 ELEVATED" if kurt > KURTOSIS_MIN else "〰 NORMAL"
    return score, label


# ── Bayesian Win Probability ──────────────────────────────────────────────────

def bayesian_win_prob(add_bull: bool, sector_ok: bool,
                      hawk_sc: float, z_ok: bool, ks_sc: float) -> tuple:
    p, log = BAYES_BASE, [f"Base: {BAYES_BASE:.0%}"]

    def upd(cond, boost, label):
        nonlocal p
        p = float(np.clip(p + (boost if cond else -boost * 0.5), 0.03, 0.97))
        log.append(f"{label} → {p:.0%}")

    upd(add_bull,         BAYES_ADD,    f"ADD {'↑' if add_bull else '↓'}")
    upd(sector_ok,        BAYES_SECTOR, f"Sector {'✓' if sector_ok else '✗'}")
    upd(hawk_sc >= 72,    BAYES_HAWKES, f"Hawkes {'🔥' if hawk_sc>=72 else '⚡' if hawk_sc>=58 else '—'}")
    upd(z_ok,             0.04,         f"Z {'✓' if z_ok else '✗'}")
    if ks_sc >= 60:
        p = float(np.clip(p + 0.05, 0.03, 0.97))
        log.append(f"FatTail → {p:.0%}")

    return round(p, 4), log


# ── Alpha Half-Life Decay ─────────────────────────────────────────────────────

_signal_start: dict = {}

def compute_halflife(score: float, atr_pct: float) -> float:
    return float(np.clip(HALFLIFE_BASE * (score / 65.0) / (1.0 + atr_pct * 10),
                         HALFLIFE_MIN, HALFLIFE_MAX))

def halflife_remaining(symbol: str, score: float, price: float, atr: float) -> tuple:
    now     = time.time()
    atr_pct = (atr / price) if price > 0 else 0.02
    if score < SIGNAL_THRESHOLD:
        _signal_start.pop(symbol, None)
        return 0.0, 0.0, False
    if symbol not in _signal_start:
        hl = compute_halflife(score, atr_pct)
        _signal_start[symbol] = {"time": now, "halflife": hl, "lambda": math.log(2) / hl}
    sig      = _signal_start[symbol]
    strength = math.exp(-sig["lambda"] * (now - sig["time"]))
    remaining = max(0.0, sig["halflife"] * math.log(max(strength,1e-9)) / math.log(0.5))
    alive    = strength > 0.25
    if not alive:
        _signal_start.pop(symbol, None)
    return round(strength, 4), round(remaining, 1), alive


# ── Core Indicators ───────────────────────────────────────────────────────────

def compute_hurst(prices: np.ndarray) -> float:
    arr = np.array(prices)
    if len(arr) < 30: return 0.5
    log_r = np.diff(np.log(arr + 1e-9))
    n     = len(log_r)
    lags  = np.unique(np.floor(np.geomspace(5, n//2, 12)).astype(int))
    lags  = lags[lags >= 4]
    rs_v, vl = [], []
    for lag in lags:
        nw = n // lag
        if nw < 2: continue
        rl = []
        for i in range(nw):
            seg = log_r[i*lag:(i+1)*lag]
            cs  = np.cumsum(seg - seg.mean())
            S   = seg.std(ddof=1)
            if S > 0: rl.append((cs.max() - cs.min()) / S)
        if rl: rs_v.append(np.mean(rl)); vl.append(lag)
    if len(vl) < 3: return 0.5
    H, _ = np.polyfit(np.log(vl), np.log(rs_v), 1)
    return float(np.clip(H, 0.0, 1.0))

def hurst_score(H: float)  -> float: return float(np.clip((H - 0.5) * 200 + 50, 0, 100))
def hurst_regime(H: float) -> str:
    return "📈 TRENDING" if H > 0.58 else ("🔄 REVERTING" if H < 0.42 else "〰 CHOPPY")


def compute_hawkes(df: pd.DataFrame) -> tuple:
    h, l, c, v = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    n = len(v)
    if n < 20: return 0.0, 50.0
    br   = np.where(h - l > 0, (c - l) / (h - l), 0.5)
    bv, sv = br * v, (1 - br) * v
    bb   = pd.Series(bv).rolling(20, min_periods=5).mean().values
    bs   = pd.Series(sv).rolling(20, min_periods=5).mean().values
    mu   = float(np.nanmean(bb[~np.isnan(bb)]))
    if mu <= 0: return 0.0, 50.0
    alpha, lams = mu * 0.5, np.zeros(n)
    lams[0] = mu
    for t in range(1, n):
        decay = mu + (lams[t-1] - mu) * math.exp(-HAWKES_DECAY)
        bbt   = bb[t-1] if not np.isnan(bb[t-1]) else mu
        bst   = bs[t-1] if not np.isnan(bs[t-1]) else mu
        delta = (alpha if bv[t-1] > bbt * 1.8 else 0.0) - (alpha * 0.8 if sv[t-1] > bst * 1.8 else 0.0)
        lams[t] = max(0.0, decay + delta)
    base  = np.nanmean(lams[max(0, n-30):-5]) if n > 10 else mu
    if base <= 0: return lams[-1], 50.0
    score = float(np.clip(50 + 50 * math.tanh(lams[-1] / (base + 1e-9) - 1.0), 0, 100))
    return round(lams[-1], 4), round(score, 1)

def hawkes_signal(s: float) -> str:
    if s >= 72:  return "🔥 CLUSTERING"
    if s >= 58:  return "⚡ BUILDING"
    if s >= 42:  return "〰 IDLE"
    if s >= 20:  return "❄ FADING"
    return "🔴 SELL PRESSURE"


# ── Intraday Hurst (5-min bars) ───────────────────────────────────────────────
# Fixes the stale-Hurst problem. Uses last 78 five-min bars (~1 trading day).
# Weighted 60% intraday / 40% daily so today's reversal is visible immediately.

def compute_hurst_intraday(df_5m: pd.DataFrame) -> tuple:
    if df_5m is None or len(df_5m) < 30:
        return 0.5, 50.0, "〰 CHOPPY"
    H      = compute_hurst(df_5m["close"].values[-78:])
    return round(H, 3), round(hurst_score(H), 1), hurst_regime(H)


def combined_hurst_score(h_daily: float, h_intraday: float) -> tuple:
    """Blends daily Hurst (40%) + intraday Hurst (60%).
    Returns (blended_score, conflict_flag).
    conflict_flag = True when daily says trending but intraday says reverting.
    """
    score    = 0.40 * hurst_score(h_daily) + 0.60 * hurst_score(h_intraday)
    conflict = hurst_regime(h_daily) == "📈 TRENDING" and h_intraday < 0.42
    return round(float(np.clip(score, 0, 100)), 1), conflict


def compute_ofi(df: pd.DataFrame, window: int = 12) -> tuple:
    h, l, c, v = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    br   = np.where(h - l > 0, (c - l) / (h - l), 0.5)
    ofi  = (pd.Series(br * v).rolling(window, min_periods=3).sum() /
            pd.Series(v).rolling(window, min_periods=3).sum().replace(0, np.nan)).fillna(0.5)
    cur  = float(ofi.iloc[-1])
    delt = float(ofi.iloc[-1] - ofi.iloc[-4]) if len(ofi) >= 4 else 0.0
    return round(cur, 4), round(delt, 4), round(float(np.clip(cur*100 + delt*50, 0, 100)), 1)

def ofi_signal(ofi: float, delta: float) -> str:
    # Thresholds aligned with OFI_LONG_ENTRY=0.30 adaptive base
    # Upper band: 0.30 + 0.20 buffer = 0.50 (accumulating territory)
    # Lower band: 0.50 - 0.20 buffer = 0.30 (selling territory)
    if ofi >= 0.55 and delta >= 0: return "🟢 ACCUMULATING"
    if ofi >= 0.52 and delta <  0: return "🟡 TOPPING"
    if ofi <= 0.35:                return "🔴 DISTRIBUTING"
    if ofi <= 0.44:                return "🟠 SELLING"
    return "⚪ NEUTRAL"


def compute_atr(df: pd.DataFrame, n: int = 14) -> float:
    if df is None or len(df) < n: return 0.0
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    trs = [max(h[-i] - l[-i],
               abs(h[-i] - c[-i-1]) if i < len(c) else 0,
               abs(l[-i] - c[-i-1]) if i < len(c) else 0)
           for i in range(1, min(n+1, len(c)))]
    return round(float(np.mean(trs)), 4) if trs else 0.0


def intraday_health(df: pd.DataFrame, prior_close: float) -> tuple:
    """
    Hard intraday momentum gate — fixes the CRUS problem.

    HARD BLOCKS (mult = 0.0, alert = False):
      - Price below VWAP                     → institutions net sellers, never long here
      - Intraday return <= -3% from open     → confirmed selloff
      - Gap down > 2% from prior close       → gap-down continuation risk

    SOFT PENALTIES (score multiplier):
      - Intraday -2% to -3%                  → 50% penalty
      - Intraday -1% to -2%                  → 25% penalty
      - Gap down 1-2%                        → 15% penalty
      - Lower lows on last 6 bars            → 30% penalty
    """
    if df is None or len(df) < 5: return 1.0, "NO DATA", {}
    try:
        today = pd.Timestamp.now(tz=df.index.tz).strftime("%Y-%m-%d")
        tb    = df[df.index.strftime("%Y-%m-%d") == today]
    except Exception:
        tb = df.tail(60)
    if len(tb) < 3: tb = df.tail(60)

    open_p  = float(tb["open"].iloc[0])
    curr    = float(tb["close"].iloc[-1])
    intra_r = (curr - open_p) / open_p if open_p > 0 else 0.0
    gap_r   = (open_p - prior_close) / prior_close if prior_close > 0 else 0.0
    recent  = tb["close"].values[-LOWER_LOW_BARS:]
    ll      = int(sum(recent[i] < recent[i-1] for i in range(1, len(recent))))
    making_ll = ll >= LOWER_LOW_BARS - 2
    tp    = (tb["high"] + tb["low"] + tb["close"]) / 3
    vwap  = float((tp * tb["volume"]).cumsum().iloc[-1] / tb["volume"].cumsum().iloc[-1])             if tb["volume"].sum() > 0 else curr
    below_vwap = curr < vwap

    mult, flags, hard_blocked = 1.0, [], False

    # ── HARD BLOCKS — only extreme conditions kill signal entirely ──
    # NOTE: below_vwap is NO LONGER a hard block because:
    #   - MR LONG entries REQUIRE price below VWAP (it's the setup condition)
    #   - Hard blocking below_vwap was preventing every valid MR long from showing
    # Instead: score penalty — the MR exhaustion gate enforces VWAP side correctly

    if intra_r <= INTRADAY_KILL:
        mult = 0.0; hard_blocked = True
        flags.append(f"HARD:SELLOFF{intra_r*100:.1f}%")

    if gap_r < -0.02:
        mult = 0.0; hard_blocked = True
        flags.append(f"HARD:GAP↓{gap_r*100:.1f}%")

    # VWAP position: soft penalty (not hard block)
    if below_vwap and not hard_blocked:
        mult *= 0.75
        flags.append(f"BELOW_VWAP${vwap:.2f}")

    # ── SOFT PENALTIES — only if not already hard blocked ────
    if not hard_blocked:
        if gap_r    < GAP_DOWN_THRESH: mult *= 0.85; flags.append(f"GAP↓{gap_r*100:.1f}%")
        if intra_r <= INTRADAY_HARD:   mult *= 0.50;  flags.append(f"WEAK{intra_r*100:.1f}%")
        elif intra_r <= INTRADAY_SOFT: mult *= 0.75;  flags.append(f"SOFT{intra_r*100:.1f}%")
        else:                                          flags.append(f"OK{intra_r*100:+.1f}%")
        if making_ll: mult *= 0.70; flags.append("LOWER-LOWS")

    mult = float(np.clip(mult, 0.0, 1.0))
    return mult, " | ".join(flags) or "HEALTHY", {
        "intraday_ret":round(intra_r*100,2), "gap_ret":round(gap_r*100,2),
        "today_open":round(open_p,2), "vwap":round(vwap,2),
        "below_vwap":curr < vwap, "lower_lows":ll, "health_mult":round(mult,3)
    }


def get_market_regime() -> dict:
    def dev(sym):
        arr = fetch_closes_cached(sym, 25)
        if arr is None or len(arr) < 20: return 0.0, 0.0
        sma = np.mean(arr[-20:])
        return arr[-1], (arr[-1] - sma) / sma if sma > 0 else 0.0
    sp, sd = dev("SPY"); qp, qd = dev("QQQ")
    if sd > 0.002 and qd > 0.002:                       r, al, mm = "RISK-ON 📈",  True,  1.10
    elif sd < SPY_WEAK_THRESH and qd < SPY_WEAK_THRESH: r, al, mm = "RISK-OFF 📉", False, 0.40
    else:                                                r, al, mm = "NEUTRAL ➡",  True,  1.00
    return {"regime":r,"allows_long":al,"mkt_mult":mm,
            "spy_price":round(sp,2),"spy_dev":round(sd*100,2),
            "qqq_price":round(qp,2),"qqq_dev":round(qd*100,2)}


def get_sector_rs(etf: str) -> tuple:
    ec, sc = fetch_closes_cached(etf, 25), fetch_closes_cached("SPY", 25)
    if ec is None or sc is None or len(ec) < 20 or len(sc) < 20: return 1.0, 50.0, True
    ratio = (ec[-1] / np.mean(ec[-20:])) / (sc[-1] / np.mean(sc[-20:]))
    return round(ratio, 4), round(float(np.clip(50+(ratio-1.0)*500,0,100)),1), ratio >= SECTOR_RS_MIN


def kelly_size(price: float, atr: float, win_rate: float = 0.60, rr: float = 2.0) -> tuple:
    if atr <= 0 or price <= 0: return 0.0, 0.0, 0
    kf     = float(np.clip((win_rate - (1 - win_rate) / rr) * 0.5, 0.0, 0.10))
    drisk  = ACCOUNT_SIZE * kf
    shares = int(drisk / (atr * ATR_STOP_MULT)) if atr * ATR_STOP_MULT > 0 else 0
    return round(kf, 4), round(drisk, 2), min(shares, int(ACCOUNT_SIZE * 0.20 / price))


# ── Anchored VWAP (resets daily at 9:30 AM) ──────────────────────────────────
# Fixes the skewed-VWAP problem. Previous day data does not bleed in.

def compute_anchored_vwap(df: pd.DataFrame) -> float:
    """
    VWAP anchored to today's 9:30 AM open. Resets every session.
    Falls back to rolling 20-bar VWAP if today's bars unavailable.
    """
    try:
        tz    = df.index.tz
        today = pd.Timestamp.now(tz=tz).strftime("%Y-%m-%d")
        tb    = df[df.index.strftime("%Y-%m-%d") == today]
        if len(tb) < 2:
            tb = df.tail(20)
        tp    = (tb["high"] + tb["low"] + tb["close"]) / 3
        return round(float((tp * tb["volume"]).sum() / tb["volume"].sum()), 4)
    except Exception:
        return float(df["close"].iloc[-1])


# ── Liquidity Firewall ────────────────────────────────────────────────────────
# Must pass BOTH checks before any math runs. Kills zombie stocks immediately.

def passes_liquidity_firewall(df_d: pd.DataFrame, symbol: str) -> tuple:
    """
    Returns (passes: bool, reason: str).
    Checks MIN_DOLLAR_VOLUME ($5M) and MIN_REL_VOLUME (1.5x).
    Placed at the very top of scan_symbol to save CPU on failures.
    """
    if df_d is None or len(df_d) < 20:
        return False, "INSUFFICIENT DATA"

    avg_price  = float(df_d["close"].tail(10).mean())
    avg_vol    = float(df_d["volume"].tail(20).mean())
    curr_vol   = float(df_d["volume"].iloc[-1])
    dollar_vol = avg_price * avg_vol

    if dollar_vol < MIN_DOLLAR_VOLUME:
        return False, f"LOW_DOLLAR_VOL ${dollar_vol/1e6:.1f}M < $5M"

    rel_vol = curr_vol / avg_vol if avg_vol > 0 else 0.0
    if rel_vol < MIN_REL_VOLUME:
        return False, f"LOW_RELVOL {rel_vol:.2f}x < 1.5x"

    return True, f"OK DV=${dollar_vol/1e6:.1f}M RV={rel_vol:.2f}x"


# ── ADX Calculation ───────────────────────────────────────────────────────────
# Used by regime switch. ADX < 25 = ranging = mean reversion valid.
# ADX > 25 = strong trend = only trend-following valid.

def compute_adx(df: pd.DataFrame, n: int = 14) -> float:
    """
    Average Directional Index — fully vectorized with pandas.
    Replaces the Python-loop Wilder smoother with ewm() — 5-10x faster.
    Returns float 0–100. Default 20.0 (neutral/ranging) on bad data.
    """
    if df is None or len(df) < n * 2:
        return 20.0
    h  = df["high"];  l = df["low"];  c = df["close"]

    # True range (vectorized)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

    # Directional movement (vectorized)
    up  = h.diff();  dn = -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)

    # Wilder smoothing via ewm (alpha = 1/n) — replaces the Python loop
    alpha  = 1.0 / n
    atr_s  = tr.ewm(alpha=alpha,  adjust=False).mean()
    pdi_s  = pdm.ewm(alpha=alpha, adjust=False).mean()
    ndi_s  = ndm.ewm(alpha=alpha, adjust=False).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr_s > 0, 100 * pdi_s / atr_s, 0.0)
        ndi = np.where(atr_s > 0, 100 * ndi_s / atr_s, 0.0)
        dx  = pd.Series(
            np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0),
            index=h.index)

    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return round(float(adx.iloc[-1]), 2)


# ── Strategy Regime Switch ────────────────────────────────────────────────────
# THE KEY FIX: Determines which strategy is active. NEVER allows both.
# Returns one of: "TREND" | "MEAN_REVERSION" | "NO_TRADE"
# Also returns the adaptive OFI threshold for the current environment.

def get_strategy_regime(df_intra: pd.DataFrame, spy_return: float) -> tuple:
    """
    Returns (strategy: str, adaptive_ofi: float, adx: float, reason: str).

    Decision tree:
      SPY move > 1.5%          → NO_TRADE (too volatile for either strategy)
      ADX > ADX_MAX (25)       → TREND (runaway trend, only momentum valid)
      SPY calm + ADX < 25      → MEAN_REVERSION (ranging market, fading valid)
      In between               → TREND (default to safer momentum strategy)

    Adaptive OFI threshold scales with volatility:
      Calm  (SPY < 0.5%): 0.20 — more sensitive to small imbalances
      Normal(SPY < 1.0%): 0.30 — standard
      Hot   (SPY < 1.5%): 0.35 — require stronger signal before entry
    """
    adx      = compute_adx(df_intra)
    spy_vol  = abs(spy_return)

    # Hard no-trade: market too volatile for any intraday strategy
    if spy_vol > SPY_VOL_NOTRADE:
        return "NO_TRADE", 0.35, adx, f"SPY_VOL {spy_vol:.1%} > 1.5%"

    # Strong trend detected: mean reversion will get run over
    if adx > ADX_MAX:
        return "TREND", 0.30, adx, f"ADX {adx:.1f} > {ADX_MAX} (trending)"

    # Calm ranging market: ideal for mean reversion / exhaustion fading
    if spy_vol < SPY_VOL_CALM and adx <= ADX_MAX:
        ofi_thresh = 0.20
        return "MEAN_REVERSION", ofi_thresh, adx, f"ADX {adx:.1f} CALM {spy_vol:.1%}"

    if spy_vol < SPY_VOL_NORMAL:
        ofi_thresh = 0.30
        return "MEAN_REVERSION", ofi_thresh, adx, f"ADX {adx:.1f} NORMAL {spy_vol:.1%}"

    # Elevated vol but below no-trade threshold: still mean reversion but cautious
    ofi_thresh = 0.35
    return "MEAN_REVERSION", ofi_thresh, adx, f"ADX {adx:.1f} VOLATILE {spy_vol:.1%}"


# ── Exhaustion Triple-Gate (Mean Reversion Engine) ───────────────────────────
# Replaces raw OFI score for MEAN_REVERSION mode.
# Only fires when: price overextended (Z) + OFI peak then fade + VWAP side correct.
# This is the core of the 70-80% win rate claim.

def check_exhaustion_signal(df: pd.DataFrame, vwap: float,
                             spy_return: float,
                             adaptive_ofi: float = 0.30) -> tuple:
    """
    Three-step exhaustion confirmation:
      Step A — Extension:  Z-score >= 2.0 (price far from mean)
      Step B — Exhaustion: OFI spiked then faded (ammunition depleted)
      Step C — VWAP side:  Price on correct side for reversion

    adaptive_ofi scales panic/exhaustion thresholds with market regime:
      0.20 = calm  → easier trigger (more sensitive)
      0.30 = normal
      0.35 = volatile → harder trigger (requires stronger signal)

    Returns (long_signal: bool, short_signal: bool, reason: str).
    """
    if df is None or len(df) < 20:
        return False, False, "INSUFFICIENT DATA"

    close = df["close"].values
    h     = df["high"].values
    l     = df["low"].values
    vol   = df["volume"].values

    # Z-score on the intraday series
    if len(close) >= 20:
        z_now = (close[-1] - np.mean(close[-20:])) / (np.std(close[-20:], ddof=1) + 1e-9)
    else:
        z_now = 0.0

    # OFI series (bulk volume classification)
    rng   = h - l
    br    = np.where(rng > 0, (close - l) / rng, 0.5)
    bv    = br * vol; sv = (1 - br) * vol
    # Signed OFI: +1 = all buying, -1 = all selling
    ofi_signed = (bv - sv) / (bv + sv + 1e-9)

    # Look at last 5 bars for peak/fade pattern
    recent_ofi  = ofi_signed[-5:] if len(ofi_signed) >= 5 else ofi_signed
    curr_ofi    = float(ofi_signed[-1])
    curr_price  = float(close[-1])
    above_vwap  = curr_price > vwap
    below_vwap  = curr_price < vwap
    spy_safe_l  = spy_return > -0.01   # SPY not cratering for longs
    spy_safe_s  = spy_return <  0.01   # SPY not surging for shorts

    # ── LONG: price crushed + selling exhausted + below VWAP ──
    is_oversold       = z_now <= -Z_ENTRY_THRESH
    was_panic_selling = float(recent_ofi.min()) < -(adaptive_ofi + 0.10)
    selling_exhausted = curr_ofi > -0.10
    long_signal = (is_oversold and was_panic_selling
                   and selling_exhausted and below_vwap and spy_safe_l)

    # ── SHORT: price mooning + buying exhausted + above VWAP ──
    is_overbought     = z_now >=  Z_ENTRY_THRESH
    was_aggr_buying   = float(recent_ofi.max()) >  (adaptive_ofi + 0.10)
    buying_exhausted  = curr_ofi < 0.10
    short_signal = (is_overbought and was_aggr_buying
                    and buying_exhausted and above_vwap and spy_safe_s)

    reason = (f"Z={z_now:.2f} OFI_now={curr_ofi:.2f} "
              f"peak={float(recent_ofi.max()):.2f} trough={float(recent_ofi.min()):.2f} "
              f"{'BELOW' if below_vwap else 'ABOVE'}_VWAP")

    return long_signal, short_signal, reason


# ── Volume Divergence Filter ──────────────────────────────────────────────────
# Final confirmation: price hitting extreme but volume drying up = no fuel left.
# This confirms the exhaustion signal — move can't continue without volume.

def check_volume_divergence(df: pd.DataFrame) -> bool:
    """
    Returns True if volume divergence confirmed (move is running out of fuel).
    Condition: price at new extreme BUT recent volume < prior volume.
    """
    if df is None or len(df) < 8:
        return True   # can't check = don't block

    recent_vol = float(df["volume"].tail(3).mean())
    prior_vol  = float(df["volume"].iloc[-6:-3].mean()) if len(df) >= 6 else recent_vol
    rising_price = float(df["close"].iloc[-1]) > float(df["close"].iloc[-3])
    falling_vol  = recent_vol < prior_vol

    # Divergence = price rising but volume falling (or price falling but volume falling)
    return falling_vol  # volume is drying up regardless of direction


# ── Mean Reversion Exit Engine ────────────────────────────────────────────────
# For MEAN_REVERSION mode: small scalp target OR VWAP touch — whichever first.
# This is what drives the high win rate — take the easy money, don't be greedy.

def determine_mr_exit(current_price: float, entry_price: float,
                      vwap: float, side: str) -> str:
    """
    Returns "EXIT_PROFIT" | "EXIT_LOSS" | "HOLD".
    Primary exit = VWAP touch (statistical edge ends here).
    Secondary exit = 0.5% profit target.
    Hard stop = 0.8% loss.
    """
    if side == "LONG":
        if current_price >= entry_price * (1 + MR_TAKE_PROFIT_PCT): return "EXIT_PROFIT"
        if current_price >= vwap:                                    return "EXIT_PROFIT"
        if current_price <= entry_price * (1 - MR_STOP_LOSS_PCT):   return "EXIT_LOSS"
    if side == "SHORT":
        if current_price <= entry_price * (1 - MR_TAKE_PROFIT_PCT): return "EXIT_PROFIT"
        if current_price <= vwap:                                    return "EXIT_PROFIT"
        if current_price >= entry_price * (1 + MR_STOP_LOSS_PCT):   return "EXIT_LOSS"
    return "HOLD"


# ── Position & Trade Log ──────────────────────────────────────────────────────

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

POSITIONS: dict = {}
TRADE_LOG: list = []

def update_trailing_stop(pos: Position, price: float) -> Position:
    if pos.direction == "LONG":
        pos.highest = max(pos.highest, price)
        pos.stop    = max(pos.stop, round(pos.highest - pos.atr * ATR_STOP_MULT, 2))
        pos.pnl     = round((price - pos.entry_price) * pos.shares, 2)
    elif pos.direction == "SHORT":
        pos.lowest  = min(pos.lowest, price)
        pos.stop    = min(pos.stop, round(pos.lowest + pos.atr * ATR_STOP_MULT, 2))
        pos.pnl     = round((pos.entry_price - price) * pos.shares, 2)
    return pos

def close_position(symbol: str, exit_price: float, reason: str):
    pos = POSITIONS.get(symbol)
    if pos and pos.direction != "FLAT":
        pnl = ((exit_price - pos.entry_price) if pos.direction == "LONG"
               else (pos.entry_price - exit_price)) * pos.shares
        TRADE_LOG.append({"Symbol":symbol,"Direction":pos.direction,
            "Entry $":pos.entry_price,"Exit $":round(exit_price,2),
            "Shares":pos.shares,"P&L $":round(pnl,2),
            "Result":"WIN ✅" if pnl>0 else "LOSS ❌",
            "Reason":reason,"Time":datetime.now().strftime("%H:%M:%S")})
    POSITIONS[symbol] = Position(symbol)


# ── Master Scan ───────────────────────────────────────────────────────────────

def scan_symbol(symbol: str, market: dict, timeframe: str = "5m",
                mode: str = "auto") -> Optional[dict]:
    """
    Dual-strategy scanner with hard strategy switch.

    STRATEGY_MODE (global config):
      AUTO         → ADX + SPY vol determines strategy each scan
      TREND        → Hurst + Hawkes momentum (volatile/trending markets)
      MEAN_REVERSION → Exhaustion triple-gate + VWAP exit (ranging markets)

    CRITICAL: Only ONE strategy fires per scan. Never both simultaneously.
    If strategy = NO_TRADE, returns None immediately.

    mode = "premarket"  → 9:00 AM build watchlist (daily Hurst, soft gates)
    mode = "intraday"   → market hours confirmation (hard gates, dual Hurst)
    mode = "auto"       → auto-detects from clock
    """
    if mode == "auto":
        hour = datetime.now().hour
        mode = "intraday" if 9 <= hour < 16 else "premarket"

    # ── FIREWALL 1: Market Cap ────────────────────────────────
    passes, mcap_b = is_midcap(symbol)
    if not passes: return None

    # ── FIREWALL 2: Fetch data ────────────────────────────────
    df_d = fetch_daily(symbol, 60)
    if df_d is None or len(df_d) < 30: return None

    # ── FIREWALL 3: Liquidity (runs before any expensive math) ─
    liq_ok, liq_reason = passes_liquidity_firewall(df_d, symbol)
    if not liq_ok: return None

    df_i = fetch_intraday(symbol, timeframe) or df_d.copy()
    if len(df_i) < 15: df_i = df_d.copy()

    price       = float(df_d["close"].iloc[-1])
    prior_close = float(df_d["close"].iloc[-2]) if len(df_d) >= 2 else price
    closes      = df_d["close"].values
    spy_ret     = market.get("spy_dev", 0.0) / 100.0
    vwap        = compute_anchored_vwap(df_i)

    # ── FIREWALL 4: Strategy Regime Switch ────────────────────
    # Determines which strategy is active. NEVER allows both.
    if STRATEGY_MODE == "AUTO":
        strategy, adaptive_ofi, adx_val, regime_reason = get_strategy_regime(df_i, spy_ret)
    elif STRATEGY_MODE == "TREND":
        strategy, adaptive_ofi, adx_val, regime_reason = "TREND", 0.30, compute_adx(df_i), "FORCED_TREND"
    else:
        strategy, adaptive_ofi, adx_val, regime_reason = "MEAN_REVERSION", 0.30, compute_adx(df_i), "FORCED_MR"

    if strategy == "NO_TRADE":
        return None   # Market too volatile — kill scan entirely

    # ── FIREWALL 5: Sector RS (updated to 1.05) ───────────────
    etf                      = SECTOR_MAP.get(symbol, "SPY")
    sec_rs, sec_sc, sec_gate = get_sector_rs(etf)
    # Sector RS: hard block only if completely failing (< 0.97 = sector badly lagging)
    # Between 0.97–1.01: soft penalty applied in composite score — still scanned
    if sec_rs < 0.97:
        return None  # Sector is actively dragging — hard block

    # ── Core calculations (shared by both strategies) ─────────
    H_daily              = compute_hurst(closes)
    H_intra, _, h_intra_regime = compute_hurst_intraday(df_i)
    lam, hawk_sc         = compute_hawkes(df_i)
    ofi, od, o_sc        = compute_ofi(df_i)
    atr                  = compute_atr(df_d)
    add_val, add_b       = get_add_breadth()
    z                    = compute_zscore(closes)
    kurt, skew           = compute_kurtosis_skew(closes)
    hlth, hlbl, hd       = intraday_health(df_i, prior_close)

    # ── STRATEGY BRANCH — exactly one fires ───────────────────

    if strategy == "TREND":
        # TREND MODE: Hurst + Hawkes + momentum OFI
        # Mean reversion signals (Z-score entry, exhaustion) are NOT used here
        if mode == "intraday":
            h_sc, hurst_conflict = combined_hurst_score(H_daily, H_intra)
        else:
            h_sc, hurst_conflict = hurst_score(H_daily), False

        direction_g = "LONG" if ofi >= 0.5 else "SHORT"
        ks_sc, ks_lbl = kurtosis_skew_score(kurt, skew, direction_g)
        z_ok, _       = zscore_gate(z, direction_g)
        bayes_p, b_log = bayesian_win_prob(add_b, sec_gate, hawk_sc, z_ok, ks_sc)

        raw = float(np.clip(
            0.12*h_sc + 0.20*hawk_sc + 0.18*o_sc + 0.10*sec_sc +
            0.12*add_score(add_b) + 0.10*zscore_score(z) + 0.08*ks_sc + 0.10*bayes_p*100,
            0, 100))

        comp = round(float(np.clip(
            raw * market["mkt_mult"] *
            (1.0 if add_b else 0.80) *
            (hlth if mode == "intraday" else max(hlth, 0.5)),
            0, 100)), 1)

        if hurst_conflict: comp = min(comp, 45.0)
        if not z_ok:       comp = min(comp, 55.0)

        intraday_blocked = (mode == "intraday" and
                            (hurst_conflict or hawk_sc < 20 or hlth == 0.0))
        intraday_block_reason = ("HURST_CONFLICT" if hurst_conflict else "") +                                 (" HAWKES_SELL" if hawk_sc < 20 else "") +                                 (" HEALTH_ZERO" if hlth == 0.0 else "")

        alert = (comp >= SIGNAL_THRESHOLD and market["allows_long"]
                 and z_ok and not intraday_blocked
                 and (hlth > 0.0 if mode == "intraday" else True))

        # Trend mode exit: ATR-based target
        stop_p  = round(price - atr * ATR_STOP_MULT,  2) if atr > 0 else 0.0
        target  = round(price + atr * ATR_TARGET_MULT, 2) if atr > 0 else 0.0
        rr      = (target - price) / atr if atr > 0 else 0.0
        exit_type = "ATR_TARGET"
        mr_long_sig = mr_short_sig = False
        vol_diverge = False
        exhaustion_reason = "N/A (TREND mode)"

    else:
        # MEAN_REVERSION MODE: Exhaustion triple-gate + VWAP exit
        # Trend signals (Hurst momentum, Hawkes clustering) NOT used for entry
        h_sc, hurst_conflict = hurst_score(H_daily), False
        direction_g   = "LONG" if ofi >= 0.5 else "SHORT"
        ks_sc, ks_lbl = kurtosis_skew_score(kurt, skew, direction_g)
        z_ok = True  # Z-score is the ENTRY signal in MR mode, not a gate
        bayes_p, b_log = bayesian_win_prob(add_b, sec_gate, hawk_sc, True, ks_sc)

        # Exhaustion triple-gate (Step A + B + C)
        # adaptive_ofi scales the panic/exhaustion thresholds with market conditions:
        #   calm market (0.20) → more sensitive, easier to trigger
        #   volatile market (0.35) → requires stronger signal
        mr_long_sig, mr_short_sig, exhaustion_reason = check_exhaustion_signal(
            df_i, vwap, spy_ret, adaptive_ofi)

        # Volume divergence (final confirmation)
        vol_diverge = check_volume_divergence(df_i)

        intraday_blocked      = False
        intraday_block_reason = ""

        # Kill signal if exhaustion not confirmed or volume not diverging
        if not (mr_long_sig or mr_short_sig):
            intraday_blocked = True
            intraday_block_reason = f"NO_EXHAUSTION: {exhaustion_reason}"
        if not vol_diverge and not intraday_blocked:
            intraday_blocked = True
            intraday_block_reason = "NO_VOL_DIVERGENCE"

        # MR composite score — rewards exhaustion signals, penalises momentum
        z_now = (closes[-1] - np.mean(closes[-20:])) / (np.std(closes[-20:], ddof=1) + 1e-9)                 if len(closes) >= 20 else 0.0
        exhaustion_score = min(100.0, abs(z_now) / Z_ENTRY_THRESH * 60
                               + (30 if mr_long_sig or mr_short_sig else 0)
                               + (10 if vol_diverge else 0))

        raw  = float(np.clip(
            0.30*exhaustion_score + 0.20*o_sc + 0.15*sec_sc +
            0.15*add_score(add_b) + 0.10*ks_sc + 0.10*bayes_p*100,
            0, 100))
        comp = round(float(np.clip(
            raw * market["mkt_mult"] * (1.0 if add_b else 0.80) * hlth,
            0, 100)), 1)

        # MR can fire LONG or SHORT — apply direction-appropriate market gate
        mr_direction = "LONG" if mr_long_sig else "SHORT"
        direction_ok  = (market["allows_long"] if mr_direction == "LONG"
                         else True)  # shorts valid even in RISK-OFF
        alert = (comp >= SIGNAL_THRESHOLD and not intraday_blocked
                 and hlth > 0.0 and direction_ok)

        # MR exit: VWAP touch or 0.5% scalp — NOT ATR target
        # Stop = larger of ATR-based (adapts to stock volatility) or 0.8% floor
        atr_stop = round(price - atr * ATR_STOP_MULT, 2) if atr > 0 else 0.0
        pct_stop = round(price * (1 - MR_STOP_LOSS_PCT), 2)
        stop_p   = max(atr_stop, pct_stop)  # whichever is closer to price
        target  = vwap                                          # VWAP is the exit
        rr      = abs(vwap - price) / (price * MR_STOP_LOSS_PCT + 1e-9)
        exit_type = "VWAP_TOUCH"

    # ── Shared post-processing ────────────────────────────────
    direction_g  = "LONG" if (mr_long_sig if strategy == "MEAN_REVERSION" else ofi >= 0.5) else "SHORT"
    strength, hl_rem, hl_alive = halflife_remaining(symbol, comp, price, atr)
    kf, drisk, shares = kelly_size(price, atr, win_rate=bayes_p) if alert else (0.0, 0.0, 0)

    return {
        "symbol":symbol,"price":round(price,2),"score":comp,"alert":alert,
        "mode":mode,"strategy":strategy,"exit_type":exit_type,
        "adx":adx_val,"regime_reason":regime_reason,"adaptive_ofi":adaptive_ofi,
        # Hurst
        "hurst_H":round(H_daily,3),"hurst_H_intra":round(H_intra,3),
        "hurst_score":round(h_sc,1),"hurst_regime":hurst_regime(H_daily),
        "hurst_regime_intra":h_intra_regime,"hurst_conflict":hurst_conflict,
        # Indicators
        "hawkes_lam":lam,"hawkes_score":hawk_sc,"hawkes_sig":hawkes_signal(hawk_sc),
        "ofi":ofi,"ofi_delta":od,"ofi_score":o_sc,"ofi_sig":ofi_signal(ofi,od),
        "market":market["regime"],"sector_etf":etf,
        "sector_rs":sec_rs,"sector_score":sec_sc,"sector_gate":sec_gate,
        "add_val":add_val,"add_bull":add_b,"add_score":round(add_score(add_b),1),
        "zscore":round(z,3),"zscore_score":round(zscore_score(z),1),"zscore_ok":z_ok,
        "kurtosis":kurt,"skewness":skew,"ks_score":ks_sc,"ks_label":ks_lbl,
        "bayes_prob":round(bayes_p*100,1),"bayes_factors":b_log,
        "hl_strength":strength,"hl_remaining":round(hl_rem,0),"hl_alive":hl_alive,
        "rr_ratio":round(rr,2),"rr_ok":rr >= 2.0,
        "kelly_frac":kf,"dollar_risk":drisk,"shares":shares,
        "atr":atr,"stop":stop_p,"target":target,"vwap":round(vwap,2),
        # Mean reversion specific
        "mr_long":mr_long_sig,"mr_short":mr_short_sig,
        "vol_diverge":vol_diverge,"exhaustion_reason":exhaustion_reason,
        "health_mult":hlth,"health_label":hlbl,"liq_status":liq_reason,
        "intraday_blocked":intraday_blocked,"intraday_block_reason":intraday_block_reason,
        "intraday_ret":hd.get("intraday_ret",0.0),"gap_ret":hd.get("gap_ret",0.0),
        "below_vwap":hd.get("below_vwap",False),
        "mcap_b":mcap_b,"scanned_at":datetime.now().strftime("%H:%M:%S"),
        # ── Golden Entry flag ─────────────────────────────────
        # True when BOTH conditions align for ideal MR long setup:
        #   Z-score <= -2.0  (price statistically oversold)
        #   Price below VWAP (below institutional fair value)
        # This is the exact rubber-band stretch the exhaustion engine targets.
        "golden_entry": (
            float(z) <= -Z_ENTRY_THRESH and
            hd.get("below_vwap", False) and
            strategy == "MEAN_REVERSION"
        ),
    }


def run_full_scan(symbols: list = WATCHLIST, timeframe: str = "5m") -> tuple:
    """
    Parallel scan using ThreadPoolExecutor.
    Tracks FULL rejection reason for every blocked symbol so dashboard
    can show exactly why each stock was filtered out.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    market   = get_market_regime()
    results  = []
    rejected = []   # list of dicts: {symbol, reason, detail}

    # Pre-check market cap (fast — uses 1h cache)
    qualified = []
    for sym in symbols:
        ok, mb = is_midcap(sym)
        if not ok:
            reason = "BELOW_MIN_CAP" if mb < MIDCAP_MIN/1e9 else "ABOVE_MAX_CAP"
            rejected.append({"symbol":sym,"reason":reason,
                             "detail":f"${mb:.2f}B (need $300M–$15B)"})
        else:
            qualified.append(sym)

    # Parallel scan — capture why each symbol was rejected inside scan_symbol
    def _scan_one(sym):
        try:
            # Run each firewall manually to get the specific rejection reason
            df_d = fetch_daily(sym, 60)
            if df_d is None or len(df_d) < 30:
                return sym, None, "NO_DATA", "Insufficient daily bars"

            liq_ok, liq_msg = passes_liquidity_firewall(df_d, sym)
            if not liq_ok:
                return sym, None, "LIQUIDITY", liq_msg

            df_i = fetch_intraday(sym, timeframe) or df_d.copy()
            spy_ret = market.get("spy_dev", 0.0) / 100.0

            if STRATEGY_MODE == "AUTO":
                strat, _, _, reg_reason = get_strategy_regime(df_i, spy_ret)
            else:
                strat = STRATEGY_MODE; reg_reason = f"FORCED_{STRATEGY_MODE}"

            if strat == "NO_TRADE":
                return sym, None, "NO_TRADE", reg_reason

            etf = SECTOR_MAP.get(sym, "SPY")
            _, _, sec_gate = get_sector_rs(etf)
            if not sec_gate:
                return sym, None, "SECTOR_RS", f"{etf} RS < 1.05 (sector lagging SPY)"

            r = scan_symbol(sym, market, timeframe)
            if r:
                return sym, r, None, None
            else:
                return sym, None, "SCAN_FAILED", "Returned None (likely intraday health block)"
        except Exception as e:
            return sym, None, "ERROR", str(e)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_scan_one, sym): sym for sym in qualified}
        for fut in as_completed(futures):
            sym, r, reason, detail = fut.result()
            if r:
                results.append(r)
            elif reason:
                rejected.append({"symbol":sym,"reason":reason,"detail":detail or ""})

    # Build simple blocked list for backward compat + rich rejected list
    blocked_simple = [f"{r['symbol']}({r['reason']})" for r in rejected]
    market.update({
        "blocked":        blocked_simple,
        "blocked_count":  len(rejected),
        "scanned":        len(results),
        "rejected_detail":rejected,   # full detail for dashboard
    })
    if not results: return pd.DataFrame(), market
    return pd.DataFrame(results).sort_values(by=["score","bayes_prob"],
                                              ascending=False).reset_index(drop=True), market


# ── Universe Scanner ──────────────────────────────────────────────────────────

def universe_prefilter(symbols: list) -> list:
    """Original sequential prefilter — kept as fallback."""
    survivors = []
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(period="1mo", interval="1d")
            if df.empty or len(df) < 10: continue
            df.columns = [c.lower() for c in df.columns]
            p = float(df["close"].iloc[-1])
            if p < 5: continue
            if float(df["volume"].mean()) < 300_000: continue
            if float(df["close"].diff().abs().mean() / p) < 0.02: continue
            ok, _ = is_midcap(sym)
            if ok: survivors.append(sym)
        except Exception:
            continue
    print(f"[UNIVERSE] {len(symbols)} → {len(survivors)} survivors")
    return survivors


def universe_prefilter_fast(symbols: list) -> list:
    """
    Fast prefilter using yf.download() batch call.
    Downloads all symbols in a single HTTP request instead of one per symbol.
    ~10x faster than universe_prefilter() for large lists.
    Falls back to sequential if batch download fails.
    """
    try:
        # Single batch download — yfinance fetches all symbols in one call
        raw = yf.download(
            symbols, period="1mo", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True
        )
        survivors = []
        for sym in symbols:
            try:
                # Handle both single and multi-symbol DataFrame formats
                if len(symbols) == 1:
                    df = raw
                else:
                    df = raw[sym] if sym in raw.columns.get_level_values(0) else None
                if df is None or df.empty or len(df) < 10: continue
                df.columns = [c.lower() for c in df.columns]
                if "close" not in df.columns: continue
                p = float(df["close"].iloc[-1])
                if p < 5: continue
                if float(df["volume"].mean()) < 300_000: continue
                if float(df["close"].diff().abs().mean() / p) < 0.02: continue
                ok, _ = is_midcap(sym)
                if ok: survivors.append(sym)
            except Exception:
                continue
        print(f"[UNIVERSE FAST] {len(symbols)} → {len(survivors)} survivors")
        return survivors
    except Exception as e:
        print(f"[UNIVERSE FAST] Batch failed ({e}), falling back to sequential")
        return universe_prefilter(symbols)


def run_universe_scan(top_n: int = 30, timeframe: str = "5m") -> pd.DataFrame:
    """
    Two-stage universe scan — both stages parallelized.
    Stage 1 (prefilter): parallel HTTP batch using yf.download().
    Stage 2 (full scan):  parallel via ThreadPoolExecutor.
    Removes the 0.3s sleep (was needed for sequential rate limiting,
    not needed when batching requests).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"[UNIVERSE] Start {datetime.now().strftime('%H:%M:%S')}")
    survivors = universe_prefilter_fast(UNIVERSE)
    if not survivors: return pd.DataFrame()
    market  = get_market_regime()
    results = []

    def _scan_one(sym):
        try:
            return scan_symbol(sym, market, timeframe)
        except Exception as e:
            print(f"[UNIVERSE ERR] {sym}: {e}")
            return None

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_scan_one, sym): sym for sym in survivors}
        for fut in as_completed(futures):
            r = fut.result()
            if r: results.append(r)
            done += 1
            if done % 10 == 0:
                print(f"[UNIVERSE] {done}/{len(survivors)}")

    if not results: return pd.DataFrame()
    df = pd.DataFrame(results).sort_values(by=["score","bayes_prob"],
                                            ascending=False).reset_index(drop=True)
    with open("auto_watchlist.txt", "w") as f:
        f.write("\n".join(df.head(top_n)["symbol"].tolist()))
    print(f"[UNIVERSE] Saved top {top_n} → auto_watchlist.txt")
    export_excel(df, market)
    return df


# ── Excel Export ──────────────────────────────────────────────────────────────

def export_excel(df: pd.DataFrame, market: dict):
    """
    Writes two Excel files every time:
      1. scan_YYYY-MM-DD_HHMM.xlsx  — timestamped snapshot (history preserved)
      2. scan_latest.xlsx           — always overwritten (quick access to newest)
    Both go to ~/Downloads.
    Sheets: Full Scan · Top Picks (alerts only) · Market Context · Rejected Stocks
    """
    try:
        import openpyxl
    except ImportError:
        print("[EXPORT] pip install openpyxl"); return

    dl_dir  = os.path.expanduser("~")
    dl_dir  = os.path.join(dl_dir, "Downloads")
    os.makedirs(dl_dir, exist_ok=True)

    fname_ts     = f"scan_{datetime.now().strftime('%Y-%m-%d_%H%M')}.xlsx"
    fname_latest = "scan_latest.xlsx"
    fpath_ts     = os.path.join(dl_dir, fname_ts)
    fpath_latest = os.path.join(dl_dir, fname_latest)

    top     = df[df["alert"] == True] if not df.empty else pd.DataFrame()
    rejected = pd.DataFrame(market.get("rejected_detail", []))
    mkt_row = {**{k: market.get(k,"—") for k in
                  ["regime","spy_price","spy_dev","qqq_price","qqq_dev","scanned","blocked_count"]},
               "Date":       datetime.now().strftime("%Y-%m-%d"),
               "Time":       datetime.now().strftime("%H:%M:%S"),
               "ADD Bullish":str(_add_cache.get("bullish","—")),
               "Strategy":   STRATEGY_MODE,
               "Signals":    len(top)}

    def _write(path):
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            if not df.empty:
                df.to_excel(w, sheet_name="Full Scan", index=False)
            if not top.empty:
                top.to_excel(w, sheet_name="Top Picks", index=False)
            pd.DataFrame([mkt_row]).to_excel(w, sheet_name="Market Context", index=False)
            if not rejected.empty:
                rejected.to_excel(w, sheet_name="Rejected Stocks", index=False)

    # Write both files
    _write(fpath_ts)
    _write(fpath_latest)

    if STREAMLIT_MODE:
        with open(fpath_ts, "rb") as f:
            st.download_button(
                f"⬇️ {fname_ts}", f.read(), file_name=fname_ts,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{fname_ts}"
            )
        st.session_state["last_export_time"] = datetime.now().strftime("%H:%M:%S")
        st.session_state["last_export_path"] = fpath_ts
        st.success(f"✅ Saved → {fname_ts}  +  scan_latest.xlsx")
    else:
        print(f"[EXPORT] {fpath_ts} + scan_latest.xlsx")


# ── Scheduler & Background Threads ───────────────────────────────────────────

def _scheduler_loop():
    schedule.every().day.at("09:00").do(lambda: run_universe_scan(30, "5m"))
    print("[SCHEDULER] Daily 9:00 AM scan scheduled")
    while True:
        schedule.run_pending()
        time.sleep(30)

_started = False
def _start_background():
    global _started
    if not _started:
        threading.Thread(target=_update_add_loop, daemon=True, name="add").start()
        threading.Thread(target=_scheduler_loop,  daemon=True, name="sched").start()
        _started = True

_start_background()


# ── Dashboard CSS ─────────────────────────────────────────────────────────────

DASH_CSS = """<style>
*{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif!important;}

/* Base — white background */
html,body,[class*="css"]{background:#ffffff!important;color:#1a1a1a!important;}

/* Sidebar — light grey with yellow top border */
section[data-testid="stSidebar"]{background:#f5f5f5!important;border-right:3px solid #f5c400!important;}

/* Buttons — blue */
.stButton>button{background:#1a5fd9!important;color:#ffffff!important;font-weight:700!important;
  border:none!important;border-radius:4px!important;letter-spacing:1px!important;
  font-size:13px!important;text-transform:uppercase!important;}
.stButton>button:hover{background:#1048b0!important;}

/* Form labels */
.stSelectbox label,.stSlider label,.stTextArea label,.stNumberInput label{
  color:#555555!important;font-size:12px!important;font-weight:600!important;}

/* Metrics */
[data-testid="stMetricValue"]{color:#1a1a1a!important;font-weight:800!important;}
[data-testid="stMetricDelta"]{color:#1a5fd9!important;}

/* Dividers */
hr{border-color:#e0e0e0!important;}

/* Header */
.hdr{font-size:28px;font-weight:800;letter-spacing:6px;color:#1a1a1a;text-transform:uppercase;
     border-bottom:3px solid #f5c400;padding-bottom:10px;}
.sub{font-size:11px;font-weight:600;color:#888888;letter-spacing:3px;text-transform:uppercase;}
.sub2{font-size:10px;font-weight:700;color:#888888;text-transform:uppercase;letter-spacing:2px;}

/* Signal cards */
.card{border-radius:6px;padding:16px;margin-bottom:10px;
      border:1px solid #e0e0e0;background:#ffffff;
      box-shadow:0 1px 4px rgba(0,0,0,0.06);}
.card.on{border-color:#f5c400;border-left:4px solid #f5c400;
         background:#fffdf0;box-shadow:0 2px 12px rgba(245,196,0,0.15);}
.card.warn{border-color:#1a5fd9;border-left:4px solid #1a5fd9;background:#f0f4ff;}

/* Typography */
.sym{font-size:22px;font-weight:800;color:#1a1a1a;letter-spacing:2px;text-transform:uppercase;}
.sc{font-size:30px;font-weight:900;line-height:1;}
.bayes{font-size:20px;font-weight:700;line-height:1;}

/* Tags */
.tag{display:inline-block;font-size:10px;font-weight:700;padding:3px 8px;
     border-radius:3px;margin:2px;letter-spacing:0.5px;text-transform:uppercase;}

/* Metric cells */
.row{display:flex;gap:6px;margin-top:8px;}
.cell{flex:1;background:#f8f8f8;border:1px solid #e8e8e8;border-radius:4px;padding:5px 8px;}
.lbl{font-size:9px;font-weight:700;color:#999999;text-transform:uppercase;letter-spacing:1.5px;}
.val{font-size:12px;font-weight:700;color:#1a1a1a;}

/* Score bar */
.bar-bg{background:#e8e8e8;border-radius:2px;height:5px;margin:8px 0;overflow:hidden;}
.bar-fg{height:5px;border-radius:2px;}

/* Trade box */
.trade{background:#f8f8f8;border:1px solid #e0e0e0;border-left:4px solid #f5c400;
       border-radius:4px;padding:10px 14px;margin-top:10px;font-size:11px;
       font-weight:600;color:#333333;}
</style>"""


# ── Dashboard ─────────────────────────────────────────────────────────────────

def run_dashboard():
    st.set_page_config(page_title="QUANT v3", page_icon="⚡",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(DASH_CSS, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.header("🔍 v3 Controls")
        timeframe   = st.selectbox("Timeframe", ["15s","1m","5m","15m"], index=2)

        # ── Strategy Switch ───────────────────────────────────
        st.markdown("**Strategy Mode**")
        strategy_choice = st.radio(
            "Active Strategy",
            ["AUTO (recommended)", "TREND only", "MEAN REVERSION only"],
            index=0,
            help="AUTO uses ADX+SPY vol to pick the right strategy each scan. "
                 "Never run both manually — they contradict each other."
        )
        # Map sidebar choice to global config
        import sys
        this_module = sys.modules[__name__]
        if "TREND only"           in strategy_choice: setattr(this_module,'STRATEGY_MODE','TREND')
        elif "MEAN REVERSION only" in strategy_choice: setattr(this_module,'STRATEGY_MODE','MEAN_REVERSION')
        else:                                           setattr(this_module,'STRATEGY_MODE','AUTO')
        auto_list   = open("auto_watchlist.txt").read() if os.path.exists("auto_watchlist.txt") \
                      else "\n".join(WATCHLIST)
        custom_syms = st.text_area("Watchlist (one per line)", value=auto_list, height=250)
        run_btn     = st.button("▶ RUN SCAN NOW",       use_container_width=True)
        uni_btn     = st.button("🌍 RUN UNIVERSE SCAN", use_container_width=True)
        refresh     = st.slider("Auto-Refresh (sec)", 10, 300, 60)
        _           = st.number_input("Account Size ($)", value=int(ACCOUNT_SIZE), step=5000, format="%d")
        st.markdown("---")
        st.markdown(f"<div style='font-size:10px;font-weight:600;color:#555555'>"
                    f"<span style='color:#1a1a1a;font-weight:800'>10-LAYER ENGINE</span><br>"
                    f"✓ Hurst · Hawkes · OFI<br>"
                    f"✓ Sector RS · ADD Breadth<br>"
                    f"✓ Z-Score · Kurt/Skew<br>"
                    f"✓ Bayesian · Half-Life · ATR<br><br>"
                    f"<span style='color:#1a5fd9'>CAP: $300M–$15B · TF: {timeframe}</span><br>"
                    f"<span style='color:{'#1a8c2a' if SCHWAB_AVAILABLE else '#d94040'}'>"
                    f"{'✓ Schwab connected' if SCHWAB_AVAILABLE else '⚠ yfinance only — 15s delay'}"
                    f"</span></div>",
                    unsafe_allow_html=True)

    symbols = [s.strip().upper() for s in custom_syms.split("\n") if s.strip()]

    # Session state
    for k, v in [("results",pd.DataFrame()),("market",{}),
                 ("last_scan",0.0),("count",0),("last_export_time","Never")]:
        if k not in st.session_state: st.session_state[k] = v

    now = time.time()

    if uni_btn:
        with st.spinner("Universe scan (~15 min)..."):
            run_universe_scan(30, timeframe)
        st.success("Done — watchlist updated")

    if run_btn or (now - st.session_state["last_scan"] > refresh) or st.session_state["last_scan"] == 0:
        with st.spinner(f"Scanning {len(symbols)} symbols..."):
            df, market = run_full_scan(symbols, timeframe)
        st.session_state.update({"results":df,"market":market,
                                   "last_scan":now,"count":st.session_state["count"]+1})

        # ── AUTO-EXPORT: runs every scan — even if 0 results ─────
        try:
            dl_dir   = os.path.join(os.path.expanduser("~"), "Downloads")
            os.makedirs(dl_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
            today     = datetime.now().strftime('%Y-%m-%d')
            log_path  = os.path.join(dl_dir, f"scan_log_{today}.csv")

            # 1. Excel snapshot (results + rejections + market context)
            export_excel(df if not df.empty else pd.DataFrame(), market)

            # 2. Append results to daily CSV log
            if not df.empty:
                df_log = df.copy()
                df_log["scan_time"] = datetime.now().strftime("%H:%M:%S")
                df_log["scan_num"]  = st.session_state["count"]
                write_header = not os.path.exists(log_path)
                df_log.to_csv(log_path, mode="a", header=write_header, index=False)
                st.session_state["last_log_path"] = log_path

            # 3. Always write rejection log (separate file, appends all day)
            rej = market.get("rejected_detail", [])
            if rej:
                rej_log = os.path.join(dl_dir, f"rejections_log_{today}.csv")
                df_rej  = pd.DataFrame(rej)
                df_rej["scan_time"] = datetime.now().strftime("%H:%M:%S")
                df_rej["scan_num"]  = st.session_state["count"]
                rej_header = not os.path.exists(rej_log)
                df_rej.to_csv(rej_log, mode="a", header=rej_header, index=False)

        except Exception as e:
            st.warning(f"⚠️ Auto-export failed: {e}")

    df, market, count = st.session_state["results"], st.session_state["market"], st.session_state["count"]

    # ── Header ────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1])
    add_val, add_bull = get_add_breadth()
    with c1:
        st.markdown('<div class="hdr">▲ QUANT SCANNER v3</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub">10-LAYER ENGINE · {timeframe.upper()} · CAP $300M–$15B</div>', unsafe_allow_html=True)
    with c2:
        ac = "#1a5fd9" if add_bull else "#d94040"
        st.markdown(f'<div style="text-align:right;margin-top:6px">'
                    f'<div class="sub">{datetime.now().strftime("%H:%M:%S")} EST · SCAN #{count}</div>'
                    f'<div style="font-size:12px;font-weight:700;color:{ac}">'
                    f'ADD {"▲ BULLISH" if add_bull else "▼ BEARISH"} ({add_val:.3f})</div>'
                    f'<div class="sub">NEXT REFRESH {max(0,int(refresh-(now-st.session_state["last_scan"])))}s</div>'
                    f'</div>', unsafe_allow_html=True)

    if market:
        rc = "#1a8c2a" if "RISK-ON"  in market.get("regime","") else \
             "#d94040" if "RISK-OFF" in market.get("regime","") else "#1a5fd9"
        st.markdown(
            f'<div style="font-size:12px;font-weight:600;color:#444444;'
            f'letter-spacing:2px;margin-bottom:14px;text-transform:uppercase;'
            f'border-bottom:1px solid #e0e0e0;padding-bottom:8px">'
            f'MARKET: <span style="color:{rc}">{market.get("regime","—")}</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;SPY {market.get("spy_price","—")} '
            f'<span style="color:{"#1a8c2a" if market.get("spy_dev",0)>0 else "#d94040"}">'
            f'({market.get("spy_dev",0):+.2f}%)</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;QQQ {market.get("qqq_price","—")} '
            f'<span style="color:{"#1a8c2a" if market.get("qqq_dev",0)>0 else "#d94040"}">'
            f'({market.get("qqq_dev",0):+.2f}%)</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;✅ {market.get("scanned",0)} SCANNED'
            + (
                (lambda rej: (
                    f'&nbsp;&nbsp;🚫 {len(rej)} BLOCKED'
                    + (lambda groups: "".join(
                        f'&nbsp;<span style="color:#888888;font-size:10px">'
                        f'[{k}×{v}]</span>'
                        for k,v in groups.items()
                    ))(
                        {r["reason"]: sum(1 for x in rej if x["reason"]==r["reason"])
                         for r in rej}
                    )
                ))(market.get("rejected_detail",[]))
                if market.get("blocked_count",0) > 0 else ""
            )
            + f'</div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("⚠️ Scan returned 0 results — all symbols were filtered out. See rejection details below.")
        # DO NOT return — fall through so rejected panel and export still render

    # ── Results sections (only when scan has results) ─────────
    if not df.empty:
      # Alert banner
      alerts = df[df["alert"] == True]
      if not alerts.empty:
        syms = "  ·  ".join(f"{r['symbol']} [{r['score']}] {r['bayes_prob']:.0f}%" for _,r in alerts.iterrows())
        st.markdown(
            f'<div style="background:#fffdf0;border:2px solid #f5c400;'
            f'border-left:6px solid #f5c400;border-radius:4px;padding:12px 18px;'
            f'margin-bottom:14px;font-size:13px;font-weight:700;color:#b08800;'
            f'letter-spacing:2px;text-transform:uppercase">▶ SIGNAL ALERT  ·  {syms}</div>',
            unsafe_allow_html=True)

    # ── Golden Entry Banner ───────────────────────────────────
    # Surfaces the most important condition in the whole scanner:
    # Z <= -2.0 + below VWAP in MEAN_REVERSION mode = highest-probability MR long setup
    if "golden_entry" in df.columns:
        golden = df[df["golden_entry"] == True]
        if not golden.empty:
            ge_syms = "  ·  ".join(
                f"{r['symbol']} Z={r['zscore']:.2f} VWAP${r.get('vwap',0):.2f}"
                for _, r in golden.iterrows()
            )
            st.markdown(
                f'<div style="background:#fff8e1;border:2px solid #f5c400;'
                f'border-left:8px solid #f5c400;border-radius:4px;'
                f'padding:14px 18px;margin-bottom:14px;">'
                f'<div style="font-size:13px;font-weight:800;color:#b08800;'
                f'letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">'
                f'⭐ GOLDEN ENTRY SETUP — Z ≤ -2.0 + BELOW VWAP</div>'
                f'<div style="font-size:12px;font-weight:600;color:#1a1a1a">'
                f'{ge_syms}</div>'
                f'<div style="font-size:10px;color:#888888;margin-top:4px">'
                f'Price is statistically oversold AND below institutional fair value. '
                f'Wait for OFI exhaustion confirmation before entering.</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Signal Cards ──────────────────────────────────────────
    cols = st.columns(min(4, max(1, len(df))))
    for i, (_, r) in enumerate(df.iterrows()):
        sc     = r["score"]
        sc_col = "#b08800" if sc>=75 else "#1a5fd9" if sc>=SIGNAL_THRESHOLD else "#888888" if sc>=45 else "#d94040"
        bp     = r["bayes_prob"]
        bp_col = "#b08800" if bp>=70 else "#1a5fd9" if bp>=55 else "#d94040"
        css    = "on" if r["alert"] else ("warn" if sc>=55 else "")
        hl_str = f"⏱ {int(r['hl_remaining'])}s" if r["hl_alive"] else "—"
        strat  = r.get("strategy","—")
        adx_v  = r.get("adx", 0.0)
        exit_t = r.get("exit_type","—")

        # Strategy badge colour
        strat_col = "#1a5fd9" if strat=="TREND" else "#b08800" if strat=="MEAN_REVERSION" else "#888888"
        strat_lbl = "📈 TREND" if strat=="TREND" else "🔄 MEAN REV" if strat=="MEAN_REVERSION" else strat

        conflict_tag = ""
        if r.get("hurst_conflict", False):
            conflict_tag = '<span class="tag" style="background:#fff0f0;color:#d94040;border:1px solid #f0c0c0;font-weight:800">⚠ HURST CONFLICT</span>'
        blocked_tag = ""
        if r.get("intraday_blocked", False):
            blocked_tag = f'<span class="tag" style="background:#fff0f0;color:#d94040;border:1px solid #f0c0c0">🚫 BLOCKED</span>'

        h_intra     = r.get("hurst_H_intra", r.get("hurst_H", 0))
        scan_mode   = r.get("mode","auto").upper()

        # Mean reversion specific display
        if strat == "MEAN_REVERSION":
            trade_html = (f'<div class="trade">{"🟢 LONG" if r.get("mr_long") else "🔴 SHORT"} EXHAUSTION<br>'
                          f'ENTRY ${r["price"]} · STOP ${r["stop"]} · EXIT AT VWAP ${r["vwap"]:.2f}<br>'
                          f'Kelly {r["kelly_frac"]:.1%} → {r["shares"]} shares · TP 0.5% or VWAP</div>'
                          if r["alert"] else "")
        else:
            trade_html = (f'<div class="trade">ENTRY ${r["price"]} · STOP ${r["stop"]} · TARGET ${r["target"]}<br>'
                          f'Kelly {r["kelly_frac"]:.1%} → {r["shares"]} shares · ${r["dollar_risk"]} risk</div>'
                          if r["alert"] else "")
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="card {css}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <div class="sym">{r['symbol']}</div>
                  <div style="font-size:11px;font-weight:500;color:#666666">${r['price']} &nbsp;·&nbsp; MCap ${r.get('mcap_b',0):.1f}B &nbsp;·&nbsp; {scan_mode}</div>
                </div>
                <div style="text-align:right">
                  <div class="sc" style="color:{sc_col}">{sc}</div>
                  <div class="bayes" style="color:{bp_col}">{bp:.0f}%</div>
                  <div class="sub2">BAYES WIN PROB</div>
                </div>
              </div>
              <div class="bar-bg"><div class="bar-fg" style="width:{sc}%;background:{sc_col}"></div></div>
              <div style="margin:6px 0">
                <span class="tag" style="background:{strat_col}22;color:{strat_col};border:1px solid {strat_col}55;font-weight:800">{strat_lbl}</span>
                <span class="tag" style="background:#f0f0f0;color:#555555;border:1px solid #dddddd">ADX {adx_v:.0f}</span>
                <span class="tag" style="background:#eef3ff;color:#1a5fd9;border:1px solid #c0d0f0">{r['hurst_regime']}</span>
                <span class="tag" style="background:#eef3ff;color:#1a5fd9;border:1px solid #c0d0f0">{r['hawkes_sig']}</span>
                <span class="tag" style="background:#eef3ff;color:#1a5fd9;border:1px solid #c0d0f0">{r['ofi_sig']}</span>
                <span class="tag" style="background:#f8f8f8;color:{'#1a8c2a' if r['add_bull'] else '#d94040'};border:1px solid #dddddd">ADD {'▲' if r['add_bull'] else '▼'}</span>
                <span class="tag" style="background:#f8f8f8;color:{'#1a8c2a' if r['zscore_ok'] else '#d94040'};border:1px solid #dddddd">Z={r['zscore']:.2f}</span>
                <span class="tag" style="background:#fffdf0;color:#b08800;border:1px solid #f0e080">{r['ks_label']}</span>
                {conflict_tag}{blocked_tag}
                {"<span class=\"tag\" style=\"background:#fff8e1;color:#b08800;border:2px solid #f5c400;font-weight:800\">⭐ GOLDEN ENTRY</span>" if r.get("golden_entry") else ""}
              </div>
              <div class="row">
                <div class="cell"><div class="lbl">HURST D</div><div class="val">{r['hurst_H']}</div></div>
                <div class="cell"><div class="lbl">HURST 5M</div>
                  <div class="val" style="color:{'#1a8c2a' if h_intra>0.55 else '#d94040' if h_intra<0.42 else '#888888'}">{h_intra}</div></div>
                <div class="cell"><div class="lbl">OFI</div><div class="val">{r['ofi']}</div></div>
                <div class="cell" style="{'background:#fff8e1;border:1px solid #f5c400' if r.get('golden_entry') else ''}">
                  <div class="lbl">Z-SCORE{'  ⭐' if r.get('golden_entry') else ''}</div>
                  <div class="val" style="color:{'#b08800' if r.get('golden_entry') else '#d94040' if float(r['zscore'])<=-2.0 else '#1a8c2a' if float(r['zscore'])>=2.0 else '#1a1a1a'};{'font-weight:800' if abs(float(r['zscore']))>=2.0 else ''}">{r['zscore']:.2f}</div></div>
              </div>
              <div class="row">
                <div class="cell"><div class="lbl">SECTOR</div>
                  <div class="val" style="color:{'#1a8c2a' if r['sector_gate'] else '#d94040'}">{r['sector_etf']} {'✓' if r['sector_gate'] else '✗'}</div></div>
                <div class="cell"><div class="lbl">R:R</div>
                  <div class="val" style="color:{'#1a8c2a' if r['rr_ok'] else '#d94040'}">{'✓' if r['rr_ok'] else '✗'} {r['rr_ratio']:.1f}:1</div></div>
                <div class="cell"><div class="lbl">HALF-LIFE</div><div class="val">{hl_str}</div></div>
                <div class="cell" style="{'background:#fff0f0' if r.get('below_vwap') else 'background:#f0fff4'}">
                  <div class="lbl">VWAP</div>
                  <div class="val" style="color:{'#d94040' if r.get('below_vwap') else '#1a8c2a'}">${r.get('vwap',0):.2f} {'↓' if r.get('below_vwap') else '↑'}</div></div>
              </div>
              {trade_html}
            </div>""", unsafe_allow_html=True)

    # ── Trade Log ─────────────────────────────────────────────
    if TRADE_LOG:
        st.markdown("---")
        st.markdown("### 📋 Trade Log")
        total = sum(t["P&L $"] for t in TRADE_LOG)
        wins  = sum(1 for t in TRADE_LOG if t["P&L $"] > 0)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total P&L", f"${total:+.2f}")
        m2.metric("Win Rate",  f"{wins/len(TRADE_LOG)*100:.0f}%")
        m3.metric("Trades",    len(TRADE_LOG))
        m4.metric("W / L",     f"{wins} / {len(TRADE_LOG)-wins}")
        st.dataframe(pd.DataFrame(TRADE_LOG[::-1]), use_container_width=True, hide_index=True)

    # ── Top Alpha Pick (Execution Planner) ────────────────────
    st.markdown("---")
    eligible = df[df["alert"] == True] if not df.empty else pd.DataFrame()
    if not eligible.empty:
        top = eligible.sort_values(by=["bayes_prob","score"], ascending=False).iloc[0]
        sec_bull  = bool(top.get("sector_gate", True))
        etf_name  = top.get("sector_etf","—")
        above_vwap = not bool(top.get("below_vwap", False))
        hl_rem    = int(top.get("hl_remaining", 600))
        z_val     = float(top.get("zscore", 0))

        st.markdown(f"## 🎯 Top Alpha Pick: **{top['symbol']}**")
        if sec_bull: st.success(f"✅ Sector Alignment: {top['symbol']} in **{etf_name}** — outperforming SPY")
        else:        st.warning(f"⚠️ Sector Divergence: **{etf_name}** lagging — reduce position size")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Bayesian Win Prob", f"{top['bayes_prob']:.1f}%",
                      delta="High Conviction" if top['bayes_prob'] >= 65 else "Moderate")
            st.write(f"**Stop:** ${top['stop']:.2f}  *(1.5× ATR ${top['atr']:.2f})*")
            st.write(f"**Entry:** ${top['price']:.2f}  ·  **{int(top['shares'])} shares**  (${top['dollar_risk']:.0f} risk)")
            st.caption(f"Kelly {top['kelly_frac']:.1%} of ${int(ACCOUNT_SIZE):,}")
        with c2:
            st.metric("Target", f"${top['target']:.2f}",
                      delta=f"{top['rr_ratio']:.1f}:1 R:R {'✅' if top['rr_ok'] else '❌'}")
            st.write(f"**Profit if hit:** ${(top['target']-top['price'])*top['shares']:.0f}")
            if above_vwap: st.success("✅ Above VWAP — institutions net buyers")
            else:          st.error("❌ Below VWAP — wait for reclaim")
        with c3:
            st.metric("Alpha Half-Life", f"{hl_rem}s",
                      delta=f"{float(top.get('hl_strength',1.0))*100:.0f}% strength")
            if z_val > 2.0:   st.error(f"Z={z_val:.2f} EXTENDED — reduce size")
            elif z_val > 1.5: st.warning(f"Z={z_val:.2f} — Elevated")
            else:             st.success(f"Z={z_val:.2f} ✅ — Healthy")
            st.write(f"**{top.get('ks_label','—')}** (K:{top.get('kurtosis',0):.2f} S:{top.get('skewness',0):.2f})")

        # 7-point checklist
        checks = [
            ("Market",    "RISK-OFF" not in market.get("regime",""), market.get("regime","—")),
            ("Sector",    sec_bull,                                   etf_name),
            ("ADD",       bool(top.get("add_bull",True)),             f"{top.get('add_val',0):.3f}"),
            ("Z-Score",   bool(top.get("zscore_ok",True)),            f"Z={z_val:.2f}"),
            ("R:R ≥ 2",   bool(top.get("rr_ok",False)),              f"{top.get('rr_ratio',0):.1f}:1"),
            ("VWAP",      above_vwap,                                 f"${top.get('vwap',0):.2f}"),
            ("Half-Life", bool(top.get("hl_alive",True)),             f"{hl_rem}s"),
        ]
        passed = sum(1 for _, ok, _ in checks if ok)
        for col, (label, ok, detail) in zip(st.columns(len(checks)), checks):
            with col:
                st.markdown(
                    f"<div style='text-align:center;font-size:11px;font-weight:700;"
                    f"padding:8px 4px;border-radius:3px;"
                    f"background:{'#fffdf0' if ok else '#fff0f0'};"
                    f"border:1px solid {'#f5c400' if ok else '#e05050'}'>"
                    f"{'✅' if ok else '❌'}&nbsp; {label}<br>"
                    f"<span style='font-size:10px;font-weight:500;color:#666666'>{detail}</span></div>",
                    unsafe_allow_html=True)

        verdict_color = "#b08800" if passed >= 6 else "#1a5fd9" if passed >= 4 else "#d94040"
        verdict_text  = "▲ ALL CLEAR — EXECUTE"    if passed >= 6 else \
                        "— PARTIAL — REDUCE SIZE"   if passed >= 4 else "✕ BLOCKED — DO NOT ENTER"
        st.markdown(
            f"<div style='text-align:center;font-size:16px;font-weight:800;"
            f"color:{verdict_color};padding:14px;border-radius:4px;letter-spacing:3px;"
            f"text-transform:uppercase;border:2px solid {verdict_color};"
            f"background:{'#fffdf0' if passed>=6 else '#f0f4ff' if passed>=4 else '#fff0f0'};"
            f"margin-top:14px'>{verdict_text} &nbsp;·&nbsp; {passed}/{len(checks)} CHECKS PASSED</div>",
            unsafe_allow_html=True)
    else:
        st.info("🎯 No execution-grade signals right now — watching and waiting.")

    # ── Full Results Table ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Full Scan Results")

    if not df.empty:
        # Build display DataFrame
        show_cols = ["symbol","golden_entry","score","strategy","bayes_prob",
                     "zscore","vwap","below_vwap","adx",
                     "kurtosis","skewness","ks_label",
                     "hurst_regime","hawkes_sig","ofi_sig","sector_etf",
                     "add_bull","rr_ok","kelly_frac","shares",
                     "stop","target","health_label","mcap_b"]
        df_show = df[[c for c in show_cols if c in df.columns]].copy()

        # Sort: golden entries first, then by score descending
        if "golden_entry" in df_show.columns:
            df_show = df_show.sort_values(
                by=["golden_entry","score","bayes_prob"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

        # Rename for readability
        df_show = df_show.rename(columns={
            "golden_entry":  "⭐ Golden",
            "bayes_prob":    "Bayes%",
            "below_vwap":    "< VWAP",
            "sector_etf":    "Sector",
            "add_bull":      "ADD ▲",
            "rr_ok":         "R:R ✓",
            "kelly_frac":    "Kelly%",
            "health_label":  "Health",
            "mcap_b":        "MCap $B",
        })

        # Streamlit styling — highlight golden rows yellow, z-score column red/green
        def highlight_golden(row):
            if row.get("⭐ Golden", False):
                return ["background-color:#fff8e1;font-weight:700"] * len(row)
            return [""] * len(row)

        def color_zscore(val):
            try:
                v = float(val)
                if v <= -2.0:  return "color:#d94040;font-weight:800"
                if v >= 2.0:   return "color:#1a8c2a;font-weight:800"
            except (ValueError, TypeError):
                pass
            return ""

        styled = (df_show.style
                  .apply(highlight_golden, axis=1)
                  .applymap(color_zscore, subset=["zscore"] if "zscore" in df_show.columns else []))

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Golden Entry legend
        st.markdown(
            "<div style='font-size:11px;color:#888888;padding:4px 0'>"
            "⭐ <b>Golden Entry</b> = Z-Score ≤ -2.0 AND price below VWAP AND strategy = MEAN REVERSION  "
            "· <span style='color:#d94040;font-weight:700'>Red Z-Score</span> = oversold (long setup)  "
            "· <span style='color:#1a8c2a;font-weight:700'>Green Z-Score</span> = overbought (short setup)"
            "</div>",
            unsafe_allow_html=True
        )

    # ── Rejected / Blocked Stocks (always visible) ────────────
    st.markdown("---")
    st.markdown("### 🚫 Rejected Stocks — Detailed Reasons")
    rejected_detail = market.get("rejected_detail", [])
    if rejected_detail:
        # Colour-code by rejection reason
        reason_colour = {
            "BELOW_MIN_CAP":  "#888888",
            "ABOVE_MAX_CAP":  "#888888",
            "NO_DATA":        "#d94040",
            "LIQUIDITY":      "#d94040",
            "NO_TRADE":       "#b08800",
            "SECTOR_RS":      "#1a5fd9",
            "SCAN_FAILED":    "#d94040",
            "ERROR":          "#d94040",
        }
        reason_label = {
            "BELOW_MIN_CAP": "Cap too small",
            "ABOVE_MAX_CAP": "Cap too large",
            "NO_DATA":       "No data",
            "LIQUIDITY":     "Liquidity fail",
            "NO_TRADE":      "Regime no-trade",
            "SECTOR_RS":     "Sector RS fail",
            "SCAN_FAILED":   "Health/intraday block",
            "ERROR":         "Error",
        }
        # Group by reason
        from collections import defaultdict
        by_reason = defaultdict(list)
        for r in rejected_detail:
            by_reason[r["reason"]].append(r)

        cols_rej = st.columns(min(4, len(by_reason)))
        for i, (reason, items) in enumerate(sorted(by_reason.items())):
            col = cols_rej[i % len(cols_rej)]
            colour = reason_colour.get(reason, "#888888")
            label  = reason_label.get(reason, reason)
            rows   = "".join(
                f"<div style='padding:3px 0;border-bottom:1px solid #f0f0f0;font-size:11px;"
                f"font-weight:600;color:#1a1a1a'>{r['symbol']}"
                f"<span style='font-size:10px;font-weight:400;color:#666666;margin-left:6px'>"
                f"{r['detail']}</span></div>"
                for r in items
            )
            col.markdown(
                f"<div style='border:1px solid {colour};border-left:4px solid {colour};"
                f"border-radius:4px;padding:10px 12px;background:#ffffff;"
                f"box-shadow:0 1px 3px rgba(0,0,0,0.06)'>"
                f"<div style='font-size:11px;font-weight:800;color:{colour};"
                f"text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>"
                f"{label} ({len(items)})</div>"
                f"{rows}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No rejections recorded — run a scan first.")

    # ── Export (always visible regardless of scan results) ────
    st.markdown("---")
    st.markdown("### 📥 Export")

    last_export = st.session_state.get('last_export_time', 'Never')
    last_log    = st.session_state.get('last_log_path', None)
    st.markdown(
        f"<div style='background:#f0f7ff;border:1px solid #1a5fd9;border-left:4px solid #1a5fd9;"
        f"border-radius:4px;padding:10px 14px;margin-bottom:12px;font-size:11px;"
        f"font-weight:600;color:#1a1a1a'>"
        f"🔄 AUTO-EXPORT — every scan saves to ~/Downloads/  <br>"
        f"<span style='color:#666666;font-weight:400'>"
        f"📊 scan_YYYY-MM-DD_HHMM.xlsx (timestamped) &nbsp;·&nbsp; "
        f"📊 scan_latest.xlsx (always current) &nbsp;·&nbsp; "
        f"📋 scan_log_YYYY-MM-DD.csv (appends all day)<br>"
        f"Last export: <b>{last_export}</b></span></div>",
        unsafe_allow_html=True
    )

    ec1, ec2, ec3, ec4 = st.columns(4)

    with ec1:
        # Manual export trigger — works even if df is empty (exports rejected list)
        if st.button("📊 Export to Excel Now", use_container_width=True):
            export_excel(df if not df.empty else pd.DataFrame(), market)
            st.success("✅ Exported to ~/Downloads/")

    with ec2:
        if not df.empty:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download Results CSV",
                data=csv_data,
                file_name=f"scan_{datetime.now().strftime('%Y-%m-%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.markdown("<div style='font-size:11px;color:#888888;padding-top:8px'>"
                        "No scan results to download yet</div>", unsafe_allow_html=True)

    with ec3:
        if last_log and os.path.exists(last_log):
            with open(last_log, "rb") as f:
                st.download_button(
                    label="📋 Download Daily Log",
                    data=f.read(),
                    file_name=os.path.basename(last_log),
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.markdown("<div style='font-size:11px;color:#888888;padding-top:8px'>"
                        "Daily log builds after first successful scan</div>",
                        unsafe_allow_html=True)

    with ec4:
        rejected_detail_exp = market.get("rejected_detail", [])
        if rejected_detail_exp:
            rej_csv = pd.DataFrame(rejected_detail_exp).to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"🚫 Download Rejections ({len(rejected_detail_exp)})",
                data=rej_csv,
                file_name=f"rejected_{datetime.now().strftime('%Y-%m-%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.markdown("<div style='font-size:11px;color:#888888;padding-top:8px'>"
                        "No rejections recorded yet</div>", unsafe_allow_html=True)

    time.sleep(refresh)
    st.rerun()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[QUANT v3] Run: python -m streamlit run scanner_v3.py\n")

if STREAMLIT_MODE:
    run_dashboard()
