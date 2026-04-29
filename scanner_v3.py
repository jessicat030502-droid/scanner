"""
QUANT SCANNER v3 — Mid/Small Cap Day Trading Engine
=====================================================
Run:   python -m streamlit run scanner_v3.py
Deps:  pip install yfinance pandas numpy scipy streamlit openpyxl schedule

10 Layers: Hurst · Hawkes · OFI · Sector RS · ADD Breadth ·
           Z-Score · Kurtosis/Skew · Bayesian Prob · Half-Life · ATR R:R

Universe Scanner auto-runs at 9:00 AM EST daily.
Cap filter: $300M–$20B. Timeframe: 15s / 1m / 5m / 15m toggle.
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
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False


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

# Thresholds — edit these to tune the engine
MIDCAP_MIN       = 300_000_000
MIDCAP_MAX       = 20_000_000_000
ACCOUNT_SIZE     = float(os.environ.get("ACCOUNT_SIZE", 50000))
SIGNAL_THRESHOLD = 65
ATR_STOP_MULT    = 1.5
ATR_TARGET_MULT  = 3.0
SPY_WEAK_THRESH  = -0.005
SECTOR_RS_MIN    = 1.02
HAWKES_DECAY     = 0.3
OFI_LONG_ENTRY   = 0.60
OFI_SHORT_ENTRY  = 0.40
OFI_LONG_EXIT    = 0.45
OFI_SHORT_EXIT   = 0.55
ZSCORE_MAX       = 2.5
ZSCORE_MIN       = -2.5
KURTOSIS_MIN     = 3.0
SKEW_LONG_MIN    = 0.1
SKEW_SHORT_MAX   = -0.1
BAYES_BASE       = 0.50
BAYES_ADD        = 0.08
BAYES_SECTOR     = 0.10
BAYES_HAWKES     = 0.14
HALFLIFE_BASE    = 600
HALFLIFE_MIN     = 120
HALFLIFE_MAX     = 1800
INTRADAY_SOFT    = -0.010
INTRADAY_HARD    = -0.020
INTRADAY_KILL    = -0.030
GAP_DOWN_THRESH  = -0.010
LOWER_LOW_BARS   = 6


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
    return "🔥 CLUSTERING" if s>=72 else ("⚡ BUILDING" if s>=58 else ("〰 IDLE" if s>=42 else "❄ FADING"))


def compute_ofi(df: pd.DataFrame, window: int = 12) -> tuple:
    h, l, c, v = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    br   = np.where(h - l > 0, (c - l) / (h - l), 0.5)
    ofi  = (pd.Series(br * v).rolling(window, min_periods=3).sum() /
            pd.Series(v).rolling(window, min_periods=3).sum().replace(0, np.nan)).fillna(0.5)
    cur  = float(ofi.iloc[-1])
    delt = float(ofi.iloc[-1] - ofi.iloc[-4]) if len(ofi) >= 4 else 0.0
    return round(cur, 4), round(delt, 4), round(float(np.clip(cur*100 + delt*50, 0, 100)), 1)

def ofi_signal(ofi: float, delta: float) -> str:
    if ofi >= 0.65 and delta >= 0: return "🟢 ACCUMULATING"
    if ofi >= 0.60 and delta <  0: return "🟡 TOPPING"
    if ofi <= 0.35:                return "🔴 DISTRIBUTING"
    if ofi <= 0.42:                return "🟠 SELLING"
    return "⚪ NEUTRAL"


def compute_atr(df: pd.DataFrame, n: int = 14) -> float:
    if df is None or len(df) < n: return 0.0
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    trs = [max(h[-i] - l[-i],
               abs(h[-i] - c[-i-1]) if i < len(c) else 0,
               abs(l[-i] - c[-i-1]) if i < len(c) else 0)
           for i in range(1, min(n+1, len(c)))]
    return round(float(np.mean(trs)), 4) if trs else 0.0


def calc_vwap(df: pd.DataFrame) -> float:
    try:
        today = pd.Timestamp.now(tz=df.index.tz).strftime("%Y-%m-%d")
        tb    = df[df.index.strftime("%Y-%m-%d") == today]
        if len(tb) < 2: tb = df.tail(20)
        tp = (tb["high"] + tb["low"] + tb["close"]) / 3
        return round(float((tp * tb["volume"]).sum() / tb["volume"].sum()), 4)
    except Exception:
        return float(df["close"].iloc[-1])


def intraday_health(df: pd.DataFrame, prior_close: float) -> tuple:
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
    tp      = (tb["high"] + tb["low"] + tb["close"]) / 3
    vwap    = float((tp * tb["volume"]).cumsum().iloc[-1] / tb["volume"].cumsum().iloc[-1]) \
              if tb["volume"].sum() > 0 else curr

    mult, flags = 1.0, []
    if gap_r    < GAP_DOWN_THRESH: mult *= 0.85; flags.append(f"GAP↓{gap_r*100:.1f}%")
    if intra_r <= INTRADAY_KILL:   mult  = 0.0;  flags.append(f"SELLOFF{intra_r*100:.1f}%")
    elif intra_r <= INTRADAY_HARD: mult *= 0.50;  flags.append(f"WEAK{intra_r*100:.1f}%")
    elif intra_r <= INTRADAY_SOFT: mult *= 0.75;  flags.append(f"SOFT{intra_r*100:.1f}%")
    else:                                          flags.append(f"OK{intra_r*100:+.1f}%")
    if ll >= LOWER_LOW_BARS - 2:   mult *= 0.70;  flags.append("LOWER-LOWS")
    if curr < vwap:                mult *= 0.80;  flags.append(f"<VWAP${vwap:.2f}")

    mult = float(np.clip(mult, 0.0, 1.0))
    return mult, " | ".join(flags) or "HEALTHY", {
        "intraday_ret":round(intra_r*100,2), "gap_ret":round(gap_r*100,2),
        "today_open":round(open_p,2), "vwap":round(vwap,2),
        "below_vwap":curr < vwap, "lower_lows":ll, "health_mult":round(mult,3)
    }


def get_market_regime() -> dict:
    def dev(sym):
        arr = fetch_closes(sym, 25)
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
    ec, sc = fetch_closes(etf, 25), fetch_closes("SPY", 25)
    if ec is None or sc is None or len(ec) < 20 or len(sc) < 20: return 1.0, 50.0, True
    ratio = (ec[-1] / np.mean(ec[-20:])) / (sc[-1] / np.mean(sc[-20:]))
    return round(ratio, 4), round(float(np.clip(50+(ratio-1.0)*500,0,100)),1), ratio >= SECTOR_RS_MIN


def kelly_size(price: float, atr: float, win_rate: float = 0.60, rr: float = 2.0) -> tuple:
    if atr <= 0 or price <= 0: return 0.0, 0.0, 0
    kf     = float(np.clip((win_rate - (1 - win_rate) / rr) * 0.5, 0.0, 0.10))
    drisk  = ACCOUNT_SIZE * kf
    shares = int(drisk / (atr * ATR_STOP_MULT)) if atr * ATR_STOP_MULT > 0 else 0
    return round(kf, 4), round(drisk, 2), min(shares, int(ACCOUNT_SIZE * 0.20 / price))


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

def scan_symbol(symbol: str, market: dict, timeframe: str = "5m") -> Optional[dict]:
    passes, mcap_b = is_midcap(symbol)
    if not passes: return None

    df_d = fetch_daily(symbol, 60)
    if df_d is None or len(df_d) < 30: return None

    df_i = fetch_intraday(symbol, timeframe) or df_d.copy()
    if len(df_i) < 15: df_i = df_d.copy()

    price       = float(df_d["close"].iloc[-1])
    prior_close = float(df_d["close"].iloc[-2]) if len(df_d) >= 2 else price
    closes      = df_d["close"].values

    H              = compute_hurst(closes)
    h_sc           = hurst_score(H)
    lam, hawk_sc   = compute_hawkes(df_i)
    ofi, od, o_sc  = compute_ofi(df_i)
    etf            = SECTOR_MAP.get(symbol, "SPY")
    sec_rs, sec_sc, sec_gate = get_sector_rs(etf)
    atr            = compute_atr(df_d)
    add_val, add_b = get_add_breadth()
    z              = compute_zscore(closes)
    direction_g    = "LONG" if ofi >= 0.5 else "SHORT"
    kurt, skew     = compute_kurtosis_skew(closes)
    ks_sc, ks_lbl  = kurtosis_skew_score(kurt, skew, direction_g)
    z_ok, _        = zscore_gate(z, direction_g)
    bayes_p, b_log = bayesian_win_prob(add_b, sec_gate, hawk_sc, z_ok, ks_sc)
    hlth, hlbl, hd = intraday_health(df_i, prior_close)

    raw = float(np.clip(
        0.12*h_sc + 0.20*hawk_sc + 0.18*o_sc + 0.10*sec_sc +
        0.12*add_score(add_b) + 0.10*zscore_score(z) + 0.08*ks_sc + 0.10*bayes_p*100,
        0, 100))
    comp  = round(float(np.clip(raw * market["mkt_mult"] *
                                (1.0 if sec_gate else 0.70) *
                                (1.0 if add_b else 0.80) * hlth, 0, 100)), 1)
    alert = comp >= SIGNAL_THRESHOLD and market["allows_long"] and hlth > 0.0 and z_ok
    if not z_ok: comp = min(comp, 55.0)

    strength, hl_rem, hl_alive = halflife_remaining(symbol, comp, price, atr)
    kf, drisk, shares = kelly_size(price, atr, win_rate=bayes_p) if alert else (0.0, 0.0, 0)
    stop_p   = round(price - atr * ATR_STOP_MULT,  2) if atr > 0 else 0.0
    target   = round(price + atr * ATR_TARGET_MULT, 2) if atr > 0 else 0.0
    rr       = (target - price) / atr if atr > 0 else 0.0

    return {
        "symbol":symbol,"price":round(price,2),"score":comp,"alert":alert,
        "hurst_H":round(H,3),"hurst_score":round(h_sc,1),"hurst_regime":hurst_regime(H),
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
        "atr":atr,"stop":stop_p,"target":target,
        "health_mult":hlth,"health_label":hlbl,
        "intraday_ret":hd.get("intraday_ret",0.0),"gap_ret":hd.get("gap_ret",0.0),
        "vwap":hd.get("vwap",0.0),"below_vwap":hd.get("below_vwap",False),
        "mcap_b":mcap_b,"scanned_at":datetime.now().strftime("%H:%M:%S"),
    }


def run_full_scan(symbols: list = WATCHLIST, timeframe: str = "5m") -> tuple:
    market   = get_market_regime()
    results, blocked = [], []
    for sym in symbols:
        try:
            ok, mb = is_midcap(sym)
            if not ok: blocked.append(f"{sym}(${mb:.1f}B)"); continue
            r = scan_symbol(sym, market, timeframe)
            if r: results.append(r)
        except Exception as e:
            print(f"[ERR] {sym}: {e}")
    market.update({"blocked":blocked,"blocked_count":len(blocked),"scanned":len(results)})
    if not results: return pd.DataFrame(), market
    return pd.DataFrame(results).sort_values(by=["score","bayes_prob"],
                                              ascending=False).reset_index(drop=True), market


# ── Universe Scanner ──────────────────────────────────────────────────────────

def universe_prefilter(symbols: list) -> list:
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


def run_universe_scan(top_n: int = 30, timeframe: str = "5m") -> pd.DataFrame:
    print(f"[UNIVERSE] Start {datetime.now().strftime('%H:%M:%S')}")
    survivors = universe_prefilter(UNIVERSE)
    if not survivors: return pd.DataFrame()
    market, results = get_market_regime(), []
    for i, sym in enumerate(survivors):
        try:
            r = scan_symbol(sym, market, timeframe)
            if r: results.append(r)
            if (i+1) % 10 == 0: print(f"[UNIVERSE] {i+1}/{len(survivors)}")
        except Exception as e:
            print(f"[UNIVERSE ERR] {sym}: {e}")
        time.sleep(0.3)
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
    try:
        import openpyxl
    except ImportError:
        print("[EXPORT] pip install openpyxl"); return

    fname = f"scan_{datetime.now().strftime('%Y-%m-%d_%H%M')}.xlsx"
    fpath = os.path.join(os.path.expanduser("~"), "Downloads", fname)
    top   = df[df["alert"] == True]
    mkt_row = {**{k: market.get(k,"—") for k in
                  ["regime","spy_price","spy_dev","qqq_price","qqq_dev","scanned","blocked_count"]},
               "Date":datetime.now().strftime("%Y-%m-%d"),
               "ADD Bullish":str(_add_cache.get("bullish","—")),
               "Signals":len(top)}

    with pd.ExcelWriter(fpath, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Full Scan", index=False)
        if not top.empty: top.to_excel(w, sheet_name="Top Picks", index=False)
        pd.DataFrame([mkt_row]).to_excel(w, sheet_name="Market Context", index=False)

    if STREAMLIT_MODE:
        with open(fpath, "rb") as f:
            st.download_button(f"⬇️ Download {fname}", f.read(), file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.session_state["last_export_time"] = datetime.now().strftime("%H:%M:%S")
        st.success(f"✅ Saved → Downloads/{fname}")
    else:
        print(f"[EXPORT] {fpath}")


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
html,body,[class*="css"]{background:#0a0a0a!important;color:#e8e0c8!important;}
section[data-testid="stSidebar"]{background:#111008!important;border-right:2px solid #f5c400!important;}
.stButton>button{background:#f5c400!important;color:#0a0a0a!important;font-weight:700!important;
  border:none!important;border-radius:4px!important;letter-spacing:1px!important;font-size:13px!important;}
.stButton>button:hover{background:#ffd700!important;}
.stSelectbox label,.stSlider label,.stTextArea label,.stNumberInput label{color:#a09060!important;font-size:12px!important;}
[data-testid="stMetricValue"]{color:#f5c400!important;font-weight:700!important;}
[data-testid="stMetricDelta"]{color:#4a90d9!important;}
hr{border-color:#1e1a08!important;}
.hdr{font-size:28px;font-weight:800;letter-spacing:6px;color:#f5c400;text-transform:uppercase;
     border-bottom:2px solid #f5c400;padding-bottom:8px;text-shadow:0 0 30px rgba(245,196,0,0.3);}
.sub{font-size:11px;font-weight:500;color:#5a5030;letter-spacing:3px;text-transform:uppercase;}
.sub2{font-size:10px;font-weight:600;color:#5a5030;text-transform:uppercase;letter-spacing:2px;}
.card{border-radius:4px;padding:16px;margin-bottom:10px;border:1px solid #1e1a08;background:#111008;}
.card.on{border-color:#f5c400;background:#14120a;box-shadow:0 0 16px rgba(245,196,0,0.12);}
.card.warn{border-color:#4a90d9;background:#0a0e14;}
.sym{font-size:22px;font-weight:800;color:#ffffff;letter-spacing:2px;text-transform:uppercase;}
.sc{font-size:30px;font-weight:900;line-height:1;}
.bayes{font-size:20px;font-weight:700;line-height:1;}
.tag{display:inline-block;font-size:10px;font-weight:700;padding:3px 8px;
     border-radius:3px;margin:2px;letter-spacing:0.5px;text-transform:uppercase;}
.row{display:flex;gap:6px;margin-top:8px;}
.cell{flex:1;background:#0d0b06;border:1px solid #1e1a08;border-radius:3px;padding:5px 8px;}
.lbl{font-size:9px;font-weight:700;color:#5a5030;text-transform:uppercase;letter-spacing:1.5px;}
.val{font-size:12px;font-weight:600;color:#c8b870;}
.bar-bg{background:#1e1a08;border-radius:2px;height:4px;margin:6px 0;overflow:hidden;}
.bar-fg{height:4px;border-radius:2px;}
.trade{background:#0d0b06;border:1px solid #2a2208;border-left:3px solid #f5c400;
       border-radius:3px;padding:8px 12px;margin-top:8px;font-size:11px;
       font-weight:500;color:#a09060;}
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
        auto_list   = open("auto_watchlist.txt").read() if os.path.exists("auto_watchlist.txt") \
                      else "\n".join(WATCHLIST)
        custom_syms = st.text_area("Watchlist (one per line)", value=auto_list, height=250)
        run_btn     = st.button("▶ RUN SCAN NOW",       use_container_width=True)
        uni_btn     = st.button("🌍 RUN UNIVERSE SCAN", use_container_width=True)
        refresh     = st.slider("Auto-Refresh (sec)", 10, 300, 60)
        _           = st.number_input("Account Size ($)", value=int(ACCOUNT_SIZE), step=5000, format="%d")
        st.markdown("---")
        st.markdown(f"<div style='font-size:10px;font-weight:600;color:#4a3a18'>"
                    f"<span style='color:#f5c400'>10-LAYER ENGINE</span><br>"
                    f"✓ Hurst · Hawkes · OFI<br>"
                    f"✓ Sector RS · ADD Breadth<br>"
                    f"✓ Z-Score · Kurt/Skew<br>"
                    f"✓ Bayesian · Half-Life · ATR<br><br>"
                    f"<span style='color:#4a90d9'>CAP: $300M–$20B</span><br>"
                    f"<span style='color:#4a90d9'>TF: {timeframe}</span></div>",
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

    df, market, count = st.session_state["results"], st.session_state["market"], st.session_state["count"]

    # ── Header ────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1])
    add_val, add_bull = get_add_breadth()
    with c1:
        st.markdown('<div class="hdr">▲ QUANT SCANNER v3</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub">10-LAYER ENGINE · {timeframe.upper()} · CAP $300M–$20B</div>', unsafe_allow_html=True)
    with c2:
        ac = "#f5c400" if add_bull else "#e05050"
        st.markdown(f'<div style="text-align:right;margin-top:6px">'
                    f'<div class="sub">{datetime.now().strftime("%H:%M:%S")} EST · SCAN #{count}</div>'
                    f'<div style="font-size:12px;font-weight:700;color:{ac}">'
                    f'ADD {"▲ BULLISH" if add_bull else "▼ BEARISH"} ({add_val:.3f})</div>'
                    f'<div class="sub">NEXT REFRESH {max(0,int(refresh-(now-st.session_state["last_scan"])))}s</div>'
                    f'</div>', unsafe_allow_html=True)

    if market:
        rc = "#f5c400" if "RISK-ON"  in market.get("regime","") else \
             "#e05050" if "RISK-OFF" in market.get("regime","") else "#4a90d9"
        st.markdown(
            f'<div style="font-size:12px;font-weight:600;color:#7a6a30;'
            f'letter-spacing:2px;margin-bottom:14px;text-transform:uppercase;'
            f'border-bottom:1px solid #1e1a08;padding-bottom:8px">'
            f'MARKET: <span style="color:{rc}">{market.get("regime","—")}</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;SPY {market.get("spy_price","—")} '
            f'<span style="color:{"#f5c400" if market.get("spy_dev",0)>0 else "#e05050"}">'
            f'({market.get("spy_dev",0):+.2f}%)</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;QQQ {market.get("qqq_price","—")} '
            f'<span style="color:{"#f5c400" if market.get("qqq_dev",0)>0 else "#e05050"}">'
            f'({market.get("qqq_dev",0):+.2f}%)</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;✅ {market.get("scanned",0)} SCANNED'
            f'{"&nbsp;&nbsp;🚫 "+str(market["blocked_count"])+" BLOCKED" if market.get("blocked_count",0)>0 else ""}'
            f'</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No results — click ▶ RUN SCAN NOW")
        time.sleep(refresh); st.rerun(); return

    # Alert banner
    alerts = df[df["alert"] == True]
    if not alerts.empty:
        syms = "  ·  ".join(f"{r['symbol']} [{r['score']}] {r['bayes_prob']:.0f}%" for _,r in alerts.iterrows())
        st.markdown(
            f'<div style="background:rgba(245,196,0,0.06);border:2px solid #f5c400;'
            f'border-left:6px solid #f5c400;border-radius:4px;padding:12px 18px;'
            f'margin-bottom:14px;font-size:13px;font-weight:700;color:#f5c400;'
            f'letter-spacing:2px;text-transform:uppercase">▶ SIGNAL ALERT  ·  {syms}</div>',
            unsafe_allow_html=True)

    # ── Signal Cards ──────────────────────────────────────────
    cols = st.columns(min(4, max(1, len(df))))
    for i, (_, r) in enumerate(df.iterrows()):
        sc     = r["score"]
        sc_col = "#f5c400" if sc>=75 else "#4a90d9" if sc>=SIGNAL_THRESHOLD else "#7a6a30" if sc>=45 else "#e05050"
        bp     = r["bayes_prob"]
        bp_col = "#f5c400" if bp>=70 else "#4a90d9" if bp>=55 else "#e05050"
        css    = "on" if r["alert"] else ("warn" if sc>=55 else "")
        hl_str = f"⏱ {int(r['hl_remaining'])}s" if r["hl_alive"] else "—"
        trade  = (f'<div class="trade">ENTRY ${r["price"]} · STOP ${r["stop"]} · TARGET ${r["target"]}<br>'
                  f'Kelly {r["kelly_frac"]:.1%} → {r["shares"]} shares · ${r["dollar_risk"]} risk</div>'
                  if r["alert"] else "")
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="card {css}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <div class="sym">{r['symbol']}</div>
                  <div style="font-size:11px;font-weight:500;color:#6a5a28">${r['price']} &nbsp;·&nbsp; MCap ${r.get('mcap_b',0):.1f}B</div>
                </div>
                <div style="text-align:right">
                  <div class="sc" style="color:{sc_col}">{sc}</div>
                  <div class="bayes" style="color:{bp_col}">{bp:.0f}%</div>
                  <div class="sub2">BAYES WIN PROB</div>
                </div>
              </div>
              <div class="bar-bg"><div class="bar-fg" style="width:{sc}%;background:{sc_col}"></div></div>
              <div style="margin:6px 0">
                <span class="tag" style="background:#1a1600;color:#4a90d9;border:1px solid #2a2000">{r['hurst_regime']}</span>
                <span class="tag" style="background:#1a1600;color:#4a90d9;border:1px solid #2a2000">{r['hawkes_sig']}</span>
                <span class="tag" style="background:#1a1600;color:#4a90d9;border:1px solid #2a2000">{r['ofi_sig']}</span>
                <span class="tag" style="background:#1a1600;color:{'#f5c400' if r['add_bull'] else '#e05050'};border:1px solid #2a2000">ADD {'▲' if r['add_bull'] else '▼'}</span>
                <span class="tag" style="background:#1a1600;color:{'#f5c400' if r['zscore_ok'] else '#e05050'};border:1px solid #2a2000">Z={r['zscore']:.2f}</span>
                <span class="tag" style="background:#1a1600;color:#c8b870;border:1px solid #2a2000">{r['ks_label']}</span>
              </div>
              <div class="row">
                <div class="cell"><div class="lbl">HURST</div><div class="val">{r['hurst_H']}</div></div>
                <div class="cell"><div class="lbl">OFI</div><div class="val">{r['ofi']}</div></div>
                <div class="cell"><div class="lbl">KURT</div><div class="val">{r['kurtosis']}</div></div>
                <div class="cell"><div class="lbl">SKEW</div><div class="val">{r['skewness']}</div></div>
              </div>
              <div class="row">
                <div class="cell"><div class="lbl">SECTOR</div>
                  <div class="val" style="color:{'#f5c400' if r['sector_gate'] else '#e05050'}">{r['sector_etf']} {'✓' if r['sector_gate'] else '✗'}</div></div>
                <div class="cell"><div class="lbl">R:R</div>
                  <div class="val" style="color:{'#f5c400' if r['rr_ok'] else '#e05050'}">{'✓' if r['rr_ok'] else '✗'} {r['rr_ratio']:.1f}:1</div></div>
                <div class="cell"><div class="lbl">HALF-LIFE</div><div class="val">{hl_str}</div></div>
                <div class="cell"><div class="lbl">HEALTH</div><div class="val">{r['health_mult']:.2f}×</div></div>
              </div>
              {trade}
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
                    f"background:{'#1a1400' if ok else '#180808'};"
                    f"border:1px solid {'#f5c400' if ok else '#e05050'}'>"
                    f"{'✅' if ok else '❌'}&nbsp; {label}<br>"
                    f"<span style='font-size:10px;font-weight:500;color:#6a5a28'>{detail}</span></div>",
                    unsafe_allow_html=True)

        verdict_color = "#f5c400" if passed >= 6 else "#4a90d9" if passed >= 4 else "#e05050"
        verdict_text  = "▲ ALL CLEAR — EXECUTE"    if passed >= 6 else \
                        "— PARTIAL — REDUCE SIZE"   if passed >= 4 else "✕ BLOCKED — DO NOT ENTER"
        st.markdown(
            f"<div style='text-align:center;font-size:16px;font-weight:800;"
            f"color:{verdict_color};padding:14px;border-radius:4px;letter-spacing:3px;"
            f"text-transform:uppercase;border:2px solid {verdict_color};"
            f"background:{'rgba(245,196,0,0.06)' if passed>=6 else 'rgba(74,144,217,0.06)' if passed>=4 else 'rgba(224,80,80,0.06)'};"
            f"margin-top:14px'>{verdict_text} &nbsp;·&nbsp; {passed}/{len(checks)} CHECKS PASSED</div>",
            unsafe_allow_html=True)
    else:
        st.info("🎯 No execution-grade signals right now — watching and waiting.")

    # ── Full Results Table ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Full Scan Results")
    show = ["symbol","score","bayes_prob","zscore","kurtosis","skewness","ks_label",
            "hurst_regime","hawkes_sig","ofi_sig","sector_etf","add_bull",
            "rr_ok","kelly_frac","shares","stop","target","health_label","mcap_b"]
    st.dataframe(df[[c for c in show if c in df.columns]], use_container_width=True, hide_index=True)

    # ── Export ─────────────────────────────────────────────────
    st.markdown("---")
    ec1, ec2 = st.columns(2)
    with ec1:
        if st.button("📥 Export to Excel"): export_excel(df, market)
    with ec2:
        st.markdown(f"<div style='font-size:11px;font-weight:600;color:#6a5a28;"
                    f"padding-top:8px;text-transform:uppercase;letter-spacing:1px'>"
                    f"Last export: {st.session_state.get('last_export_time','Never')}</div>",
                    unsafe_allow_html=True)

    time.sleep(refresh)
    st.rerun()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[QUANT v3] Run: python -m streamlit run scanner_v3.py\n")

if STREAMLIT_MODE:
    run_dashboard()
