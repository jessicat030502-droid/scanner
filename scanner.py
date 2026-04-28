"""
============================================================
  HIGH-OCTANE QUANT SCANNER — PYTHON LAYER
  Indicators: Hurst Exponent | Hawkes Intensity | True OFI
  Data Source: TD Ameritrade / Schwab API (OHLCV + Quotes)
  Author: JT | Ontario, CA
  Usage: Run pre-market (07:30–09:30 EST) or intraday
============================================================

ARCHITECTURE OVERVIEW
─────────────────────
Layer 1 — Hurst Exponent  : Regime filter. Is today trending (H>0.6)
                             or choppy (H<0.4)? Gates the whole scanner.
Layer 2 — Hawkes Intensity : Are volume spikes self-exciting (real momentum
                             cluster) or random noise?
Layer 3 — True OFI Proxy  : Is buy pressure institutional (sustained) or
                             retail churn (random)?
Final    — Composite Score : Combines all three into a 0–100 score.
                             Score >= 65 = High-conviction signal.

REQUIREMENTS
─────────────
pip install schwab-py pandas numpy scipy requests

SCHWAB API SETUP
────────────────
1. Register at developer.schwab.com → create an app → get API key + secret
2. Set environment variables:
   SCHWAB_API_KEY=your_key
   SCHWAB_API_SECRET=your_secret
   SCHWAB_CALLBACK_URL=https://127.0.0.1
"""

import os
import time
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

# ── Try to import schwab-py; fall back to yfinance for testing ──────────────
try:
    import schwab
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False
    print("[INFO] schwab-py not installed. Using yfinance as fallback for testing.")
    import yfinance as yf


# ============================================================
#  CONFIG — edit these values
# ============================================================
WATCHLIST = [
    "NVDA", "AMD", "META", "TSLA", "MSTR",
    "PLTR", "SOFI", "RIVN", "COIN", "SMCI"
]

# Regime filter thresholds
HURST_TREND_MIN    = 0.55   # H > this → trending → run momentum scanner
HURST_REVERT_MAX   = 0.45   # H < this → mean-reverting → run reversion scanner

# Hawkes decay constant (λ): smaller = longer memory of spikes
HAWKES_DECAY       = 0.3    # tune between 0.1 (slow decay) – 0.5 (fast decay)
HAWKES_SPIKE_MULT  = 1.8    # volume must be this × baseline to count as event

# OFI thresholds
OFI_BULL_THRESHOLD = 0.60   # >60% buy pressure = bullish
OFI_BEAR_THRESHOLD = 0.40   # <40% buy pressure = bearish

# Composite score weights (must sum to 1.0)
W_HURST   = 0.30
W_HAWKES  = 0.35
W_OFI     = 0.35

# Final signal threshold
SIGNAL_THRESHOLD = 65       # 0–100 score; >= this = print the signal


# ============================================================
#  LAYER 0: DATA FETCHER
#  Fetches daily OHLCV bars. Schwab API used when available,
#  yfinance as fallback for testing/paper trading.
# ============================================================

def fetch_ohlcv(symbol: str, period_days: int = 120) -> Optional[pd.DataFrame]:
    """
    Returns DataFrame with columns: [open, high, low, close, volume]
    Index: DatetimeIndex
    """
    if SCHWAB_AVAILABLE:
        return _fetch_schwab(symbol, period_days)
    else:
        return _fetch_yfinance(symbol, period_days)


def _fetch_yfinance(symbol: str, period_days: int) -> Optional[pd.DataFrame]:
    """Fallback: yfinance. Good for testing. NOT for live trading."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{period_days}d", interval="1d")
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df
    except Exception as e:
        print(f"[ERROR] yfinance fetch failed for {symbol}: {e}")
        return None


def _fetch_schwab(symbol: str, period_days: int) -> Optional[pd.DataFrame]:
    """
    Live fetch via schwab-py.
    Docs: https://schwab-py.readthedocs.io/
    
    NOTE: First run requires OAuth browser login. After that, token is cached.
    """
    try:
        token_path = os.path.expanduser("~/.schwab_token.json")
        api_key    = os.environ["SCHWAB_API_KEY"]
        api_secret = os.environ["SCHWAB_API_SECRET"]
        callback   = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1")

        c = schwab.auth.client_from_token_file(token_path, api_key, api_secret)

        end   = datetime.now()
        start = end - timedelta(days=period_days)

        resp = c.get_price_history_every_day(
            symbol,
            start_datetime=start,
            end_datetime=end,
        )
        resp.raise_for_status()
        raw = resp.json()

        candles = raw.get("candles", [])
        if not candles:
            return None

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df = df.set_index("datetime")
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df

    except KeyError as e:
        print(f"[ERROR] Missing env var: {e}. Set SCHWAB_API_KEY / SCHWAB_API_SECRET.")
        return None
    except Exception as e:
        print(f"[ERROR] Schwab fetch failed for {symbol}: {e}")
        return None


# ============================================================
#  LAYER 1: HURST EXPONENT
#  Measures market "memory." Tells you if today is a trending
#  or mean-reverting regime BEFORE you place a trade.
#
#  H > 0.6  →  Trending    → Momentum scanner is valid
#  H = 0.5  →  Random walk → No edge, stay flat
#  H < 0.4  →  Mean-reverting → Reversion scanner is valid
#
#  Method: Rescaled Range (R/S) Analysis — the classic Hurst method.
# ============================================================

def hurst_exponent(close_prices: np.ndarray, min_lags: int = 10) -> float:
    """
    Calculate Hurst Exponent via R/S analysis.
    
    Parameters
    ----------
    close_prices : array of closing prices (at least 50 bars recommended)
    min_lags     : minimum number of lag windows to compute
    
    Returns
    -------
    float : Hurst exponent H in [0, 1]
    """
    prices = np.array(close_prices)
    if len(prices) < 20:
        return 0.5  # not enough data → assume random walk

    # Work in log returns space (normalizes volatility across stocks)
    log_returns = np.diff(np.log(prices))
    n = len(log_returns)

    # Build a range of lag sizes (geometrically spaced)
    lags = np.unique(
        np.floor(np.geomspace(10, n // 2, num=min_lags)).astype(int)
    )
    lags = lags[lags >= 4]  # need at least 4 points per sub-series

    rs_values = []
    valid_lags = []

    for lag in lags:
        # Split returns into non-overlapping windows of size `lag`
        n_windows = n // lag
        if n_windows < 2:
            continue

        rs_list = []
        for i in range(n_windows):
            segment = log_returns[i * lag : (i + 1) * lag]

            # Mean-adjust
            mean_adj = segment - segment.mean()

            # Cumulative sum (the "range" walk)
            cumsum = np.cumsum(mean_adj)

            # R/S = range divided by std dev
            R = cumsum.max() - cumsum.min()
            S = segment.std(ddof=1)

            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append(np.mean(rs_list))
            valid_lags.append(lag)

    if len(valid_lags) < 3:
        return 0.5  # not enough points to fit a line

    # H is the slope of log(R/S) vs log(lag)
    log_lags = np.log(valid_lags)
    log_rs   = np.log(rs_values)
    H, _     = np.polyfit(log_lags, log_rs, 1)

    # Clamp to [0, 1] — numerical edge cases
    return float(np.clip(H, 0.0, 1.0))


def hurst_score(H: float) -> float:
    """
    Convert Hurst exponent to a 0–100 score.
    100 = strong trend, 0 = strong mean-reversion, 50 = random.
    """
    return float(np.clip((H - 0.5) * 200 + 50, 0, 100))


def hurst_regime(H: float) -> str:
    if H > HURST_TREND_MIN:
        return "TRENDING"
    elif H < HURST_REVERT_MAX:
        return "REVERTING"
    else:
        return "CHOPPY"


# ============================================================
#  LAYER 2: HAWKES INTENSITY
#  Models volume spikes as a "self-exciting" point process.
#  Like earthquake aftershocks — one real spike causes more.
#
#  λ(t) = μ + Σ α·exp(−β·(t − tᵢ))  for tᵢ < t
#
#  If λ is rising → momentum is clustering → real move
#  If λ decays quickly → one-off spike → fade it
# ============================================================

def hawkes_intensity(
    volumes: np.ndarray,
    decay: float = HAWKES_DECAY,
    spike_multiplier: float = HAWKES_SPIKE_MULT
) -> tuple[np.ndarray, float]:
    """
    Compute Hawkes process intensity λ(t) for a volume series.
    
    Parameters
    ----------
    volumes          : array of daily volume
    decay            : β parameter — how fast the excitation dies (0.1–0.5)
    spike_multiplier : threshold to classify a bar as a "spike event"
    
    Returns
    -------
    intensities : λ(t) array for each bar
    current_lambda : λ at the most recent bar (the signal value)
    """
    n = len(volumes)
    if n < 5:
        return np.zeros(n), 0.0

    # Baseline: rolling 20-day average volume
    baseline = pd.Series(volumes).rolling(20, min_periods=5).mean().values

    # Background rate μ = normalized mean
    mu = np.nanmean(baseline[~np.isnan(baseline)])
    if mu == 0:
        return np.zeros(n), 0.0

    # α (excitation amplitude): how much each spike adds to intensity
    # Set so that a 2× spike roughly doubles the intensity temporarily
    alpha = mu * 0.5

    intensities = np.zeros(n)
    intensities[0] = mu

    for t in range(1, n):
        # Decay existing intensity
        decayed = intensities[t - 1] * math.exp(-decay)

        # Add excitation if previous bar was a spike
        prev_baseline = baseline[t - 1] if not np.isnan(baseline[t - 1]) else mu
        is_spike = volumes[t - 1] > (prev_baseline * spike_multiplier)

        excitation = alpha if is_spike else 0.0

        intensities[t] = mu + decayed - mu * math.exp(-decay) + excitation

    current_lambda = intensities[-1]
    return intensities, current_lambda


def hawkes_score(intensities: np.ndarray, current_lambda: float) -> float:
    """
    Score 0–100. Measures how elevated current intensity is vs. baseline.
    100 = maximum clustering (strongest momentum signal)
    """
    if len(intensities) < 10:
        return 50.0

    baseline_lambda = np.nanmean(intensities[:-5])  # average of all but last 5
    if baseline_lambda == 0:
        return 50.0

    ratio = current_lambda / baseline_lambda
    # ratio of 1.0 = neutral (score 50), ratio of 2.0 = very elevated (score ~90)
    score = 50 + 50 * math.tanh(ratio - 1.0)
    return float(np.clip(score, 0, 100))


def hawkes_signal(score: float) -> str:
    if score >= 70:
        return "CLUSTERING"   # strong momentum — join it
    elif score >= 50:
        return "BUILDING"     # momentum forming — watch
    else:
        return "FADING"       # spike was noise — ignore


# ============================================================
#  LAYER 3: TRUE OFI PROXY (Order Flow Imbalance)
#  Schwab doesn't expose tick-level bid/ask, so we use a
#  proven OHLCV proxy: the "Bulk Volume Classification" method.
#
#  Logic: Each candle's volume is split into buy/sell pressure
#  based on where price closed relative to the candle's range.
#
#  Close near HIGH  → most volume was buying  (bulls winning)
#  Close near LOW   → most volume was selling (bears winning)
#  Close in middle  → contested / institutional absorption
#
#  The OFI is the rolling buy ratio. Sustained above 0.60
#  = hidden accumulation. Spike above 0.75 = aggressive buy.
# ============================================================

def true_ofi_proxy(
    df: pd.DataFrame,
    window: int = 10
) -> tuple[pd.Series, float, float]:
    """
    Compute Order Flow Imbalance proxy from OHLCV data.
    
    Parameters
    ----------
    df     : DataFrame with [open, high, low, close, volume]
    window : rolling window to smooth OFI
    
    Returns
    -------
    ofi_series    : rolling OFI (0 to 1) for each bar
    current_ofi   : OFI at most recent bar
    ofi_delta     : change in OFI over last 3 bars (momentum of OFI)
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    vol   = df["volume"].values

    bar_range = high - low

    # Buy volume proportion per bar (0 = all selling, 1 = all buying)
    buy_ratio = np.where(
        bar_range > 0,
        (close - low) / bar_range,
        0.5  # doji — split equally
    )

    buy_vol  = buy_ratio * vol
    sell_vol = (1 - buy_ratio) * vol

    # Rolling OFI: sum(buy_vol) / sum(total_vol) over window
    buy_roll  = pd.Series(buy_vol).rolling(window, min_periods=3).sum()
    tot_roll  = pd.Series(vol).rolling(window, min_periods=3).sum()

    ofi_series = (buy_roll / tot_roll.replace(0, np.nan)).fillna(0.5)

    current_ofi = float(ofi_series.iloc[-1])

    # OFI delta: is buying pressure accelerating or decelerating?
    if len(ofi_series) >= 4:
        ofi_delta = float(ofi_series.iloc[-1] - ofi_series.iloc[-4])
    else:
        ofi_delta = 0.0

    return ofi_series, current_ofi, ofi_delta


def ofi_score(ofi: float, ofi_delta: float) -> float:
    """
    Score 0–100.
    >60 = bullish accumulation
    <40 = bearish distribution
    50  = neutral / contested
    """
    # Base score from absolute OFI level
    base = ofi * 100  # 0–100

    # Bonus/penalty for direction of change
    delta_bonus = ofi_delta * 100  # adds up to ~±10 pts

    score = base + delta_bonus * 0.5
    return float(np.clip(score, 0, 100))


def ofi_signal(ofi: float, ofi_delta: float) -> str:
    if ofi >= OFI_BULL_THRESHOLD and ofi_delta >= 0:
        return "ACCUMULATING"   # institutions quietly buying
    elif ofi >= OFI_BULL_THRESHOLD and ofi_delta < 0:
        return "TOPPING"        # was bullish, now fading
    elif ofi <= OFI_BEAR_THRESHOLD:
        return "DISTRIBUTING"   # selling pressure dominant
    else:
        return "NEUTRAL"


# ============================================================
#  COMPOSITE SCORER
#  Combines all three layers into a single conviction score.
#  Weighted average: Hurst 30%, Hawkes 35%, OFI 35%
# ============================================================

def composite_score(h_score: float, hawk_score: float, o_score: float) -> float:
    score = (
        W_HURST  * h_score  +
        W_HAWKES * hawk_score +
        W_OFI    * o_score
    )
    return round(float(np.clip(score, 0, 100)), 1)


def signal_label(score: float, regime: str) -> str:
    if score >= 75:
        return f"🟢 HIGH CONVICTION ({regime})"
    elif score >= SIGNAL_THRESHOLD:
        return f"🟡 MODERATE SIGNAL ({regime})"
    else:
        return f"🔴 NO EDGE ({regime})"


# ============================================================
#  MAIN SCANNER
#  Loops through watchlist, computes all three layers,
#  prints a ranked signal table.
# ============================================================

def run_scanner(symbols: list[str] = WATCHLIST, verbose: bool = True) -> pd.DataFrame:
    """
    Run the full three-layer scanner on a list of symbols.
    
    Returns a DataFrame sorted by composite score descending.
    Symbols above SIGNAL_THRESHOLD are printed as actionable signals.
    """
    results = []

    print(f"\n{'='*62}")
    print(f"  QUANT SCANNER  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data: {'Schwab API' if SCHWAB_AVAILABLE else 'yfinance (TEST MODE)'}")
    print(f"{'='*62}\n")

    for symbol in symbols:
        if verbose:
            print(f"  Analyzing {symbol}...", end="\r")

        df = fetch_ohlcv(symbol, period_days=120)

        if df is None or len(df) < 30:
            print(f"  [{symbol}] ⚠ Insufficient data — skipped")
            continue

        close   = df["close"].values
        volumes = df["volume"].values

        # ── LAYER 1: HURST ──────────────────────────────────────
        H       = hurst_exponent(close)
        h_sc    = hurst_score(H)
        regime  = hurst_regime(H)

        # ── LAYER 2: HAWKES ─────────────────────────────────────
        intensities, cur_lambda = hawkes_intensity(volumes)
        hawk_sc = hawkes_score(intensities, cur_lambda)
        hawk_sig = hawkes_signal(hawk_sc)

        # ── LAYER 3: OFI ────────────────────────────────────────
        _, cur_ofi, ofi_delta = true_ofi_proxy(df)
        o_sc    = ofi_score(cur_ofi, ofi_delta)
        o_sig   = ofi_signal(cur_ofi, ofi_delta)

        # ── COMPOSITE ───────────────────────────────────────────
        c_score = composite_score(h_sc, hawk_sc, o_sc)
        label   = signal_label(c_score, regime)

        results.append({
            "Symbol"      : symbol,
            "Score"       : c_score,
            "Signal"      : label,
            "Hurst_H"     : round(H, 3),
            "Hurst_Score" : round(h_sc, 1),
            "Regime"      : regime,
            "Hawkes_λ"    : round(cur_lambda, 0),
            "Hawkes_Score": round(hawk_sc, 1),
            "Hawkes_Sig"  : hawk_sig,
            "OFI"         : round(cur_ofi, 3),
            "OFI_Delta"   : round(ofi_delta, 3),
            "OFI_Score"   : round(o_sc, 1),
            "OFI_Sig"     : o_sig,
        })

        # Rate limit guard (Schwab allows ~120 req/min on free tier)
        time.sleep(0.3)

    if not results:
        print("  No results returned. Check your symbols or data connection.")
        return pd.DataFrame()

    df_out = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)

    # ── PRINT SIGNAL TABLE ──────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  {'SYM':<6} {'SCORE':>5}  {'REGIME':<11} {'HAWKES':<12} {'OFI':<14} SIGNAL")
    print(f"{'─'*62}")

    for _, row in df_out.iterrows():
        flag = "◄◄" if row["Score"] >= SIGNAL_THRESHOLD else "  "
        print(
            f"  {row['Symbol']:<6} {row['Score']:>5}  "
            f"{row['Regime']:<11} {row['Hawkes_Sig']:<12} "
            f"{row['OFI_Sig']:<14} {flag}"
        )

    high_conv = df_out[df_out["Score"] >= SIGNAL_THRESHOLD]
    print(f"\n{'─'*62}")
    print(f"  Signals above threshold ({SIGNAL_THRESHOLD}): {len(high_conv)}/{len(df_out)}")

    if len(high_conv) > 0:
        print(f"\n  ── TOP PICKS ──────────────────────────────────────────")
        for _, row in high_conv.iterrows():
            print(f"\n  {row['Symbol']}  [{row['Score']}]  {row['Signal']}")
            print(f"    Hurst  : H={row['Hurst_H']}  → {row['Regime']}")
            print(f"    Hawkes : λ={row['Hawkes_λ']}  score={row['Hawkes_Score']}  → {row['Hawkes_Sig']}")
            print(f"    OFI    : {row['OFI']}  Δ={row['OFI_Delta']}  → {row['OFI_Sig']}")

    print(f"\n{'='*62}\n")

    return df_out


# ============================================================
#  INTEGRATION GUIDE
#  How to wire this into your existing TOS workflow
# ============================================================

INTEGRATION_NOTES = """
HOW TO USE WITH YOUR TOS SCANNER
─────────────────────────────────
1. TOS fires at 07:45 PST → gives you your Cyan/Yellow watchlist.

2. BEFORE entering any trade, run:
      results = run_scanner(["NVDA", "AMD", "TSLA"])  ← paste TOS hits

3. GATE RULES:
   ┌─────────────────────────────────────────────────────┐
   │  TOS Signal  │  Python Score  │  Action             │
   ├─────────────────────────────────────────────────────┤
   │  CYAN (Mom)  │  Score >= 65   │  ENTER — full size  │
   │  CYAN (Mom)  │  Score 50–64   │  ENTER — half size  │
   │  CYAN (Mom)  │  Score < 50    │  SKIP — choppy day  │
   │  YELLOW (Rev)│  OFI < 0.40    │  SHORT — confirmed  │
   │  YELLOW (Rev)│  OFI 0.40–0.60 │  WAIT — watch L2    │
   └─────────────────────────────────────────────────────┘

4. HURST AS DAILY REGIME GATE:
   - Run hurst_exponent(SPY_prices) each morning.
   - If SPY H < 0.45 → CHOPPY DAY → reduce position sizing 50%
   - If SPY H > 0.55 → TRENDING DAY → run scanner at full confidence

5. HAWKES AS ENTRY TIMER:
   - Don't enter on the first spike.
   - Wait for hawkes_signal() == "CLUSTERING" (λ self-exciting).
   - This is the confirmation the move is institutional, not retail.

6. OFI AS EXIT SIGNAL:
   - While in trade, monitor ofi_delta.
   - If OFI was ACCUMULATING and delta turns negative → start trailing stop.
   - This is often 2–5 bars BEFORE the price actually rolls over.
"""


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Print integration guide on first run
    print(INTEGRATION_NOTES)

    # Run the scanner
    results_df = run_scanner(WATCHLIST, verbose=True)

    # Optional: export to CSV for logging
    if not results_df.empty:
        fname = f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        results_df.to_csv(fname, index=False)
        print(f"  Results saved → {fname}\n")
