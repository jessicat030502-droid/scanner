"""
QUANT SCANNER v4 - Terminal Mode
=================================
Run:  python scanner_terminal_v4.py                    # single scan
      python scanner_terminal_v4.py --loop               # auto-refresh every 60s
      python scanner_terminal_v4.py --loop --interval 30 # refresh every 30s
      python scanner_terminal_v4.py --tf 1m              # 1-minute bars
      python scanner_terminal_v4.py --r2k                # Russell 2000 scan
      python scanner_terminal_v4.py --once               # scan once and exit

Deps: pip install yfinance pandas numpy scipy openpyxl schedule colorama

Imports all engine logic from scanner_v4.py - no duplication.
Outputs terminal table + auto-exports to Excel/CSV on every scan.
"""

import os, sys, time, argparse
from datetime import datetime

# -- Locate scanner_v4.py --
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

try:
    import scanner_v4 as engine
except ImportError as e:
    print(f"[ERROR] Cannot import scanner_v4.py: {e}")
    print("Make sure scanner_v4.py is in the same folder as this script.")
    sys.exit(1)

import pandas as pd
import numpy as np

# -- Optional colour support --
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    C_RESET  = Style.RESET_ALL
    C_YELLOW = Fore.YELLOW
    C_GREEN  = Fore.GREEN
    C_RED    = Fore.RED
    C_CYAN   = Fore.CYAN
    C_WHITE  = Fore.WHITE
    C_DIM    = Style.DIM
    C_BOLD   = Style.BRIGHT
    HAS_COLOR = True
except ImportError:
    C_RESET=C_YELLOW=C_GREEN=C_RED=C_CYAN=C_WHITE=C_DIM=C_BOLD=""
    HAS_COLOR = False


# -- Terminal helpers --

W = 90  # terminal width

def sep(char="-"): print(char * W)

def header_line():
    now    = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    src    = "SCHWAB LIVE" if engine.SCHWAB_AVAILABLE else "yfinance (15s delay)"
    umode  = getattr(engine, "UNIVERSE_MODE", "WATCHLIST")
    kill   = getattr(engine, "GLOBAL_KILL_SWITCH", False)
    lock   = getattr(engine, "GLOBAL_REGIME_LOCK", None)
    limit_o= getattr(engine, "LIMIT_ORDERS_ONLY", True)

    flags = []
    if kill:    flags.append(f"{C_RED}[KILL SWITCH]{C_RESET}")
    if lock:    flags.append(f"{C_YELLOW}[LOCKED: {lock}]{C_RESET}")
    if limit_o: flags.append(f"{C_GREEN}[LIMIT ORDERS]{C_RESET}")
    else:       flags.append(f"{C_RED}[MARKET ORDERS - WARNING]{C_RESET}")

    try:
        if engine.is_eod_soft_exit():        flags.append(f"{C_YELLOW}[EOD SOFT EXIT]{C_RESET}")
        allowed, _ = engine.is_trading_allowed()
        if not allowed:                       flags.append(f"{C_RED}[TRADING HALTED]{C_RESET}")
    except Exception:
        pass

    print("=" * W)
    print(f"  QUANT SCANNER v4   |   {now}   |   {src}   |   {umode}")
    if flags:
        print("  " + "  ".join(flags))
    print("=" * W)

def market_line(market: dict):
    regime = clean(market.get("regime","--"))
    rc  = C_GREEN if "RISK-ON" in regime else C_RED if "RISK-OFF" in regime else C_YELLOW
    spy = market.get("spy_price","--")
    spd = market.get("spy_dev", 0)
    qqq = market.get("qqq_price","--")
    qd  = market.get("qqq_dev", 0)
    print(f"  MARKET: {rc}{regime:<14}{C_RESET}"
          f"  |  SPY  {spy}  {pct_color(spd)}({spd:+.2f}%){C_RESET}"
          f"  |  QQQ  {qqq}  {pct_color(qd)}({qd:+.2f}%){C_RESET}")

def blocked_line(market: dict):
    rej = market.get("rejected_detail", [])
    if not rej:
        return
    from collections import Counter
    counts  = Counter(r["reason"] for r in rej)
    summary = "  ".join(f"{k}: {v}" for k,v in sorted(counts.items()))
    print(f"{C_DIM}  Filtered {len(rej)} symbols  |  {summary}{C_RESET}")
    cap_b = [r for r in rej if "CAP" in r["reason"]]
    if cap_b:
        names = "  ".join(f"{clean(r['symbol'])} ({clean(r['detail'])})"
                          for r in cap_b[:10])
        extra = f"  +{len(cap_b)-10} more" if len(cap_b) > 10 else ""
        print(f"{C_DIM}  Cap blocked:  {names}{extra}{C_RESET}")


def fmt_health(h: dict, health_label: str) -> str:
    """Format health string like: OK +2.1% | <VWAP$16.09"""
    return health_label[:28] if health_label else "--"

def clean(s: str) -> str:
    """Strip all non-ASCII characters."""
    return "".join(c for c in str(s) if ord(c) < 128)

def pad(s: str, width: int) -> str:
    return clean(s)[:width].ljust(width)

def yn(val) -> str:
    return "YES" if val else "NO "

def ok_fail(val) -> str:
    return "OK  " if val else "FAIL"

def pct_color(v: float) -> str:
    return C_GREEN if v > 0 else C_RED if v < 0 else C_RESET

def fmt_sig(raw: str, keywords: dict, width: int) -> str:
    cleaned = clean(raw).upper()
    for key, label in keywords.items():
        if key in cleaned:
            return label[:width].ljust(width)
    return cleaned[:width].ljust(width)

def fmt_hawkes(sig: str) -> str:
    return fmt_sig(sig, {
        "CLUSTERING":"CLUSTERING","BUILDING":"BUILDING","IDLE":"IDLE",
        "FADING":"FADING","SELL":"SELL PRESS"
    }, 14)

def fmt_ofi(sig: str) -> str:
    return fmt_sig(sig, {
        "ACCUMULATING":"ACCUMULATING","TOPPING":"TOPPING",
        "DISTRIBUTING":"DISTRIBUTING","SELLING":"SELLING","NEUTRAL":"NEUTRAL"
    }, 14)

def score_color(sc: float) -> str:
    if sc >= 75: return C_GREEN
    if sc >= 65: return C_CYAN
    if sc >= 45: return C_YELLOW
    return C_RED

def z_color(z: float) -> str:
    if z <= -2.0: return C_RED
    if z >= 2.0:  return C_GREEN
    return C_RESET


def print_results_table(df: pd.DataFrame, market: dict):
    """Print the main results table matching the screenshot format."""
    sep()
    # Column header
    print(f"  {'SYM':<7} {'SCORE':>6} {'STRAT':<5} "
          f"{'HEALTH':<26} {'HAWKES':<14} {'OFI':<14} "
          f"{'Z':>6} {'KELLY':>6} {'SHRS':>5}  STATUS")
    print("-" * W)

    if df.empty:
        print(f"  {C_YELLOW}No results - all symbols were filtered out."
              f"  Check the FILTER lines above.{C_RESET}")
        print("-" * W)
        return

    for _, r in df.iterrows():
        sym     = str(r["symbol"])
        score   = float(r.get("score", 0))
        raw_sc  = float(r.get("hurst_score", score))
        health  = fmt_health({}, str(r.get("health_label","--")))
        hawkes  = fmt_hawkes(str(r.get("hawkes_sig","--")))
        ofi     = fmt_ofi(str(r.get("ofi_sig","--")))
        kelly   = float(r.get("kelly_frac", 0)) * 100
        shares  = int(r.get("shares", 0))
        golden  = r.get("golden_entry", False)
        strat   = str(r.get("strategy","--"))
        vwap    = float(r.get("vwap", 0))
        below_v = bool(r.get("below_vwap", False))
        z       = float(r.get("zscore", 0))
        alert   = bool(r.get("alert", False))
        hl_alive  = bool(r.get("hl_alive", False))
        hl_rem    = float(r.get("hl_remaining", 0))
        i_blocked = bool(r.get("intraday_blocked", False))

        sc_c  = score_color(score)
        z_c   = z_color(z)
        al_mk = ""  # handled in status column
        ge_mk = ""  # handled in status column

        # Health with VWAP tag
        vwap_tag = f" <VWAP${vwap:.2f}" if below_v else ""
        health_str = health + vwap_tag

        # Half-life tag
        if i_blocked:
            hl_tag = f" {C_RED}[BLK]{C_RESET}"
        elif hl_alive:
            hl_tag = f" {C_GREEN}[{int(hl_rem)}s]{C_RESET}"
        else:
            hl_tag = f" {C_DIM}[exp]{C_RESET}"

        # Status column text - no emoji
        if i_blocked:
            status = "BLOCKED"
            sc_flag = C_RED
        elif not hl_alive:
            status = "EXPIRED"
            sc_flag = C_DIM
        elif alert and golden:
            status = "SIGNAL  GOLDEN"
            sc_flag = C_YELLOW
        elif alert:
            status = "SIGNAL"
            sc_flag = C_GREEN
        else:
            status = "WATCH"
            sc_flag = C_DIM

        strat_tag = "MR" if strat == "MEAN_REVERSION" else "TR"
        hs_pad    = pad(health_str, 26)
        hk_pad    = hawkes[:14].ljust(14)
        of_pad    = ofi[:14].ljust(14)

        print(f"  {C_BOLD}{sym:<7}{C_RESET}"
              f" {sc_c}{score:>6.1f}{C_RESET}"
              f" {strat_tag:<5}"
              f" {hs_pad}"
              f" {hk_pad}"
              f" {of_pad}"
              f" {z_c}{z:>+6.2f}{C_RESET}"
              f" {kelly:>6.1f}"
              f" {shares:>5}"
              f"  {sc_flag}{status}{C_RESET}")

    print("-" * W)
    alerts    = df[df["alert"] == True] if not df.empty else pd.DataFrame()
    goldens   = df[df["golden_entry"] == True] if "golden_entry" in df.columns else pd.DataFrame()
    strat_c   = df["strategy"].value_counts().to_dict() if "strategy" in df.columns else {}
    strat_str = "  ".join(f"{k}: {v}" for k,v in strat_c.items())
    gate      = "OPEN" if market.get("allows_long") else "CLOSED"
    gate_c    = C_GREEN if market.get("allows_long") else C_RED
    print(f"  Signals: {C_GREEN}{len(alerts)}{C_RESET}/{len(df)}"
          f"  |  Golden entries: {C_YELLOW}{len(goldens)}{C_RESET}"
          f"  |  Gate: {gate_c}{gate}{C_RESET}"
          f"  |  {strat_str}")


def print_top_picks(df: pd.DataFrame):
    """Print detailed breakdown of alert-level signals."""
    if df.empty: return
    alerts = df[df["alert"] == True].head(5)
    if alerts.empty:
        print(f"\n{C_DIM}  No signals above threshold this scan.{C_RESET}")
        return

    print()
    print("-" * W)
    print(f"  TOP PICKS  ({len(alerts)} signal(s))")
    print("-" * W)

    for idx, (_, r) in enumerate(alerts.iterrows(), 1):
        sym    = r["symbol"]
        price  = float(r.get("price", 0))
        mcap   = float(r.get("mcap_b", 0))
        score  = float(r.get("score", 0))
        strat  = r.get("strategy","--")
        golden = bool(r.get("golden_entry", False))

        golden_note = "   *** GOLDEN ENTRY  Z <= -2.0 + BELOW VWAP ***" if golden else ""
        print(f"\n  [{idx}]  {C_BOLD}{sym}{C_RESET}   "
              f"${price:.2f}   MCap ${mcap:.1f}B   "
              f"Score: {score_color(score)}{score:.1f}{C_RESET}"
              f"   Strategy: {clean(strat)}"
              f"{C_YELLOW}{golden_note}{C_RESET}")

        # Intraday
        intra_r = float(r.get("intraday_ret", 0))
        gap_r   = float(r.get("gap_ret", 0))
        vwap    = float(r.get("vwap", 0))
        below_v = bool(r.get("below_vwap", False))
        vwap_pos = "BELOW" if below_v else "ABOVE"
        print(f"  {'Intraday':<18} {pct_color(intra_r)}{intra_r:+.2f}%{C_RESET} from open"
              f"   Gap: {pct_color(gap_r)}{gap_r:+.2f}%{C_RESET}"
              f"   VWAP ${vwap:.2f}  ({vwap_pos})")

        # Health
        hlth  = float(r.get("health_mult", 1.0))
        hlbl  = str(r.get("health_label","--"))
        hc    = C_GREEN if hlth > 0.7 else C_YELLOW if hlth > 0.3 else C_RED
        print(f"  {'Health':<18} {hc}{hlbl}{C_RESET}  ({hlth:.2f}x)")

        # Indicators
        H      = float(r.get("hurst_H", 0))
        h_reg  = str(r.get("hurst_regime","--"))
        Hi     = float(r.get("hurst_H_intra", 0))
        hawk_l = float(r.get("hawkes_lam", 0))
        hawk_s = str(r.get("hawkes_sig","--"))
        ofi_v  = float(r.get("ofi", 0))
        ofi_s  = str(r.get("ofi_sig","--"))
        z_v    = float(r.get("zscore", 0))
        adx_v  = float(r.get("adx", 0))
        print(f"  {'Hurst (daily)':<18} H={H:.3f}  {clean(h_reg)}"
              f"   5m H={Hi:.3f}   ADX={adx_v:.1f}")
        zc = z_color(z_v)
        print(f"  {'Hawkes':<18} lam={hawk_l:.4f}   {fmt_hawkes(hawk_s)}")
        print(f"  {'OFI':<18} val={ofi_v:.4f}   {fmt_ofi(ofi_s)}")
        print(f"  {'Z-Score':<18} {zc}{z_v:+.3f}{C_RESET}"
              f"   (threshold +/- {engine.Z_ENTRY_THRESH:.1f})")

        # Sector + ADD
        sec_etf  = str(r.get("sector_etf","--"))
        sec_rs   = float(r.get("sector_rs", 1.0))
        sec_gate = bool(r.get("sector_gate", False))
        add_bull = bool(r.get("add_bull", True))
        add_val  = float(r.get("add_val", 0))
        sec_c    = C_GREEN if sec_gate else C_RED
        add_c    = C_GREEN if add_bull else C_RED
        print(f"  {'Sector RS':<18} {sec_c}{sec_etf}  RS={sec_rs:.4f}  ({ok_fail(sec_gate)}){C_RESET}")
        print(f"  {'ADD Breadth':<18} {add_c}{'BULLISH' if add_bull else 'BEARISH'}"
              f"  val={add_val:.3f}{C_RESET}")

        # Strategy-specific entry info
        if strat == "MEAN_REVERSION":
            mr_l = bool(r.get("mr_long", False))
            mr_s = bool(r.get("mr_short", False))
            exh  = str(r.get("exhaustion_reason","--"))
            vol_div = bool(r.get("vol_diverge", False))
            adapt_ofi = float(r.get("adaptive_ofi", 0.30))
            dir_str = "LONG" if mr_l else "SHORT" if mr_s else "--"
            dc = C_GREEN if mr_l else C_RED if mr_s else C_DIM
            print(f"  {'MR Direction':<18} {dc}{dir_str}{C_RESET}"
                  f"   OFI threshold: {adapt_ofi:.2f}"
                  f"   Vol divergence: {yn(vol_div)}")
            print(f"  {'Exhaustion':<18} {exh[:60]}")
        else:
            h_sc = float(r.get("hurst_score", 0))
            print(f"  TREND - Hurst score: {h_sc:.1f}  |  ADX={adx_v:.1f}")

        # Half-life decay
        hl_rem   = float(r.get("hl_remaining", 0))
        hl_alive = bool(r.get("hl_alive", False))
        hl_str   = float(r.get("hl_strength", 0))
        hl_c     = C_GREEN if hl_alive else C_RED
        print(f"  Signal half-life: {hl_c}{'ALIVE' if hl_alive else 'EXPIRED'}{C_RESET}"
              f"  {hl_rem:.0f}s remaining  |  Strength: {hl_str*100:.0f}%")

        # Intraday block reason (if blocked)
        i_blocked = bool(r.get("intraday_blocked", False))
        i_reason  = str(r.get("intraday_block_reason",""))
        if i_blocked and i_reason:
            print(f"  {C_RED}[BLOCKED]  {i_reason}{C_RESET}")

        # Regime reason + liquidity status
        reg_reason = str(r.get("regime_reason","--"))
        liq_status = str(r.get("liq_status","--"))
        print(f"  Regime: {reg_reason}  |  Liquidity: {liq_status}")

        # Range Filter confirmation
        rf_val      = float(r.get("rf_val", 0))
        rf_up       = bool(r.get("rf_up", False))
        rf_confirms = bool(r.get("rf_confirms", False))
        rf_warning  = bool(r.get("rf_warning", False))
        rf_len      = int(r.get("rf_length", 25))
        rf_dir   = "UPWARD" if rf_up else "DOWNWARD"
        rf_label = "CONFIRMED" if rf_confirms else "NOT CONFIRMED"
        rf_c     = C_GREEN if rf_confirms else C_YELLOW
        print(f"  {'Range Filter':<18} len={rf_len}   val=${rf_val:.4f}   "
              f"trend={rf_dir}   {rf_c}{rf_label}{C_RESET}")
        if rf_warning:
            print(f"  {C_YELLOW}  [WARN] Score above threshold but Range Filter"
                  f" not yet confirming -- wait.{C_RESET}")
        stop_p  = float(r.get("stop", 0))
        target  = float(r.get("target", vwap))
        rr      = float(r.get("rr_ratio", 0))
        kf      = float(r.get("kelly_frac", 0))
        sh      = int(r.get("shares", 0))
        drisk   = float(r.get("dollar_risk", 0))
        exit_t  = str(r.get("exit_type","--")).replace("_"," ")
        bayes   = float(r.get("bayes_prob", 0))
        print(f"  {'TRADE PLAN':-<50}")
        print(f"  {'Entry':<18} ${price:.2f}")
        print(f"  {'Stop':<18} ${stop_p:.2f}")
        print(f"  {'Target':<18} ${target:.2f}   ({exit_t})")
        print(f"  {'Risk/Reward':<18} {rr:.1f}:1")
        print(f"  {'Kelly':<18} {kf*100:.1f}%  ->  {sh} shares   (${drisk:.0f} at risk)")
        print(f"  {'Bayes Win Prob':<18} {C_CYAN}{bayes:.1f}%{C_RESET}")
        print()


def export_results(df: pd.DataFrame, market: dict):
    """
    CONSOLIDATED SINGLE EXCEL -- one file per scan containing all sheets.
    Only generates the scan Excel if stocks were actually found.
    Rejections CSV always exports (even if 0 stocks found) so you can debug.
    """
    try:
        import openpyxl
    except ImportError:
        print(f"  {C_YELLOW}[EXPORT] pip install openpyxl to enable Excel export{C_RESET}")
        _export_csv_only(df, market)
        return

    dl_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(dl_dir, exist_ok=True)
    now   = datetime.now()
    ts    = now.strftime("%Y-%m-%d_%H%M")
    today = now.strftime("%Y-%m-%d")
    rej   = market.get("rejected_detail", [])

    # -- Always export rejections CSV (even if 0 stocks found) --
    if len(rej) > 0:
        rej_df   = pd.DataFrame(rej)
        rej_df["scan_time"] = now.strftime("%H:%M:%S")
        rej_path = os.path.join(dl_dir, f"rejections_log_{today}.csv")
        rej_first = not os.path.exists(rej_path)
        rej_df.to_csv(rej_path, mode="a", header=rej_first, index=False)
        print(f"  {C_GREEN}[EXPORT] Rejections ->{C_RESET} "
              f"rejections_log_{today}.csv  ({len(rej)} blocked)")
    else:
        rej_df = pd.DataFrame()

    # -- Only write scan Excel if stocks were actually scanned --
    if df.empty:
        print(f"  {C_YELLOW}[SKIP] No stocks passed filters -- scan Excel not generated.{C_RESET}")
        print(f"  {C_DIM}  Rejections CSV still exported above for debugging.{C_RESET}")
        return

    # -- CONSOLIDATED Excel: single file with all sheets --
    top     = df[df["alert"] == True]
    mkt_row = {
        "Date":     today, "Time": now.strftime("%H:%M:%S"),
        "Regime":   market.get("regime","--"),
        "SPY":      market.get("spy_price","--"), "SPY%": market.get("spy_dev",0),
        "QQQ":      market.get("qqq_price","--"), "QQQ%": market.get("qqq_dev",0),
        "Scanned":  market.get("scanned",0),
        "Blocked":  market.get("blocked_count",0),
        "Signals":  len(top),
        "Strategy": engine.STRATEGY_MODE,
        "Universe": engine.UNIVERSE_MODE,
    }

    def _write_xl(path):
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Full Scan", index=False)
            if not top.empty:
                top.to_excel(w, sheet_name="Top Picks", index=False)
            pd.DataFrame([mkt_row]).to_excel(w, sheet_name="Market Context", index=False)
            if not rej_df.empty:
                rej_df.to_excel(w, sheet_name="Rejected Stocks", index=False)
            # Daily cumulative log as a sheet too
            log_path = os.path.join(dl_dir, f"scan_log_{today}.csv")
            if os.path.exists(log_path):
                pd.read_csv(log_path).to_excel(w, sheet_name="Day Log", index=False)

    fname_ts  = f"scan_{ts}.xlsx"
    fname_now = "scan_latest.xlsx"
    _write_xl(os.path.join(dl_dir, fname_ts))
    _write_xl(os.path.join(dl_dir, fname_now))
    print(f"\n  {C_GREEN}[EXPORT] Excel (all sheets) ->{C_RESET} {fname_ts}")
    print(f"  {C_DIM}  Sheets: Full Scan . Top Picks . Market Context . "
          f"Rejected Stocks . Day Log{C_RESET}")

    # -- Daily scan CSV log (appends all day) --
    log_path = os.path.join(dl_dir, f"scan_log_{today}.csv")
    df_log   = df.copy()
    df_log["scan_time"] = now.strftime("%H:%M:%S")
    first = not os.path.exists(log_path)
    df_log.to_csv(log_path, mode="a", header=first, index=False)
    print(f"  {C_GREEN}[EXPORT] Day log ->{C_RESET} scan_log_{today}.csv  (appended)")


def _export_csv_only(df: pd.DataFrame, market: dict):
    dl_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(dl_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    if not df.empty:
        p = os.path.join(dl_dir, f"scan_{ts}.csv")
        df.to_csv(p, index=False)
        print(f"  {C_GREEN}[EXPORT] CSV ->{C_RESET} {p}")
    rej = market.get("rejected_detail",[])
    if len(rej) > 0:  # Fixed: was "if rej:" which fails on DataFrame
        rp = os.path.join(dl_dir, f"rejected_{ts}.csv")
        pd.DataFrame(rej).to_csv(rp, index=False)
        print(f"  {C_GREEN}[EXPORT] Rejections CSV ->{C_RESET} {rp}")


def run_scan(timeframe: str = "5m"):
    """Run one full scan, print results, export files."""
    if not hasattr(run_scan, "_count"):
        run_scan._count = 1
    os.system("cls" if os.name == "nt" else "clear")
    sep("=")
    header_line()
    sep("=")

    print(f"\n  {C_DIM}Scanning {len(engine.WATCHLIST)} symbols on {timeframe} bars...{C_RESET}")
    df, market = engine.run_full_scan(engine.WATCHLIST, timeframe)

    # Market + filter summary
    print()
    market_line(market)
    blocked_line(market)
    print()

    # Results table
    print_results_table(df, market)

    # Top picks detail
    print_top_picks(df)

    # Export
    sep()
    export_results(df, market)

    # -- Single daily Excel log (all scans in one workbook) --
    try:
        import scanner_research_v4 as research
        scan_n = getattr(run_scan, "_count", 1)
        research.append_scan_to_daily_excel(df, market, scan_num=scan_n)
        run_scan._count = scan_n + 1
    except Exception as e:
        pass  # Research module optional

    sep("=")

    return df, market



# -- Help text shown with --help --

HELP_TEXT = """
COMMANDS:
  Single scan (runs once and exits):
    python scanner_terminal_v4.py
    python scanner_terminal_v4.py --once

  Continuous loop (re-scans every N seconds):
    python scanner_terminal_v4.py --loop
    python scanner_terminal_v4.py --loop --interval 30
    python scanner_terminal_v4.py --loop --interval 120

  Change timeframe:
    python scanner_terminal_v4.py --tf 1m    # 1-minute bars
    python scanner_terminal_v4.py --tf 5m    # 5-minute (default)
    python scanner_terminal_v4.py --tf 15m   # 15-minute

  Russell 2000 full scan (8-12 min, runs once):
    python scanner_terminal_v4.py --r2k
    python scanner_terminal_v4.py --r2k --tf 5m

  Override universe mode:
    python scanner_terminal_v4.py --universe WATCHLIST
    python scanner_terminal_v4.py --universe RUSSELL2000

  Combine options:
    python scanner_terminal_v4.py --loop --interval 60 --tf 5m
    python scanner_terminal_v4.py --loop --interval 300 --r2k

RESEARCH COMMANDS (separate script):
    python scanner_research_v4.py --profile ALKT CRUS BOOT --days 60
    python scanner_research_v4.py --backtest --days 30
    python scanner_research_v4.py --sensitivity --days 20
    python scanner_research_v4.py --consistency
    python scanner_research_v4.py --twap ALKT 100 --side LONG --price 16.27

OUTPUT FILES (in ~/Downloads/):
    scan_YYYY-MM-DD_HHMM.xlsx   -- timestamped Excel snapshot per scan
    scan_latest.xlsx            -- always the most recent scan
    scan_log_YYYY-MM-DD.csv     -- daily rolling log (appends every scan)
    rejections_log_YYYY-MM-DD.csv -- blocked stocks with reasons
    daily_YYYY-MM-DD.xlsx       -- single daily workbook (all scans in one)
"""

# -- Entry point --

def main():
    parser = argparse.ArgumentParser(
        description="Quant Scanner v4 -- Terminal Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_TEXT)
    parser.add_argument("--tf",       default="5m",        help="Timeframe: 15s/1m/5m/15m")
    parser.add_argument("--loop",     action="store_true", help="Auto-refresh loop")
    parser.add_argument("--once",     action="store_true", help="Single scan and exit")
    parser.add_argument("--interval", type=int, default=60,help="Refresh interval seconds")
    parser.add_argument("--r2k",      action="store_true", help="Run full Russell 2000 scan")
    parser.add_argument("--universe",  default=None,
                        help="WATCHLIST | RUSSELL2000 -- override UNIVERSE_MODE config")
    parser.add_argument("--kill",      action="store_true",
                        help="Set GLOBAL_KILL_SWITCH=True (halt all new entries)")
    parser.add_argument("--resume",    action="store_true",
                        help="Set GLOBAL_KILL_SWITCH=False (re-enable entries)")
    parser.add_argument("--regime",    default=None,
                        help="Lock strategy: TREND | MEAN_REVERSION | AUTO")
    parser.add_argument("--no-color",  action="store_true",
                        help="Disable colors (for log files or basic terminals)")
    args = parser.parse_args()

    # -- Apply CLI overrides --
    if args.no_color:
        global C_RESET,C_YELLOW,C_GREEN,C_RED,C_CYAN,C_WHITE,C_DIM,C_BOLD
        C_RESET=C_YELLOW=C_GREEN=C_RED=C_CYAN=C_WHITE=C_DIM=C_BOLD=""
        print("  Colors disabled")

    if args.universe:
        engine.UNIVERSE_MODE = args.universe.upper()
        print(f"  Universe: {engine.UNIVERSE_MODE}")

    if args.kill:
        engine.GLOBAL_KILL_SWITCH = True
        print(f"  [ON]  KILL SWITCH ENABLED -- no new entries will fire")

    if args.resume:
        engine.GLOBAL_KILL_SWITCH = False
        print(f"  [OFF] KILL SWITCH DISABLED -- entries re-enabled")

    if args.regime:
        reg = args.regime.upper()
        if reg in ("TREND","MEAN_REVERSION","AUTO"):
            engine.GLOBAL_REGIME_LOCK = None if reg == "AUTO" else reg
            engine.STRATEGY_MODE      = reg
            print(f"  Strategy locked to: {reg}")
        else:
            print(f"  Unknown regime '{reg}' -- use TREND, MEAN_REVERSION, or AUTO")

    if args.r2k or engine.UNIVERSE_MODE == "RUSSELL2000":
        print(f"\n  {C_YELLOW}{'='*60}")
        print(f"  RUSSELL 2000 MODE -- Full universe scan")
        print(f"  Stricter thresholds: $20M DV . 1.5x RelVol . Score>72")
        print(f"  Expected time: 8-12 min . Rate limit pauses between batches")
        print(f"  {'='*60}{C_RESET}\n")
        engine.UNIVERSE_MODE = "RUSSELL2000"
        df, market = engine.run_russell2000_scan(args.tf)
        print_results_table(df, market)
        print_top_picks(df)
        sep()
        export_results(df, market)
        sep("=")
        return

    if args.once or not args.loop:
        run_scan(args.tf)
        return

    # Continuous loop
    print(f"  Auto-scan every {args.interval}s on {args.tf} bars. "
          f"Universe: {engine.UNIVERSE_MODE}. Press Ctrl+C to stop.\n")
    scan_count = 0
    while True:
        try:
            # Reload watchlist from file every scan -- catches updates
            if os.path.exists("auto_watchlist.txt"):
                syms = [s.strip().upper()
                        for s in open("auto_watchlist.txt").read().split("\n")
                        if s.strip()]
                if syms:
                    engine.WATCHLIST = syms

            scan_count += 1
            print(f"\n  -- Scan #{scan_count} --  "
                  f"({len(engine.WATCHLIST)} symbols in watchlist)")
            run_scan(args.tf)
            print(f"\n  Next scan in {args.interval}s...  (Ctrl+C to stop)")
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n\n  Stopped after {scan_count} scan(s). Files saved to ~/Downloads/")
            break


if __name__ == "__main__":
    main()
