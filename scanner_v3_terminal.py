"""
QUANT SCANNER v3 — Terminal Mode
=================================
Run:  python scanner_terminal.py
      python scanner_terminal.py --loop          # auto-refresh every 60s
      python scanner_terminal.py --tf 5m         # timeframe: 15s/1m/5m/15m
      python scanner_terminal.py --once          # single scan then exit

Deps: pip install yfinance pandas numpy scipy openpyxl schedule colorama

Imports all engine logic from scanner_v3.py — no duplication.
Outputs terminal table + auto-exports to Excel/CSV on every scan.
"""

import os, sys, time, argparse
from datetime import datetime

# ── Locate scanner_v3.py ──────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

try:
    import scanner_v3 as engine
except ImportError as e:
    print(f"[ERROR] Cannot import scanner_v3.py: {e}")
    print("Make sure scanner_v3.py is in the same folder as this script.")
    sys.exit(1)

import pandas as pd
import numpy as np

# ── Optional colour support ───────────────────────────────────
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


# ── Terminal helpers ──────────────────────────────────────────

W = 90  # terminal width

def sep(char="─"): print(char * W)

def header_line():
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = f"yfinance {'(15s delay)' if not engine.SCHWAB_AVAILABLE else ''}"
    src  = "SCHWAB LIVE" if engine.SCHWAB_AVAILABLE else mode
    print(f"{C_BOLD}{C_CYAN}  QUANT SCANNER v3{C_RESET}  │  {now}  │  {src}")

def market_line(market: dict):
    regime = market.get("regime","—")
    rc = C_GREEN if "RISK-ON" in regime else C_RED if "RISK-OFF" in regime else C_YELLOW
    spy  = market.get("spy_price","—")
    spyd = market.get("spy_dev",0)
    qqq  = market.get("qqq_price","—")
    qqqd = market.get("qqq_dev",0)
    sdc  = C_GREEN if spyd > 0 else C_RED
    qdc  = C_GREEN if qqqd > 0 else C_RED
    print(f"  MARKET: {rc}{regime}{C_RESET}"
          f"  │  SPY {spy} {sdc}({spyd:+.2f}%){C_RESET}"
          f"  │  QQQ {qqq} {qdc}({qqqd:+.2f}%){C_RESET}")

def blocked_line(market: dict):
    rej = market.get("rejected_detail", [])
    if not rej: return
    from collections import Counter
    counts = Counter(r["reason"] for r in rej)
    parts  = "  ".join(f"{k}×{v}" for k,v in counts.items())
    scanned = market.get("scanned", 0)
    print(f"{C_DIM}  [FILTER] Blocked {len(rej)} — {parts}{C_RESET}")
    # Show individual blocked symbols with cap where relevant
    cap_blocked = [r for r in rej if "CAP" in r["reason"]]
    if cap_blocked:
        names = ", ".join(f"{r['symbol']} ({r['detail']})" for r in cap_blocked[:12])
        if len(cap_blocked) > 12:
            names += f" + {len(cap_blocked)-12} more"
        print(f"{C_DIM}  [FILTER] Blocked {len(cap_blocked)} large/mega-cap(s): {names}{C_RESET}")


def fmt_health(h: dict, health_label: str) -> str:
    """Format health string like: OK +2.1% | <VWAP$16.09"""
    return health_label[:28] if health_label else "—"

def fmt_hawkes(sig: str) -> str:
    icons = {"🔥 CLUSTERING":"■ CLUSTERING","⚡ BUILDING":"■ BUILDING",
             "〰 IDLE":"□ IDLE","❄ FADING":"□ FADING","🔴 SELL PRESSURE":"▼ SELL"}
    return icons.get(sig, sig[:12])

def fmt_ofi(sig: str) -> str:
    icons = {"🟢 ACCUMULATING":"■ ACCUMULATING","🟡 TOPPING":"□ TOPPING",
             "🔴 DISTRIBUTING":"▼ DISTRIBUTING","🟠 SELLING":"▼ SELLING","⚪ NEUTRAL":"□ NEUTRAL"}
    return icons.get(sig, sig[:14])

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
    print(f"  {'SYM':<6} {'SCORE':<6} {'RAW':<6} {'HEALTH':<22} "
          f"{'HAWKES':<16} {'OFI':<16} {'KELLY%':<8} {'SHR':<6} {'★'}")
    sep("─")

    if df.empty:
        print(f"  {C_YELLOW}No results — all symbols filtered. See FILTER lines above.{C_RESET}")
        sep()
        return

    for _, r in df.iterrows():
        sym     = str(r["symbol"])
        score   = float(r.get("score", 0))
        raw_sc  = float(r.get("hurst_score", score))
        health  = fmt_health({}, str(r.get("health_label","—")))
        hawkes  = fmt_hawkes(str(r.get("hawkes_sig","—")))
        ofi     = fmt_ofi(str(r.get("ofi_sig","—")))
        kelly   = float(r.get("kelly_frac", 0)) * 100
        shares  = int(r.get("shares", 0))
        golden  = r.get("golden_entry", False)
        strat   = str(r.get("strategy","—"))
        vwap    = float(r.get("vwap", 0))
        below_v = bool(r.get("below_vwap", False))
        z       = float(r.get("zscore", 0))
        alert   = bool(r.get("alert", False))
        hl_alive  = bool(r.get("hl_alive", False))
        hl_rem    = float(r.get("hl_remaining", 0))
        i_blocked = bool(r.get("intraday_blocked", False))

        sc_c  = score_color(score)
        z_c   = z_color(z)
        ge_mk = f"  {C_YELLOW}★{C_RESET}" if golden else ""
        al_mk = f" {C_GREEN}◄◄{C_RESET}" if alert else ""

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

        print(f"  {C_BOLD}{sym:<6}{C_RESET} "
              f"{sc_c}{score:<6.1f}{C_RESET} "
              f"{raw_sc:<6.1f} "
              f"{health_str:<22} "
              f"{hawkes:<16} "
              f"{ofi:<16} "
              f"{kelly:<8.1f} "
              f"{shares:<6}"
              f"{hl_tag}{al_mk}{ge_mk}")

    sep("─")
    alerts = df[df["alert"] == True] if not df.empty else pd.DataFrame()
    strat_counts = df["strategy"].value_counts().to_dict() if "strategy" in df.columns else {}
    strat_str = "  ".join(f"{k}:{v}" for k,v in strat_counts.items())
    print(f"  Signals: {C_GREEN}{len(alerts)}{C_RESET}/{len(df)}  │  "
          f"Market gate: {C_GREEN if market.get('allows_long') else C_RED}"
          f"{'■ OPEN' if market.get('allows_long') else '■ CLOSED'}{C_RESET}"
          f"  │  {strat_str}")


def print_top_picks(df: pd.DataFrame):
    """Print detailed breakdown of alert-level signals."""
    if df.empty: return
    alerts = df[df["alert"] == True].head(5)
    if alerts.empty:
        print(f"\n{C_DIM}  — No signals above threshold this scan —{C_RESET}")
        return

    print(f"\n{C_BOLD}{'─'*W}")
    print(f"  TOP PICKS{C_RESET}")
    print("─" * W)

    for _, r in alerts.iterrows():
        sym    = r["symbol"]
        price  = float(r.get("price", 0))
        mcap   = float(r.get("mcap_b", 0))
        score  = float(r.get("score", 0))
        strat  = r.get("strategy","—")
        golden = bool(r.get("golden_entry", False))

        ge_tag = f"  {C_YELLOW}★ GOLDEN ENTRY{C_RESET}" if golden else ""
        print(f"\n  {C_BOLD}{C_CYAN}{sym}{C_RESET}  ${price:.2f}  MCap ${mcap:.1f}B  [{score:.1f}]{ge_tag}")

        # Intraday
        intra_r = float(r.get("intraday_ret", 0))
        gap_r   = float(r.get("gap_ret", 0))
        vwap    = float(r.get("vwap", 0))
        below_v = bool(r.get("below_vwap", False))
        vwap_pos = "BELOW" if below_v else "ABOVE"
        ic = C_RED if intra_r < 0 else C_GREEN
        gc = C_RED if gap_r   < 0 else C_GREEN
        print(f"  Intraday: {ic}{intra_r:+.2f}%{C_RESET} from open  │  "
              f"Gap: {gc}{gap_r:+.2f}%{C_RESET}  │  "
              f"VWAP ${vwap:.2f} {vwap_pos}")

        # Health
        hlth  = float(r.get("health_mult", 1.0))
        hlbl  = str(r.get("health_label","—"))
        hc    = C_GREEN if hlth > 0.7 else C_YELLOW if hlth > 0.3 else C_RED
        print(f"  Health: {hc}{hlbl}{C_RESET} (mult {hlth:.2f}×)")

        # Indicators
        H      = float(r.get("hurst_H", 0))
        h_reg  = str(r.get("hurst_regime","—"))
        Hi     = float(r.get("hurst_H_intra", 0))
        hawk_l = float(r.get("hawkes_lam", 0))
        hawk_s = str(r.get("hawkes_sig","—"))
        ofi_v  = float(r.get("ofi", 0))
        ofi_s  = str(r.get("ofi_sig","—"))
        z_v    = float(r.get("zscore", 0))
        adx_v  = float(r.get("adx", 0))
        print(f"  Hurst: D={H:.3f} ({h_reg})  5m={Hi:.3f}  │  ADX={adx_v:.1f}")
        zc = z_color(z_v)
        print(f"  Hawkes: λ={hawk_l:.4f}  {hawk_s}  │  "
              f"OFI: {ofi_v:.4f}  {ofi_s}  │  "
              f"Z: {zc}{z_v:+.3f}{C_RESET}")

        # Sector + ADD
        sec_etf  = str(r.get("sector_etf","—"))
        sec_rs   = float(r.get("sector_rs", 1.0))
        sec_gate = bool(r.get("sector_gate", False))
        add_bull = bool(r.get("add_bull", True))
        add_val  = float(r.get("add_val", 0))
        sec_c    = C_GREEN if sec_gate else C_RED
        add_c    = C_GREEN if add_bull else C_RED
        print(f"  Sector: {sec_c}{sec_etf} RS={sec_rs:.4f} {'✓' if sec_gate else '✗'}{C_RESET}"
              f"  │  ADD: {add_c}{'▲' if add_bull else '▼'} {add_val:.3f}{C_RESET}")

        # Strategy-specific entry info
        if strat == "MEAN_REVERSION":
            mr_l = bool(r.get("mr_long", False))
            mr_s = bool(r.get("mr_short", False))
            exh  = str(r.get("exhaustion_reason","—"))
            vol_div = bool(r.get("vol_diverge", False))
            adapt_ofi = float(r.get("adaptive_ofi", 0.30))
            dir_str = "LONG" if mr_l else "SHORT" if mr_s else "—"
            dc = C_GREEN if mr_l else C_RED if mr_s else C_DIM
            print(f"  MR Direction: {dc}{dir_str}{C_RESET}"
                  f"  │  Adaptive OFI threshold: {adapt_ofi:.2f}"
                  f"  │  Vol divergence: {'✓' if vol_div else '✗'}")
            print(f"  Exhaustion: {exh[:65]}")
        else:
            h_sc = float(r.get("hurst_score", 0))
            print(f"  TREND — Hurst score: {h_sc:.1f}  │  ADX={adx_v:.1f}")

        # Half-life decay
        hl_rem   = float(r.get("hl_remaining", 0))
        hl_alive = bool(r.get("hl_alive", False))
        hl_str   = float(r.get("hl_strength", 0))
        hl_c     = C_GREEN if hl_alive else C_RED
        print(f"  Signal half-life: {hl_c}{'ALIVE' if hl_alive else 'EXPIRED'}{C_RESET}"
              f"  {hl_rem:.0f}s remaining  │  Strength: {hl_str*100:.0f}%")

        # Intraday block reason (if blocked)
        i_blocked = bool(r.get("intraday_blocked", False))
        i_reason  = str(r.get("intraday_block_reason",""))
        if i_blocked and i_reason:
            print(f"  {C_RED}⚠ INTRADAY BLOCKED: {i_reason}{C_RESET}")

        # Regime reason + liquidity status
        reg_reason = str(r.get("regime_reason","—"))
        liq_status = str(r.get("liq_status","—"))
        print(f"  Regime: {reg_reason}  │  Liquidity: {liq_status}")

        # Trade plan
        stop_p  = float(r.get("stop", 0))
        target  = float(r.get("target", vwap))
        rr      = float(r.get("rr_ratio", 0))
        kf      = float(r.get("kelly_frac", 0))
        sh      = int(r.get("shares", 0))
        drisk   = float(r.get("dollar_risk", 0))
        exit_t  = str(r.get("exit_type","—")).replace("_"," ")
        bayes   = float(r.get("bayes_prob", 0))
        print(f"  {C_BOLD}Entry: ${price:.2f}  │  Stop: ${stop_p:.2f}  │  "
              f"Target: ${target:.2f}  ({exit_t})  │  R:R {rr:.1f}:1{C_RESET}")
        print(f"  {kelly*100:.1f}% Kelly → {sh} shares  "
              f"(${drisk:.0f} risk)  │  Bayes win: {C_CYAN}{bayes:.1f}%{C_RESET}")


def export_results(df: pd.DataFrame, market: dict):
    """Write Excel + CSV to ~/Downloads. Called after every scan."""
    try:
        import openpyxl
    except ImportError:
        print(f"  {C_YELLOW}[EXPORT] pip install openpyxl to enable Excel export{C_RESET}")
        _export_csv_only(df, market)
        return

    dl_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(dl_dir, exist_ok=True)
    now       = datetime.now()
    ts        = now.strftime("%Y-%m-%d_%H%M")
    today     = now.strftime("%Y-%m-%d")

    # ── Excel ─────────────────────────────────────────────────
    fname_ts  = f"scan_{ts}.xlsx"
    fname_now = "scan_latest.xlsx"

    top     = df[df["alert"] == True] if not df.empty else pd.DataFrame()
    rej     = pd.DataFrame(market.get("rejected_detail", []))
    mkt_row = {
        "Date":       today,
        "Time":       now.strftime("%H:%M:%S"),
        "Regime":     market.get("regime","—"),
        "SPY":        market.get("spy_price","—"),
        "SPY%":       market.get("spy_dev",0),
        "QQQ":        market.get("qqq_price","—"),
        "QQQ%":       market.get("qqq_dev",0),
        "Scanned":    market.get("scanned",0),
        "Blocked":    market.get("blocked_count",0),
        "Signals":    len(top),
        "Strategy":   engine.STRATEGY_MODE,
    }

    def _write_xl(path):
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            if not df.empty:
                df.to_excel(w, sheet_name="Full Scan", index=False)
            if not top.empty:
                top.to_excel(w, sheet_name="Top Picks (Signals)", index=False)
            pd.DataFrame([mkt_row]).to_excel(w, sheet_name="Market Context", index=False)
            if not rej.empty:
                rej.to_excel(w, sheet_name="Rejected Stocks", index=False)

    _write_xl(os.path.join(dl_dir, fname_ts))
    _write_xl(os.path.join(dl_dir, fname_now))
    print(f"\n  {C_GREEN}[EXPORT] Excel →{C_RESET} {fname_ts}  +  scan_latest.xlsx")

    # ── Daily scan log CSV (appends) ─────────────────────────
    if not df.empty:
        log_path = os.path.join(dl_dir, f"scan_log_{today}.csv")
        df_log   = df.copy()
        df_log["scan_time"] = now.strftime("%H:%M:%S")
        first    = not os.path.exists(log_path)
        df_log.to_csv(log_path, mode="a", header=first, index=False)
        print(f"  {C_GREEN}[EXPORT] Daily log →{C_RESET} scan_log_{today}.csv  (appended)")

    # ── Rejection log CSV (appends) ──────────────────────────
    if not rej.empty:
        rej_path = os.path.join(dl_dir, f"rejections_log_{today}.csv")
        rej["scan_time"] = now.strftime("%H:%M:%S")
        rej_first = not os.path.exists(rej_path)
        rej.to_csv(rej_path, mode="a", header=rej_first, index=False)
        print(f"  {C_GREEN}[EXPORT] Rejections →{C_RESET} rejections_log_{today}.csv  (appended)")


def _export_csv_only(df: pd.DataFrame, market: dict):
    dl_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(dl_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    if not df.empty:
        p = os.path.join(dl_dir, f"scan_{ts}.csv")
        df.to_csv(p, index=False)
        print(f"  {C_GREEN}[EXPORT] CSV →{C_RESET} {p}")
    rej = market.get("rejected_detail",[])
    if rej:
        rp = os.path.join(dl_dir, f"rejected_{ts}.csv")
        pd.DataFrame(rej).to_csv(rp, index=False)
        print(f"  {C_GREEN}[EXPORT] Rejections CSV →{C_RESET} {rp}")


def run_scan(timeframe: str = "5m"):
    """Run one full scan, print results, export files."""
    os.system("cls" if os.name == "nt" else "clear")
    sep("═")
    header_line()
    sep("═")

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
    sep("═")

    return df, market


# ── Entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quant Scanner v3 — Terminal Mode")
    parser.add_argument("--tf",    default="5m",  help="Timeframe: 15s/1m/5m/15m")
    parser.add_argument("--loop",  action="store_true", help="Auto-refresh loop")
    parser.add_argument("--once",  action="store_true", help="Single scan and exit")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval seconds")
    args = parser.parse_args()

    if args.once or not args.loop:
        run_scan(args.tf)
        return

    # Continuous loop
    print(f"  Auto-scan every {args.interval}s on {args.tf} bars. Press Ctrl+C to stop.\n")
    scan_count = 0
    while True:
        try:
            scan_count += 1
            print(f"\n  ── Scan #{scan_count} ──")
            run_scan(args.tf)
            print(f"\n  Next scan in {args.interval}s...  (Ctrl+C to stop)")
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n\n  Stopped after {scan_count} scan(s). Files saved to ~/Downloads/")
            break


if __name__ == "__main__":
    main()
