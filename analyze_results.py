# analyze_results.py
# Quick analysis of last night's multi-strategy test results
#
# Usage: python analyze_results.py

import os
import json
import pandas as pd
import glob
from datetime import datetime

def find_latest_logs():
    """Find the most recent log sessions.

    Prefer summary_*.json if they exist. Otherwise fall back to positions_*.csv.
    Returns a list of file paths.
    """
    summaries = glob.glob("logs/summary_*.json")
    if summaries:
        summaries.sort(reverse=True)
        return summaries

    # Fallback: no summaries (likely crashed before save). Use positions logs.
    positions = glob.glob("logs/positions_*.csv")
    if not positions:
        print("No summary files found, and no positions files found in logs/")
        return None

    positions.sort(reverse=True)
    return positions


def load_summary(filepath):
    """Load a summary JSON file"""
    with open(filepath, 'r', encoding="utf-8") as f:
        return json.load(f)
    
def _infer_log_paths_from_positions(positions_path: str):
    """Given logs/positions_<session_id>.csv infer trades/events/snapshots paths."""
    base = os.path.basename(positions_path)
    if not base.startswith("positions_") or not base.endswith(".csv"):
        return {}

    session_id = base[len("positions_"):-len(".csv")]
    return {
        "positions": f"logs/positions_{session_id}.csv",
        "trades": f"logs/trades_{session_id}.csv",
        "events": f"logs/events_{session_id}.csv",
        "snapshots": f"logs/snapshots_{session_id}.csv",
        # summary path doesn't exist in this fallback mode
    }

def build_summary_from_positions(positions_path: str):
    """Build a minimal summary dict from positions CSV so analysis can run."""
    log_files = _infer_log_paths_from_positions(positions_path)

    # Infer game label from session id prefix: "<game_label>_<YYYYMMDD_HHMMSS>"
    session_id = os.path.basename(positions_path)[len("positions_"):-len(".csv")]
    game_label = session_id
    # try split on last timestamp pattern
    parts = session_id.rsplit("_", 2)
    if len(parts) == 3:
        game_label = parts[0]

    summary = {
        "game_label": game_label,
        "duration_secs": 0,
        "strategies": {},
        "log_files": log_files,
    }

    if not os.path.exists(positions_path):
        return summary

    df = pd.read_csv(positions_path)
    if len(df) == 0:
        return summary

    # Ensure numeric
    for col in ["gross_pnl", "net_pnl", "hold_secs", "entry_fee", "exit_fee"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Strategy aggregation
    for strat, g in df.groupby("strategy"):
        trades = len(g)
        wins = int((g["net_pnl"] > 0).sum())
        losses = int((g["net_pnl"] < 0).sum())
        locks = int((g["exit_type"] == "lock").sum())
        stops = int((g["exit_type"] == "stop").sum())

        gross_pnl = float(g["gross_pnl"].sum())
        fees = float((g["entry_fee"] + g["exit_fee"]).sum())
        net_pnl = float(g["net_pnl"].sum())

        avg_hold = float(g["hold_secs"].mean()) if trades else 0.0
        avg_win = float(g.loc[g["net_pnl"] > 0, "net_pnl"].mean()) if wins else 0.0
        avg_loss = float(g.loc[g["net_pnl"] < 0, "net_pnl"].mean()) if losses else 0.0

        summary["strategies"][strat] = {
            "strategy": strat,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / trades) if trades else 0.0,
            "locks": locks,
            "lock_rate": (locks / trades) if trades else 0.0,
            "stops": stops,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
            "avg_hold_secs": avg_hold,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

    # Rough duration estimate from entry/exit times if present
    if "entry_time" in df.columns and "exit_time" in df.columns:
        try:
            t0 = pd.to_datetime(df["entry_time"], errors="coerce").min()
            t1 = pd.to_datetime(df["exit_time"], errors="coerce").max()
            if pd.notna(t0) and pd.notna(t1):
                summary["duration_secs"] = int((t1 - t0).total_seconds())
        except Exception:
            pass

    return summary


def analyze_game(summary_path):
    """Analyze results for one game"""
    if summary_path.endswith(".json"):
        summary = load_summary(summary_path)
    else:
        summary = build_summary_from_positions(summary_path)

    
    game_label = summary.get('game_label', 'Unknown')
    print(f"\n{'='*70}")
    print(f"  {game_label}")
    print(f"{'='*70}")
    
    # High-level stats
    duration = summary.get('duration_secs', 0)
    print(f"\nDuration: {duration/60:.0f} minutes")
    
    # Strategy results
    strategies = summary.get('strategies', {})
    
    print("\nStrategy Performance:")
    print(f"{'Strategy':<20} {'Trades':>7} {'Win%':>6} {'Locks':>6} {'Lock%':>6} {'Net P&L':>10}")
    print("-" * 70)
    
    total_net = 0
    
    for strat_name, stats in strategies.items():
        trades = int(stats.get('trades', 0))
        wins = int(stats.get('wins', 0))
        win_rate = float(stats.get('win_rate', 0.0))
        locks = int(stats.get('locks', 0))
        lock_rate = float(stats.get('lock_rate', 0.0))
        net_pnl = float(stats.get('net_pnl', 0.0))

        total_net += net_pnl

        if trades == 0:
            win_rate_str = "   — "
            lock_rate_str = "   — "
        else:
            win_rate_str = f"{win_rate:>5.1%}"
            lock_rate_str = f"{lock_rate:>5.1%}"

        print(
            f"{strat_name:<20} {trades:>7} {win_rate_str} {locks:>6} {lock_rate_str} {net_pnl:>9.1f}¢"
        )

    
    print("-" * 70)
    print(f"{'TOTAL':<20} {' '*19} {' '*13} {total_net:>9.1f}¢")
    
    # Detailed stats
    print("\nDetailed Statistics:")
    for strat_name, stats in strategies.items():
        print(f"\n  {strat_name}:")
        trades = int(stats.get('trades', 0))
        print(f"    Total Trades: {trades}")
        print(f"    Wins/Losses: {stats.get('wins', 0)}W - {stats.get('losses', 0)}L")
        print(f"    Locks: {stats.get('locks', 0)} ({float(stats.get('lock_rate', 0.0)):.1%})")
        print(f"    Stops: {stats.get('stops', 0)}")
        print(f"    Gross P&L: {float(stats.get('gross_pnl', 0.0)):.1f}¢")
        print(f"    Fees Paid: {float(stats.get('fees', 0.0)):.1f}¢")
        print(f"    Net P&L: {float(stats.get('net_pnl', 0.0)):.1f}¢")

        if trades == 0:
            print("    Note: NO TRADES (no averages computed)")
        else:
            print(f"    Avg Win: {float(stats.get('avg_win', 0.0)):.1f}¢")
            print(f"    Avg Loss: {float(stats.get('avg_loss', 0.0)):.1f}¢")
            print(f"    Avg Hold: {float(stats.get('avg_hold_secs', 0.0))/60:.1f} min")

    
    # Load detailed position data if available
    positions_file = summary.get('log_files', {}).get('positions')
    if positions_file and os.path.exists(positions_file):
        analyze_positions(positions_file, game_label)
    
    return total_net

def analyze_positions(filepath, game_label):
    """Detailed analysis of position data"""
    try:
        df = pd.read_csv(filepath)
        
        if len(df) == 0:
            return
        
        print("\n  Position Analysis:")
        
        # Lock success analysis
        total_entries = len(df)
        locks = len(df[df['exit_type'] == 'lock'])
        stops = len(df[df['exit_type'] == 'stop'])
        
        print(f"    Lock Success: {locks}/{total_entries} ({locks/total_entries:.1%})")
        print(f"    Stop Outs: {stops}/{total_entries} ({stops/total_entries:.1%})")
        
        # Hold time analysis
        avg_hold = df['hold_secs'].mean()
        print(f"    Avg Hold Time: {avg_hold/60:.1f} minutes")
        
        # Fee analysis
        total_fees = (df['entry_fee'] + df['exit_fee']).sum()
        total_gross = df['gross_pnl'].abs().sum()
        if total_gross > 0:
            fee_ratio = total_fees / total_gross
            print(f"    Total Fees: {total_fees:.1f}¢")
            print(f"    Fee Ratio: {fee_ratio:.1%} of gross P&L")
        
        # Best/worst trades
        best = df.loc[df['net_pnl'].idxmax()]
        worst = df.loc[df['net_pnl'].idxmin()]
        
        print(f"\n    Best Trade: {best['strategy']} - {best['net_pnl']:.1f}¢ ({best['exit_type']})")
        print(f"    Worst Trade: {worst['strategy']} - {worst['net_pnl']:.1f}¢ ({worst['exit_type']})")
        
    except Exception as e:
        print(f"    Error analyzing positions: {e}")

def analyze_trades(filepath):
    """Analyze execution quality from trades log"""
    try:
        df = pd.read_csv(filepath)
        
        # Entry attempts vs fills
        entry_attempts = len(df[df['action'] == 'entry_attempt'])
        entry_fills = len(df[df['action'] == 'entry_fill'])
        
        if entry_attempts > 0:
            fill_rate = entry_fills / entry_attempts
            print(f"\n  Execution Quality:")
            print(f"    Entry Fill Rate: {entry_fills}/{entry_attempts} ({fill_rate:.1%})")
        
        # Liquidity check effectiveness
        liquidity_skips = len(df[(df['action'] == 'entry_attempt') & (df['liquidity_ok'] == False)])
        if liquidity_skips > 0:
            print(f"    Liquidity Blocks: {liquidity_skips} (prevented bad orders)")
        
        # VWAP vs intended price (slippage)
        fills = df[df['action'].str.contains('fill') & df['vwap'].notna()]
        if len(fills) > 0:
            fills['slippage'] = fills['vwap'] - fills['price']
            avg_slippage = fills['slippage'].mean()
            print(f"    Avg Slippage: {avg_slippage:.2f}¢")
        
    except Exception as e:
        print(f"  Error analyzing trades: {e}")

def main():
    print("="*70)
    print("  MULTI-STRATEGY RESULTS ANALYSIS")
    print("="*70)
    
    summaries = find_latest_logs()
    if not summaries:
        print("\nNo results found. Did you run tonight_runner.py?")
        return 1
    
    print(f"\nFound {len(summaries)} game(s)")
    
    total_portfolio_pnl = 0
    
    for summary_path in summaries:
        game_pnl = analyze_game(summary_path)
        total_portfolio_pnl += game_pnl
        
        # Also analyze trades if available
        if summary_path.endswith(".json"):
            summary = load_summary(summary_path)
        else:
            summary = build_summary_from_positions(summary_path)

        trades_file = summary.get('log_files', {}).get('trades')
        if trades_file and os.path.exists(trades_file):
            analyze_trades(trades_file)

    
    # Portfolio summary
    print("\n" + "="*70)
    print("  PORTFOLIO SUMMARY")
    print("="*70)
    print(f"\nTotal Net P&L: {total_portfolio_pnl:.1f}¢ (${total_portfolio_pnl/100:.2f})")
    print(f"Games Analyzed: {len(summaries)}")
    
    if total_portfolio_pnl > 0:
        print("\n✓ PROFITABLE SESSION")
    elif total_portfolio_pnl < 0:
        print("\n✗ LOSING SESSION")
    else:
        print("\n→ BREAKEVEN SESSION")
    
    print("\n" + "="*70)
    print("\nDetailed logs available in logs/ directory:")
    print("  - snapshots_*.csv  : Market tick data")
    print("  - trades_*.csv     : Order execution details")
    print("  - positions_*.csv  : Complete trade records")
    print("  - events_*.csv     : Strategy decision log")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
