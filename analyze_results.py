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
    """Find the most recent log files"""
    summaries = glob.glob("logs/summary_*.json")
    if not summaries:
        print("No summary files found in logs/")
        return None
    
    # Sort by timestamp in filename
    summaries.sort(reverse=True)
    return summaries

def load_summary(filepath):
    """Load a summary JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_game(summary_path):
    """Analyze results for one game"""
    summary = load_summary(summary_path)
    
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
        trades = stats.get('trades', 0)
        wins = stats.get('wins', 0)
        win_rate = stats.get('win_rate', 0)
        locks = stats.get('locks', 0)
        lock_rate = stats.get('lock_rate', 0)
        net_pnl = stats.get('net_pnl', 0)
        
        total_net += net_pnl
        
        print(f"{strat_name:<20} {trades:>7} {win_rate:>5.1%} {locks:>6} {lock_rate:>5.1%} {net_pnl:>9.1f}¢")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {' '*19} {' '*13} {total_net:>9.1f}¢")
    
    # Detailed stats
    print("\nDetailed Statistics:")
    for strat_name, stats in strategies.items():
        print(f"\n  {strat_name}:")
        print(f"    Total Trades: {stats.get('trades', 0)}")
        print(f"    Wins/Losses: {stats.get('wins', 0)}W - {stats.get('losses', 0)}L")
        print(f"    Locks: {stats.get('locks', 0)} ({stats.get('lock_rate', 0):.1%})")
        print(f"    Stops: {stats.get('stops', 0)}")
        print(f"    Gross P&L: {stats.get('gross_pnl', 0):.1f}¢")
        print(f"    Fees Paid: {stats.get('fees', 0):.1f}¢")
        print(f"    Net P&L: {stats.get('net_pnl', 0):.1f}¢")
        
        if stats.get('trades', 0) > 0:
            print(f"    Avg Win: {stats.get('avg_win', 0):.1f}¢")
            print(f"    Avg Loss: {stats.get('avg_loss', 0):.1f}¢")
            print(f"    Avg Hold: {stats.get('avg_hold_secs', 0)/60:.1f} min")
    
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
        summary = load_summary(summary_path)
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
