#!/usr/bin/env python3
# tonights_kalshi_ledger.py
# Pull Kalshi ledger for tonight's specific games and reconcile

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from fetch_kalshi_ledger import (
    _load_private_key,
    fetch_fills,
    fetch_orders,
    export_fills_csv,
    summarize_fills,
)

load_dotenv()

# Tonight's games (Feb 5, 2026)
TONIGHTS_GAMES = [
    "FAIR",  # Fairfield (completed)
    "TAR",   # Tarleton (still running)
    "IOWA",  # Iowa (if you added it)
    "USU",   # Utah State (if you added it)
    "WSU",   # Washington State (if you added it)
]

def main():
    print("\n" + "="*70)
    print("  TONIGHT'S GAMES - KALSHI LEDGER")
    print("  Feb 5, 2026")
    print("="*70 + "\n")
    
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    
    if not api_key or not key_path:
        print("ERROR: Set credentials")
        return
    
    private_key = _load_private_key(key_path)
    
    # Fetch all fills from today (midnight UTC in Unix milliseconds)
    # 2026-02-06 00:00:00 UTC
    today_start_dt = datetime(2026, 2, 6, 0, 0, 0, tzinfo=timezone.utc)
    today_start_ms = int(today_start_dt.timestamp() * 1000)
    
    print(f"Fetching all fills from tonight (since {today_start_dt})...")
    all_fills = fetch_fills(private_key, min_ts=str(today_start_ms))
    
    print(f"\nTotal fills tonight: {len(all_fills)}")
    
    # Group by game
    by_game = {}
    for fill in all_fills:
        ticker = fill.get("ticker", "")
        # Extract team code from ticker (e.g., KXNCAAMBGAME-26FEB05FAIR-FAIR)
        game_code = None
        for code in TONIGHTS_GAMES:
            if code in ticker:
                game_code = code
                break
        
        if game_code:
            if game_code not in by_game:
                by_game[game_code] = []
            by_game[game_code].append(fill)
    
    # Print per-game summary
    print("\n" + "="*70)
    print("PER-GAME BREAKDOWN")
    print("="*70)
    
    for game_code in TONIGHTS_GAMES:
        if game_code not in by_game:
            print(f"\n{game_code}: No fills")
            continue
        
        fills = by_game[game_code]
        summary = summarize_fills(fills)
        
        print(f"\n{game_code}:")
        print(f"  Fills: {summary['total_fills']}")
        print(f"  Contracts: {summary['total_contracts']} ({summary['buy_contracts']} buys, {summary['sell_contracts']} sells)")
        print(f"  Fees: {summary['estimated_fees_cents']:.2f}¢")
        
        # Export
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"kalshi_{game_code}_{ts}.csv"
        export_fills_csv(fills, filename)
    
    # Overall summary
    print("\n" + "="*70)
    print("TOTAL TONIGHT")
    print("="*70)
    
    all_summary = summarize_fills(all_fills)
    print(f"  Fills: {all_summary['total_fills']}")
    print(f"  Contracts: {all_summary['total_contracts']}")
    print(f"  Fees: {all_summary['estimated_fees_cents']:.2f}¢")
    
    # Export all
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_fills_csv(all_fills, f"kalshi_tonight_all_{ts}.csv")
    
    print("\n✓ Done!\n")


if __name__ == "__main__":
    main()