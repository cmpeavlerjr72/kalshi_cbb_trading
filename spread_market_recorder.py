# spread_market_recorder.py
# Record spread market data for post-game analysis
#
# This script:
# 1. Fetches orderbook snapshots every 30-60 seconds
# 2. Records prices, spreads, volumes for all tickers
# 3. Saves to CSV for easy analysis tomorrow
# 4. Runs until you stop it (Ctrl+C)

import os
import sys
import time
import csv
import json
import datetime as dt
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Import from existing modules
from combo_vnext import fetch_orderbook, derive_prices, parse_iso, utc_now


@dataclass
class MarketSnapshot:
    """Single snapshot of a spread market's orderbook"""
    timestamp: str
    ticker: str
    
    # Parsed spread info
    team: str
    line: float
    
    # Best prices
    yes_best_bid: Optional[int]
    yes_imp_ask: Optional[int]
    no_best_bid: Optional[int]
    no_imp_ask: Optional[int]
    
    # Spread metrics
    spread_cents: Optional[int]
    midpoint: Optional[float]
    
    # Volume/depth (top level only)
    yes_bid_qty: int = 0
    yes_ask_qty: int = 0
    no_bid_qty: int = 0
    no_ask_qty: int = 0
    
    # Metadata
    fetch_success: bool = True
    error_msg: str = ""


def parse_spread_ticker_simple(ticker: str) -> tuple[Optional[str], Optional[float]]:
    """
    Extract team and line from ticker.
    
    Examples:
        KXNCAAMBSPREAD-26FEB06BELUIC-BEL12 -> ("BEL", 12.0)
        KXNCAAMBSPREAD-26FEB06CONNSJU-CONN7 -> ("CONN", 7.0)
    """
    try:
        parts = ticker.split("-")
        if len(parts) < 3:
            return None, None
        
        last = parts[-1]
        
        import re
        match = re.match(r'^([A-Z]+)(\d+)$', last)
        if not match:
            return None, None
        
        team = match.group(1)
        line = float(match.group(2))
        
        return team, line
    except:
        return None, None


def fetch_market_snapshot(ticker: str) -> MarketSnapshot:
    """Fetch and parse a single market's orderbook"""
    
    team, line = parse_spread_ticker_simple(ticker)
    
    if team is None or line is None:
        return MarketSnapshot(
            timestamp=utc_now().isoformat(),
            ticker=ticker,
            team="UNKNOWN",
            line=0.0,
            yes_best_bid=None,
            yes_imp_ask=None,
            no_best_bid=None,
            no_imp_ask=None,
            spread_cents=None,
            midpoint=None,
            fetch_success=False,
            error_msg="Could not parse ticker"
        )
    
    try:
        # Fetch orderbook
        ob = fetch_orderbook(ticker)
        prices = derive_prices(ob)
        
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_bid = prices.get("best_no_bid")
        no_ask = prices.get("imp_no_ask")
        
        # Calculate spread and midpoint
        spread = None
        midpoint = None
        
        if yes_bid is not None and yes_ask is not None:
            spread = yes_ask - yes_bid
            midpoint = (yes_bid + yes_ask) / 2
        
        # Get top-level quantities
        yes_levels = prices.get("yes_levels", [])
        no_levels = prices.get("no_levels", [])
        
        yes_bid_qty = yes_levels[0][1] if yes_levels else 0
        no_bid_qty = no_levels[0][1] if no_levels else 0
        
        # For asks, we need to look at opposite book
        # YES ask comes from NO bid at complementary price
        yes_ask_qty = no_bid_qty  # Approximate
        no_ask_qty = yes_bid_qty  # Approximate
        
        return MarketSnapshot(
            timestamp=utc_now().isoformat(),
            ticker=ticker,
            team=team,
            line=line,
            yes_best_bid=yes_bid,
            yes_imp_ask=yes_ask,
            no_best_bid=no_bid,
            no_imp_ask=no_ask,
            spread_cents=spread,
            midpoint=midpoint,
            yes_bid_qty=yes_bid_qty,
            yes_ask_qty=yes_ask_qty,
            no_bid_qty=no_bid_qty,
            no_ask_qty=no_ask_qty,
            fetch_success=True,
            error_msg=""
        )
    
    except Exception as e:
        return MarketSnapshot(
            timestamp=utc_now().isoformat(),
            ticker=ticker,
            team=team,
            line=line,
            yes_best_bid=None,
            yes_imp_ask=None,
            no_best_bid=None,
            no_imp_ask=None,
            spread_cents=None,
            midpoint=None,
            fetch_success=False,
            error_msg=str(e)
        )


class SpreadMarketRecorder:
    """
    Record spread market data at regular intervals.
    """
    
    def __init__(
        self,
        tickers: List[str],
        output_file: str,
        interval_secs: int = 60,
        label: str = "spread_recording"
    ):
        """
        Args:
            tickers: List of spread tickers to monitor
            output_file: CSV filename to write to
            interval_secs: How often to snapshot (default: 60 seconds)
            label: Label for this recording session
        """
        self.tickers = tickers
        self.output_file = output_file
        self.interval_secs = interval_secs
        self.label = label
        
        self.snapshots_taken = 0
        self.start_time = utc_now()
        
        # Initialize CSV file
        self._init_csv()
    
    def _init_csv(self):
        """Create CSV file with headers"""
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'ticker', 'team', 'line',
                'yes_best_bid', 'yes_imp_ask', 'no_best_bid', 'no_imp_ask',
                'spread_cents', 'midpoint',
                'yes_bid_qty', 'yes_ask_qty', 'no_bid_qty', 'no_ask_qty',
                'fetch_success', 'error_msg'
            ])
            writer.writeheader()
        
        print(f"✅ Initialized recording file: {self.output_file}")
    
    def record_snapshot(self):
        """Fetch and record one snapshot of all markets"""
        
        snapshots = []
        
        print(f"\n[{utc_now().strftime('%H:%M:%S')}] Recording snapshot #{self.snapshots_taken + 1}...")
        
        for ticker in self.tickers:
            snapshot = fetch_market_snapshot(ticker)
            snapshots.append(snapshot)
            
            # Brief status
            status = "✅" if snapshot.fetch_success else "❌"
            if snapshot.fetch_success and snapshot.yes_imp_ask is not None:
                print(f"  {status} {snapshot.team:4} {snapshot.line:>4.0f}: "
                      f"{snapshot.yes_imp_ask:>3}¢ ask, {snapshot.spread_cents:>2}¢ spread")
            else:
                print(f"  {status} {ticker}: {snapshot.error_msg}")
            
            # Small delay to avoid rate limits
            time.sleep(0.2)
        
        # Write to CSV
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'ticker', 'team', 'line',
                'yes_best_bid', 'yes_imp_ask', 'no_best_bid', 'no_imp_ask',
                'spread_cents', 'midpoint',
                'yes_bid_qty', 'yes_ask_qty', 'no_bid_qty', 'no_ask_qty',
                'fetch_success', 'error_msg'
            ])
            
            for snap in snapshots:
                writer.writerow(asdict(snap))
        
        self.snapshots_taken += 1
        
        elapsed = (utc_now() - self.start_time).total_seconds()
        print(f"✅ Snapshot saved. Total: {self.snapshots_taken} | Elapsed: {elapsed/60:.1f} min")
    
    def run(self, duration_minutes: Optional[int] = None):
        """
        Run the recorder until stopped (Ctrl+C) or duration expires.
        
        Args:
            duration_minutes: Optional max duration. If None, runs indefinitely.
        """
        print("\n" + "="*70)
        print(f"  SPREAD MARKET RECORDER")
        print("="*70)
        print(f"Recording {len(self.tickers)} markets")
        print(f"Output: {self.output_file}")
        print(f"Interval: {self.interval_secs}s")
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
        else:
            print(f"Duration: Until stopped (Ctrl+C)")
        print("="*70)
        
        deadline = None
        if duration_minutes:
            deadline = time.time() + duration_minutes * 60
        
        try:
            while True:
                # Record snapshot
                self.record_snapshot()
                
                # Check if we should stop
                if deadline and time.time() >= deadline:
                    print(f"\n⏰ Reached {duration_minutes} minute duration limit")
                    break
                
                # Wait for next interval
                print(f"\n⏸️  Waiting {self.interval_secs}s until next snapshot...")
                time.sleep(self.interval_secs)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Recording stopped by user (Ctrl+C)")
        
        finally:
            self._print_summary()
    
    def _print_summary(self):
        """Print final summary"""
        elapsed = (utc_now() - self.start_time).total_seconds()
        
        print("\n" + "="*70)
        print("  RECORDING COMPLETE")
        print("="*70)
        print(f"Total snapshots: {self.snapshots_taken}")
        print(f"Duration: {elapsed/60:.1f} minutes")
        print(f"Markets tracked: {len(self.tickers)}")
        print(f"Data saved to: {self.output_file}")
        print(f"\nTo analyze: python analyze_recordings.py {self.output_file}")
        print("="*70)


# =============================================================================
# TONIGHT'S GAMES
# =============================================================================

GAMES = {
    "belmont_uic": {
        "label": "Belmont vs UIC",
        "tickers": [
            "KXNCAAMBSPREAD-26FEB06BELUIC-BEL12",
            "KXNCAAMBSPREAD-26FEB06BELUIC-BEL15",
            "KXNCAAMBSPREAD-26FEB06BELUIC-BEL18",
            "KXNCAAMBSPREAD-26FEB06BELUIC-BEL3",
            "KXNCAAMBSPREAD-26FEB06BELUIC-BEL6",
            "KXNCAAMBSPREAD-26FEB06BELUIC-BEL9",
            "KXNCAAMBSPREAD-26FEB06BELUIC-UIC12",
            "KXNCAAMBSPREAD-26FEB06BELUIC-UIC3",
            "KXNCAAMBSPREAD-26FEB06BELUIC-UIC6",
            "KXNCAAMBSPREAD-26FEB06BELUIC-UIC9"
        ]
    },
    "uconn_sju": {
        "label": "UConn vs St. John's",
        "tickers": [
            "KXNCAAMBSPREAD-26FEB06CONNSJU-CONN1",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-CONN10",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-CONN13",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-CONN16",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-CONN4",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-CONN7",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-SJU11",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-SJU14",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-SJU2",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-SJU5",
            "KXNCAAMBSPREAD-26FEB06CONNSJU-SJU8"
        ]
    },
    "murray_siu": {
        "label": "Murray State vs SIU",
        "tickers": [
            "KXNCAAMBSPREAD-26FEB06MURRSIU-MURR11",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-MURR14",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-MURR2",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-MURR5",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-MURR8",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-SIU1",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-SIU10",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-SIU13",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-SIU16",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-SIU4",
            "KXNCAAMBSPREAD-26FEB06MURRSIU-SIU7"
        ]
    },
    "bradley_uni": {
        "label": "Bradley vs Northern Iowa",
        "tickers": [
            "KXNCAAMBSPREAD-26FEB06BRADUNI-BRAD12",
            "KXNCAAMBSPREAD-26FEB06BRADUNI-BRAD3",
            "KXNCAAMBSPREAD-26FEB06BRADUNI-BRAD6",
            "KXNCAAMBSPREAD-26FEB06BRADUNI-BRAD9",
            "KXNCAAMBSPREAD-26FEB06BRADUNI-UNI12",
            "KXNCAAMBSPREAD-26FEB06BRADUNI-UNI15",
            "KXNCAAMBSPREAD-26FEB06BRADUNI-UNI18",
            "KXNCAAMBSPREAD-26FEB06BRADUNI-UNI3",
            "KXNCAAMBSPREAD-26FEB06BRADUNI-UNI6",
            "KXNCAAMBSPREAD-26FEB06BRADUNI-UNI9"
        ]
    }
}


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Record spread market data")
    parser.add_argument(
        "--games",
        nargs="+",
        choices=list(GAMES.keys()) + ["all"],
        default=["all"],
        help="Which games to record (default: all)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between snapshots (default: 60)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Minutes to record (default: until stopped)"
    )
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="Output directory (default: recordings/)"
    )
    
    args = parser.parse_args()
    
    # Determine which games to record
    if "all" in args.games:
        games_to_record = list(GAMES.keys())
    else:
        games_to_record = args.games
    
    print(f"\n📊 Recording {len(games_to_record)} game(s):")
    for game_key in games_to_record:
        game = GAMES[game_key]
        print(f"   - {game['label']} ({len(game['tickers'])} markets)")
    
    # Create recorders for each game
    recorders = []
    
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    
    for game_key in games_to_record:
        game = GAMES[game_key]
        
        output_file = os.path.join(
            args.output_dir,
            f"{game_key}_{timestamp}.csv"
        )
        
        recorder = SpreadMarketRecorder(
            tickers=game['tickers'],
            output_file=output_file,
            interval_secs=args.interval,
            label=game['label']
        )
        
        recorders.append((game_key, recorder))
    
    # If recording multiple games, we need to interleave
    # For simplicity, let's record them sequentially in each interval
    
    if len(recorders) == 1:
        # Single game - simple
        _, recorder = recorders[0]
        recorder.run(duration_minutes=args.duration)
    
    else:
        # Multiple games - custom loop
        print("\n" + "="*70)
        print("  MULTI-GAME RECORDING")
        print("="*70)
        print(f"Recording {len(recorders)} games simultaneously")
        print(f"Interval: {args.interval}s per game")
        if args.duration:
            print(f"Duration: {args.duration} minutes")
        print("="*70)
        
        start_time = time.time()
        deadline = None
        if args.duration:
            deadline = start_time + args.duration * 60
        
        snapshot_count = 0
        
        try:
            while True:
                snapshot_count += 1
                print(f"\n{'='*70}")
                print(f"Snapshot #{snapshot_count} - {utc_now().strftime('%H:%M:%S')}")
                print(f"{'='*70}")
                
                for game_key, recorder in recorders:
                    print(f"\n📊 {GAMES[game_key]['label']}:")
                    recorder.record_snapshot()
                
                if deadline and time.time() >= deadline:
                    print(f"\n⏰ Reached {args.duration} minute duration limit")
                    break
                
                print(f"\n⏸️  Waiting {args.interval}s until next snapshot...")
                time.sleep(args.interval)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Recording stopped by user (Ctrl+C)")
        
        finally:
            elapsed = time.time() - start_time
            print("\n" + "="*70)
            print("  RECORDING COMPLETE")
            print("="*70)
            print(f"Total snapshots: {snapshot_count}")
            print(f"Duration: {elapsed/60:.1f} minutes")
            print(f"\nData files:")
            for game_key, recorder in recorders:
                print(f"  - {recorder.output_file}")
            print("="*70)


if __name__ == "__main__":
    main()
