# tonight_runner.py
# Production multi-strategy runner for Feb 5, 2026 games
#
# GAMES:
# 1. Iowa at Washington (Iowa 58% - Away Favorite)
# 2. Utah St. at New Mexico (Utah St. 54% - Road Favorite)
# 3. Washington St. at Oregon St. (Washington St. 60% - Road Favorite)
#
# Each game runs 3 strategies in parallel with separate allocations

import os
import sys
import threading
import json
from typing import Dict, Any, List
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from combo_vnext import (
    _load_private_key,
    _get,
    get_markets_in_series,
    parse_iso,
    utc_now,
    print_status,
    SERIES_TICKER,
)

from espn_game_clock import EspnGameClock

# Import all the fixed strategies
from production_strategies import (
    ModelEdgeStrategy,
    MeanReversionStrategy,
    SpreadCaptureStrategy,
    GameRunner,
)

# =============================================================================
# TONIGHT'S GAMES CONFIGURATION
# =============================================================================

GAMES = [
    {
        "label": "Tarleton_vs_CalBaptist",
        "team_name": "TARL",  # Team code for ticker resolution
        "model_p_win": 0.49,
        "segment": "home Dog",
        "espn_date": "20260205",
        "espn_team": "TAR", #
        "espn_opponent": "CBU", #
    },
    {
        "label": "Fairfield at Sacred Heart",
        "team_name": "FAIR", 
        "model_p_win": 0.58,
        "segment": "Away Fav",
        "espn_date": "20260205",
        "espn_team": "FAIR", #
        "espn_opponent": "SHU", #
    },
]

# Per-strategy allocations (dollars per game)
ALLOCATIONS = {
    "model_edge": 2.00,
    "mean_reversion": 2.00,
    "spread_capture": 2.00,
}

TOTAL_PER_GAME = sum(ALLOCATIONS.values())

# =============================================================================
# MARKET RESOLUTION
# =============================================================================

def find_market_for_team(private_key, team_name: str) -> tuple:
    """
    Find the ML market for a given team.
    Uses ticker suffix matching first for accuracy.
    """
    markets = get_markets_in_series(private_key, SERIES_TICKER)
    team_code = team_name.upper()
    
    # Try exact ticker suffix match first
    suffix = f"-{team_code}"
    candidates = [
        m for m in markets
        if (m.get("ticker") or "").upper().endswith(suffix)
    ]
    
    # Fallback: title search
    if not candidates:
        team_l = team_name.lower()
        for m in markets:
            title = (m.get("title") or "").lower()
            if team_l in title:
                candidates.append(m)
    
    if not candidates:
        raise RuntimeError(f"No markets found for team '{team_name}'")
    
    # Filter to future markets
    now = utc_now()
    future = [m for m in candidates if parse_iso(m["close_time"]) > now]
    if future:
        candidates = future
    
    # Take earliest closing
    candidates.sort(key=lambda m: parse_iso(m["close_time"]))
    chosen = candidates[0]
    
    return chosen["ticker"], chosen

# =============================================================================
# ESPN CLOCK SETUP
# =============================================================================

def setup_espn_clock(game_config: Dict[str, Any]) -> EspnGameClock:
    """Create ESPN clock for a game"""
    try:
        clock = EspnGameClock(
            yyyymmdd=game_config["espn_date"],
            team_code=game_config["espn_team"],
            opponent_code=game_config["espn_opponent"],
            cache_ttl_secs=10,
        )
        print_status(
            f"[{game_config['label']}] ESPN clock configured: "
            f"{game_config['espn_team']} vs {game_config['espn_opponent']}"
        )
        return clock
    except Exception as e:
        print_status(
            f"[{game_config['label']}] ESPN clock setup failed: {e} "
            f"(will use Kalshi close time)"
        )
        return None

# =============================================================================
# GAME WORKER
# =============================================================================

def run_game(
    game_config: Dict[str, Any],
    private_key,
    results: Dict[str, Any]
):
    """Worker function for one game"""
    label = game_config["label"]
    
    try:
        print_status(f"\n[{label}] Initializing...")
        
        # Find market
        ticker, market = find_market_for_team(private_key, game_config["team_name"])
        print_status(f"[{label}] Found market: {ticker}")
        print_status(f"[{label}] Title: {market.get('title', 'N/A')}")
        
        # Setup ESPN clock
        espn_clock = setup_espn_clock(game_config)
        
        # Create strategies
        model_fair = int(game_config["model_p_win"] * 100)
        
        strategies = [
            ModelEdgeStrategy(
                max_capital=ALLOCATIONS["model_edge"],
                model_fair_cents=model_fair
            ),
            MeanReversionStrategy(
                max_capital=ALLOCATIONS["mean_reversion"]
            ),
            SpreadCaptureStrategy(
                max_capital=ALLOCATIONS["spread_capture"]
            ),
        ]
        
        print_status(
            f"[{label}] Strategies initialized: "
            f"model@{model_fair}¢ | MR | SC"
        )
        
        # Create runner
        runner = GameRunner(
            game_label=label,
            ticker=ticker,
            market=market,
            strategies=strategies,
            private_key=private_key,
            espn_clock=espn_clock,
        )
        
        # Run
        summary = runner.run()
        results[label] = summary
        
        print_status(f"[{label}] ✓ Complete")
        
    except Exception as e:
        print_status(f"[{label}] ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[label] = {"error": str(e)}

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("  PRODUCTION MULTI-STRATEGY TESTING")
    print("  February 5, 2026")
    print("="*80)
    print("\nGAMES:")
    for i, game in enumerate(GAMES, 1):
        print(f"  {i}. {game['label']}")
        print(f"     Model: {game['team_name']} @ {game['model_p_win']:.0%}")
        print(f"     Segment: {game['segment']}")
    
    print(f"\nPER-GAME ALLOCATION: ${TOTAL_PER_GAME:.2f}")
    for strat, alloc in ALLOCATIONS.items():
        print(f"  {strat}: ${alloc:.2f}")
    
    print(f"\nTOTAL CAPITAL AT RISK: ${TOTAL_PER_GAME * len(GAMES):.2f}")
    
    # Load credentials
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    
    if not api_key or not key_path:
        print("\n✗ ERROR: Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        return 1
    
    try:
        private_key = _load_private_key(key_path)
        print(f"\n✓ Private key loaded")
    except Exception as e:
        print(f"\n✗ ERROR loading private key: {e}")
        return 1
    
    # Check balance
    try:
        resp = _get(private_key, "/trade-api/v2/portfolio/balance")
        balance = int(resp.get("balance", 0)) / 100
        print(f"✓ Account balance: ${balance:.2f}")
        
        required = TOTAL_PER_GAME * len(GAMES)
        if balance < required:
            print(f"\n⚠ WARNING: Balance ${balance:.2f} < Required ${required:.2f}")
            print("  Consider reducing allocations or number of games")
    except Exception as e:
        print(f"\n✗ ERROR checking balance: {e}")
        return 1
    
    print("\n" + "="*80)
    print("READY TO START")
    print("="*80)
    print("\nPress ENTER to begin trading...")
    print("(Press Ctrl+C to abort)\n")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nAborted")
        return 0
    
    print("\n" + "="*80)
    print("STARTING ALL GAMES")
    print("="*80 + "\n")
    
    # Run all games in parallel
    results = {}
    threads = []
    
    for game in GAMES:
        # Each thread gets its own private key for thread safety
        game_private_key = _load_private_key(key_path)
        
        t = threading.Thread(
            target=run_game,
            args=(game, game_private_key, results),
            name=game["label"],
            daemon=False,
        )
        t.start()
        threads.append(t)
        print_status(f"Started: {game['label']}")
    
    # Wait for all games
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print_status("\n⚠ Interrupted - games will continue running")
        print_status("  (They will stop automatically near market close)")
    
    # Final summary
    print("\n" + "="*80)
    print("  ALL GAMES COMPLETE - FINAL RESULTS")
    print("="*80 + "\n")
    
    total_net = 0.0
    
    for label, result in results.items():
        if "error" in result:
            print(f"{label}: ERROR - {result['error']}")
            continue
        
        print(f"\n{label}:")
        
        strategies = result.get("strategies", {})
        game_net = 0.0
        
        for strat_name, stats in strategies.items():
            net = stats.get("net_pnl", 0)
            game_net += net
            
            print(f"  {strat_name}:")
            print(f"    Trades: {stats.get('trades', 0)} "
                  f"({stats.get('wins', 0)}W-{stats.get('losses', 0)}L)")
            print(f"    Locks: {stats.get('locks', 0)} "
                  f"({stats.get('lock_rate', 0):.1%})")
            print(f"    Net: {net:.1f}¢ (${net/100:.2f})")
        
        print(f"  GAME TOTAL: {game_net:.1f}¢ (${game_net/100:.2f})")
        total_net += game_net
    
    print("\n" + "="*80)
    print(f"PORTFOLIO NET P&L: {total_net:.1f}¢ (${total_net/100:.2f})")
    print("="*80 + "\n")
    
    print("Log files created in ./logs/")
    print("Review snapshots, trades, positions, and events CSVs tomorrow")
    print("\nKey files to analyze:")
    print("  - snapshots_*.csv : Market tick data")
    print("  - trades_*.csv    : Individual order attempts/fills")
    print("  - positions_*.csv : Complete trade lifecycle")
    print("  - events_*.csv    : Strategy decisions")
    print("  - summary_*.json  : Final statistics\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
