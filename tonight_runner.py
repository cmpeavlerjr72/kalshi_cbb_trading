# tonight_runner.py
# Production multi-strategy runner — Feb 6, 2026
#
# Changes from Feb 5:
#   - Updated game configs for tonight's slate
#   - Integrated P1–P5 fixes from production_strategies.py
#   - Per-strategy allocations slightly rebalanced (MR gets more, ME gets less)
#   - ESPN clock wiring for all games
#
# Run with: python tonight_runner.py

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

from production_strategies import (
    MeanReversionStrategy,
    GameRunner,
)



# =============================================================================
# TONIGHT'S GAMES — Feb 6, 2026
# =============================================================================
# Fill these in with your model's probabilities + ESPN team codes.
# team_name must match the Kalshi ticker suffix (e.g. "IOWA", "USU", "WSU").

GAMES = [
    # EXAMPLE — replace with tonight's actual games before running:
    #
    {
        "label": "USC at Penn St.",
        "team_name": "USC",
        "model_p_win": 0.55,
        "partner_p_win":0.55,
        "segment": "Home Fav",
        "espn_date": "20260208",
        "espn_team": "USC",
        "espn_opponent": "PSU",
    },
    {
        "label": "Tulsa at South Florida",
        "team_name": "USF",
        "model_p_win": 0.51,
        "partner_p_win":0.51,
        "segment": "Home Fav",
        "espn_date": "20260208",
        "espn_team": "USF",
        "espn_opponent": "TLSA",
    },
    {
        "label": "Texas Tech at West Virginia",
        "team_name": "TTU",
        "model_p_win": 0.59,
        "partner_p_win":0.59,
        "segment": "Home Fav",
        "espn_date": "20260208",
        "espn_team": "TTU",
        "espn_opponent": "WVU",
    },
    {
        "label": "Wichita St. at Tulane",
        "team_name": "WICH",
        "model_p_win": 0.62,
        "partner_p_win":0.62,
        "segment": "Away Dog",
        "espn_date": "20260208",
        "espn_team": "WICH",
        "espn_opponent": "TULN",
    },
    {
        "label": "UCF at Cincinnati",
        "team_name": "UCF",
        "model_p_win": 0.51,
        "partner_p_win":0.51,
        "segment": "Away Fav",
        "espn_date": "20260208",
        "espn_team": "UCF",
        "espn_opponent": "CIN",
    },


]


# =============================================================================
# MARKET RESOLUTION
# =============================================================================

def find_market_for_team(private_key, team_name: str) -> tuple:
    """Find the ML market for a given team (ticker suffix match, then title fallback)."""
    markets = get_markets_in_series(private_key, SERIES_TICKER)
    team_code = team_name.upper()

    suffix = f"-{team_code}"
    candidates = [
        m for m in markets
        if (m.get("ticker") or "").upper().endswith(suffix)
    ]

    if not candidates:
        team_l = team_name.lower()
        for m in markets:
            title = (m.get("title") or "").lower()
            if team_l in title:
                candidates.append(m)

    if not candidates:
        raise RuntimeError(f"No markets found for team '{team_name}'")

    now = utc_now()
    future = [m for m in candidates if parse_iso(m["close_time"]) > now]
    if future:
        candidates = future

    candidates.sort(key=lambda m: parse_iso(m["close_time"]))
    chosen = candidates[0]
    return chosen["ticker"], chosen


# =============================================================================
# ESPN CLOCK
# =============================================================================

def setup_espn_clock(game_config: Dict[str, Any]):
    try:
        clock = EspnGameClock(
            yyyymmdd=game_config["espn_date"],
            team_code=game_config["espn_team"],
            opponent_code=game_config.get("espn_opponent"),
            cache_ttl_secs=10,
        )
        print_status(
            f"[{game_config['label']}] ESPN clock: "
            f"{game_config['espn_team']} vs {game_config.get('espn_opponent', '?')}"
        )
        return clock
    except Exception as e:
        print_status(f"[{game_config['label']}] ESPN clock failed: {e}")
        return None


# =============================================================================
# GAME WORKER
# =============================================================================

def run_game(game_config: Dict[str, Any], private_key, results: Dict[str, Any]):
    label = game_config["label"]
    try:
        print_status(f"\n[{label}] Initializing...")

        ticker, market = find_market_for_team(private_key, game_config["team_name"])
        print_status(f"[{label}] Market: {ticker} — {market.get('title', 'N/A')}")

        espn_clock = setup_espn_clock(game_config)

        model_fair = int(game_config["model_p_win"] * 100)
        partner_fair = int(game_config["partner_p_win"] * 100)

        preferred_side = "yes" if game_config["model_p_win"] >= 0.50 else "no"

        strategies = [
            MeanReversionStrategy(
                max_capital=float(game_config["allocation"]),
                preferred_side=preferred_side,
            ),
        ]



        print_status(
            f"[{label}] Strategy: MR ONLY | "
            f"Alloc:${game_config['allocation']:.2f} | "
            f"Preferred:{preferred_side.upper()} | "
            f"ModelFair:{int(game_config['model_p_win']*100)}c"
        )



        runner = GameRunner(
            game_label=label,
            ticker=ticker,
            market=market,
            strategies=strategies,
            private_key=private_key,
            espn_clock=espn_clock,
        )

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
    print("\n" + "=" * 80)
    print("  PRODUCTION MULTI-STRATEGY RUNNER — Feb 6, 2026")
    print("  Fixes applied: P1-P5 from Feb 5 post-mortem")
    print("=" * 80)

    if not GAMES:
        print("\n⚠  GAMES list is empty!")
        print("   Edit tonight_runner.py and fill in tonight's games before running.")
        print("   Each game needs: label, team_name, model_p_win, espn_date, espn_team")
        return 1

    print("\nGAMES:")
    for i, game in enumerate(GAMES, 1):
        print(f"  {i}. {game['label']}")
        print(f"     Model: {game['team_name']} @ {game['model_p_win']:.0%}")
        print(f"     Segment: {game.get('segment', 'N/A')}")

    print(f"\nRUN MODE: MR ONLY (equal-split bankroll)")


    print("\nFIXES ACTIVE:")
    print("  MR: stop-loss OFF")
    print("  MR: size scales to per-game allocation")
    print("  MR: last-5min directional (only preferred side + flatten wrong-side)")
    print("  P3: Market quality pre-filter (skip dead/illiquid after warmup)")
    print("  P4: Settlement-aware warnings + live R/U/T PnL display")


    # Load credentials
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()

    if not api_key or not key_path:
        print("\n✗ Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        return 1

    try:
        private_key = _load_private_key(key_path)
        print(f"\n✓ Private key loaded")
    except Exception as e:
        print(f"\n✗ Key error: {e}")
        return 1

    try:
        resp = _get(private_key, "/trade-api/v2/portfolio/balance")
        balance = int(resp.get("balance", 0)) / 100
        print(f"✓ Balance: ${balance:.2f}")

        per_game_allocation = balance / max(1, len(GAMES))

        for g in GAMES:
            g["allocation"] = per_game_allocation

        print(f"✓ Per-game allocation (equal split): ${per_game_allocation:.2f}")

    except Exception as e:
        print(f"✗ Balance check failed: {e}")
        return 1

    print("\n" + "=" * 80)
    print("Press ENTER to start (Ctrl+C to abort)...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nAborted")
        return 0

    print("=" * 80 + "\n")

    results = {}
    threads = []

    for game in GAMES:
        game_pk = _load_private_key(key_path)
        t = threading.Thread(
            target=run_game,
            args=(game, game_pk, results),
            name=game["label"],
            daemon=False,
        )
        t.start()
        threads.append(t)
        print_status(f"Started: {game['label']}")

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print_status("\n⚠ Interrupted — games continue until market close")

    # Final report
    print("\n" + "=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80 + "\n")

    total_net = 0.0

    for label, result in results.items():
        if "error" in result:
            print(f"{label}: ERROR — {result['error']}")
            continue

        print(f"{label}:")
        strategies = result.get("strategies", {})
        game_net = 0.0

        for strat_name, stats in strategies.items():
            net = stats.get("net_pnl", 0)
            game_net += net
            print(f"  {strat_name}: "
                  f"{stats.get('trades', 0)} trades "
                  f"({stats.get('wins', 0)}W-{stats.get('losses', 0)}L) | "
                  f"Locks:{stats.get('locks', 0)} Stops:{stats.get('stops', 0)} | "
                  f"Net:{net:.1f}¢ (${net/100:.2f}) | "
                  f"Fees:{stats.get('fees', 0):.1f}¢")

        print(f"  GAME TOTAL: {game_net:.1f}¢ (${game_net/100:.2f})")
        total_net += game_net

    print(f"\n{'=' * 80}")
    print(f"PORTFOLIO NET: {total_net:.1f}¢ (${total_net/100:.2f})")
    print(f"{'=' * 80}\n")

    print("Log files in ./logs/:")
    print("  *_snapshots.csv  — Market ticks + quality metrics")
    print("  *_trades.csv     — Order attempts and fills")
    print("  *_positions.csv  — Complete trade lifecycle")
    print("  *_events.csv     — Strategy decisions and warnings")
    print("  summary_*.json   — Final stats\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
