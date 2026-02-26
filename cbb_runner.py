# cbb_runner.py
# College Basketball (CBB) game runner — Rolling Average (TMA/TMA crossover) strategy
#
# Modern runner architecture (mirrors tennis_runner.py):
#   - ESPN game clock per game
#   - Market discovery via KXNCAAMBGAME series or ncaam bundle
#   - RA strategy with TMA/TMA 80/200 t=10
#   - Maker entries + exits, min_entry_price=20
#   - ExposureTracker per game
#   - Organized log dirs: kalshi-logs/kalshi/cbb/<date>/<game_label>/
#   - Cloud mode with R2 sync
#   - Hot-add game watcher from cbb_games.json
#   - Bundle-based market discovery (todays_ncaam_bundle.json)
#
# Usage:
#   python cbb_runner.py                          # uses GAMES list below
#   python cbb_runner.py --games-json cbb_games.json
#   python cbb_runner.py --bundle todays_ncaam_bundle.json
#   python cbb_runner.py --cloud                  # cloud mode with R2 sync
#
# Env vars:
#   KALSHI_ENV=PROD|DEMO
#   KALSHI_API_KEY_ID=...
#   KALSHI_PRIVATE_KEY_PATH=...

import os
import sys
import json
import time
import threading
import argparse
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import datetime as dt

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from combo_vnext import (
    _load_private_key,
    _get,
    fetch_market,
    fetch_orderbook,
    derive_prices,
    get_markets_in_series,
    parse_iso,
    utc_now,
    print_status,
)

from espn_game_clock import EspnGameClock

from production_strategies import (
    RollingAverageStrategy,
    ExposureTracker,
    GameRunner,
)

import re

# ============================================================================
# CONFIGURATION
# ============================================================================

CBB_SERIES = "KXNCAAMBGAME"

GAMES_JSON_FILE = REPO_ROOT / "cbb_games.json"

# CBB-optimal RA params: TMA/TMA 80/200 t=10
CBB_RA_OVERRIDES = {
    "strategy_type": "tma_tma",
    "fast_window": 80,
    "slow_window": 200,
    "threshold_cents": 10,
    "spread_max": 6,
    "max_positions": 1,
    "entry_frac": 0.50,
    "min_qty": 3,
    "max_order_qty": 200,
    "profit_defense_activate_cents": 8.0,
    "profit_defense_giveback_frac": 0.40,
    "profit_defense_min_keep_cents": 2.0,
    "flatten_before_close_secs": 120,
    "directional_close_secs": 300,
}

# Today's games — fallback if cbb_games.json not found
# team_name must match the Kalshi ticker suffix (e.g. "DUKE", "UNC", "KU")
# espn_date = YYYYMMDD, espn_team/espn_opponent = ESPN abbreviations
GAMES = [
    # {"label": "Duke at UNC", "team_name": "DUKE", "espn_date": "20260225", "espn_team": "DUKE", "espn_opponent": "UNC", "model_p_win": 0.55},
]


# ============================================================================
# LOG DIRECTORY
# ============================================================================

LOG_ROOT = Path(os.getenv("KALSHI_LOG_ROOT", "kalshi-logs"))


def safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)


def build_game_log_dir(game_label: str, game_date: str) -> Path:
    return (
        LOG_ROOT
        / "kalshi"
        / "cbb"
        / "ncaam"
        / game_date
        / safe_name(game_label)
    )


# ============================================================================
# MARKET DISCOVERY
# ============================================================================

def find_ml_ticker_for_team(private_key, team_name: str,
                            game_key: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Find the ML (moneyline) market for a team by ticker suffix match.
    Searches KXNCAAMBGAME series for an active market ending in -<TEAM_CODE>.
    If game_key is provided, filters candidates whose ticker contains the game_key
    substring (e.g. "26FEB25DUKEUNC") to disambiguate teams with the same code.
    """
    markets = get_markets_in_series(private_key, CBB_SERIES)
    code = team_name.upper()
    suffix = f"-{code}"

    candidates = [
        m for m in markets
        if (m.get("ticker") or "").upper().endswith(suffix)
    ]

    if not candidates:
        # Fallback: search titles
        code_lower = team_name.lower()
        for m in markets:
            title = (m.get("title") or "").lower()
            if code_lower in title:
                candidates.append(m)

    # Filter by game_key if provided (disambiguates e.g. same team code in different games)
    if game_key and candidates:
        gk_upper = game_key.upper()
        filtered = [m for m in candidates if gk_upper in (m.get("ticker") or "").upper()]
        if filtered:
            candidates = filtered

    if not candidates:
        raise RuntimeError(f"No CBB market found for team '{team_name}'" +
                           (f" with game_key '{game_key}'" if game_key else ""))

    # Prefer future markets
    now = utc_now()
    future = [m for m in candidates if parse_iso(m["close_time"]) > now]
    if future:
        candidates = future

    # Sort by close time (nearest first)
    candidates.sort(key=lambda m: parse_iso(m["close_time"]))
    chosen = candidates[0]
    return chosen["ticker"], chosen


def find_ml_ticker_from_bundle(bundle: Dict[str, Any], team_code: str,
                               game_key: Optional[str] = None) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Resolve ML ticker from an ncaam bundle JSON.
    Returns (ticker, game_info, bundle_game).
    If game_key is provided, looks up that specific game first for disambiguation.
    """
    games = bundle.get("games", {})
    code = team_code.upper()

    # Direct lookup by game_key if provided
    if game_key and game_key in games:
        game = games[game_key]
        outcomes = game.get("ml_outcomes", {})
        if code in outcomes:
            return outcomes[code], game, game

    # Fallback: scan all games
    for gk, game in games.items():
        outcomes = game.get("ml_outcomes", {})
        if code in outcomes:
            return outcomes[code], game, game

    raise RuntimeError(f"Team code '{team_code}' not found in ncaam bundle" +
                       (f" (game_key='{game_key}')" if game_key else ""))


# ============================================================================
# ESPN CLOCK SETUP
# ============================================================================

def setup_espn_clock(game_config: Dict[str, Any]) -> Optional[EspnGameClock]:
    """Create an ESPN game clock for a CBB game."""
    espn_date = game_config.get("espn_date")
    espn_team = game_config.get("espn_team")
    if not espn_date or not espn_team:
        print_status(f"[{game_config['label']}] No ESPN config — clock disabled")
        return None

    try:
        clock = EspnGameClock(
            yyyymmdd=espn_date,
            team_code=espn_team,
            opponent_code=game_config.get("espn_opponent"),
            cache_ttl_secs=10,
        )
        print_status(
            f"[{game_config['label']}] ESPN clock: "
            f"{espn_team} vs {game_config.get('espn_opponent', '?')}"
        )
        return clock
    except Exception as e:
        print_status(f"[{game_config['label']}] ESPN clock failed: {e}")
        return None


# ============================================================================
# PER-GAME WORKER
# ============================================================================

def run_game(game_config: Dict[str, Any], private_key, results: Dict[str, Any],
             bundle: Optional[Dict[str, Any]] = None, uploader=None):
    label = game_config["label"]
    try:
        print_status(f"\n[{label}] Initializing...")

        team_name = game_config["team_name"]
        game_key = game_config.get("game_key")

        # Discover ML ticker
        if bundle:
            ml_ticker, game_info, _ = find_ml_ticker_from_bundle(bundle, team_name, game_key)
            ml_market = fetch_market(private_key, ml_ticker)
        else:
            ml_ticker, ml_market = find_ml_ticker_for_team(private_key, team_name, game_key)

        print_status(f"[{label}] ML Market: {ml_ticker} — {ml_market.get('title', 'N/A')}")
        print_status(f"[{label}] Close time: {ml_market.get('close_time', 'N/A')}")

        # ESPN clock
        espn_clock = setup_espn_clock(game_config)

        preferred_side = "yes" if float(game_config.get("model_p_win", 0.50)) >= 0.50 else "no"

        allocation = float(game_config["allocation"])

        # Shared exposure tracker for this game
        exposure = ExposureTracker(max_exposure_dollars=allocation)

        # Log directory
        # Use Eastern time for date so evening games don't roll to next day
        eastern_offset = dt.timezone(dt.timedelta(hours=-5))
        game_date = dt.datetime.now(eastern_offset).strftime("%Y-%m-%d")
        log_dir = build_game_log_dir(label, game_date)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Restore existing CSVs from R2 so GameRunner appends instead of overwriting
        if uploader:
            try:
                uploader.download_match_logs(log_dir)
            except Exception as e:
                print_status(f"[{label}] Warning: failed to restore logs from R2: {e}")

        # Load strategy config overrides
        strategy_config_overrides = {}
        try:
            cfg_path = REPO_ROOT / "strategy_config.json"
            if cfg_path.exists():
                strategy_config_overrides = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            print_status(f"[{label}] Warning: failed to load strategy_config.json: {e}")

        # Strategy: RA-only (Rolling Average / TMA crossover)
        strategies = [
            RollingAverageStrategy(
                max_capital=allocation,
                preferred_side=preferred_side,
                exposure_tracker=exposure,
            ),
        ]

        # Apply CBB RA defaults, then strategy_config.json overrides on top
        for strat in strategies:
            strat.update_params(CBB_RA_OVERRIDES)
            overrides = strategy_config_overrides.get(strat.name, {})
            if overrides:
                strat.update_params(overrides)

        st = strat.params.get("strategy_type", "tma_tma")
        fw = strat.params.get("fast_window", 80)
        sw = strat.params.get("slow_window", 200)
        tc = strat.params.get("threshold_cents", 10)
        print_status(
            f"[{label}] RA:{st} {fw}/{sw} t={tc} | ${allocation:.2f} | "
            f"Preferred:{preferred_side.upper()} | Maker entries + MP20 | "
            f"ESPN:{'ON' if espn_clock else 'OFF'}"
        )

        # Run GameRunner (with ESPN clock for CBB)
        runner = GameRunner(
            game_label=label,
            ticker=ml_ticker,
            market=ml_market,
            strategies=strategies,
            private_key=private_key,
            espn_clock=espn_clock,
            log_dir=str(log_dir),
            maker_entries=True,
            maker_exits=True,
            min_entry_price=20,
            mq_params={
                "std_min": 1.5,
                "spread_max": 8.0,
                "range_min": 3.0,
            },
            base_strategy_overrides={"rolling_avg": CBB_RA_OVERRIDES},
        )

        summary = runner.run()
        results[label] = summary
        print_status(f"[{label}] Complete")

    except Exception as e:
        print_status(f"[{label}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[label] = {"error": str(e)}


# ============================================================================
# CLOUD MODE HELPERS (reuse from tonight_runner_cloud.py)
# ============================================================================

CBB_GAMES_R2_KEY = "kalshi/config/cbb_games.json"


def _load_cloud_helpers():
    """Lazy import cloud helpers to avoid import errors when running locally without boto3."""
    from tonight_runner_cloud import (
        setup_workdir,
        R2Uploader,
        StopFlag,
        uploader_loop,
        load_private_key_for_worker,
    )
    return setup_workdir, R2Uploader, StopFlag, uploader_loop, load_private_key_for_worker


def _r2_download_cbb_games(uploader) -> bool:
    """Download cbb_games.json from R2 if changed. Returns True if updated."""
    if not uploader or not uploader.enabled():
        return False
    return uploader._download_r2_file(CBB_GAMES_R2_KEY, Path("cbb_games.json"), "CBB Games")


def load_games_from_env() -> Optional[List[Dict[str, Any]]]:
    """
    Load games from env vars (cloud mode).
    Priority: GAMES_JSON (inline) -> GAMES_JSON_PATH (file path)
    """
    inline = os.getenv("GAMES_JSON", "").strip()
    if inline:
        try:
            data = json.loads(inline)
            if isinstance(data, list):
                return data
        except Exception as e:
            print_status(f"[GAMES] Invalid GAMES_JSON: {e}")

    path = os.getenv("GAMES_JSON_PATH", "").strip()
    if path:
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception as e:
                print_status(f"[GAMES] Invalid JSON file {path}: {e}")
        else:
            print_status(f"[GAMES] GAMES_JSON_PATH not found: {path}")

    return None


def load_games_from_file() -> Optional[List[Dict[str, Any]]]:
    """Load games from cbb_games.json (local or R2-synced).
    Check cwd first (R2-synced copy from uploader), then repo root."""
    for path in [Path("cbb_games.json"), GAMES_JSON_FILE]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return data
            except Exception:
                pass
    return None


# ============================================================================
# GAME WATCHER — hot-add new games mid-session
# ============================================================================

def game_watcher_loop(running_keys: set, running_keys_lock: threading.Lock,
                      results: Dict[str, Any], threads: List[threading.Thread],
                      private_key_path: str, bundle: Optional[Dict[str, Any]],
                      cloud_mode: bool, balance_per_game: float,
                      stop_event: Optional[Any] = None, uploader=None):
    """
    Periodically re-reads cbb_games.json and launches threads for new games.
    Runs as a daemon thread.
    """
    check_interval = 60  # seconds
    while True:
        if stop_event and stop_event.is_set():
            break
        time.sleep(check_interval)

        try:
            # Re-download from R2 if available (cloud mode hot-add)
            if uploader:
                _r2_download_cbb_games(uploader)

            new_games = load_games_from_file()
            if not new_games:
                continue

            with running_keys_lock:
                for g in new_games:
                    gk = g.get("game_key", g.get("label", ""))
                    if gk in running_keys:
                        continue

                    # New game found — launch thread
                    running_keys.add(gk)
                    g["allocation"] = float(g.get("allocation") or balance_per_game)

                    try:
                        game_pk = _load_private_key(private_key_path)
                    except Exception as e:
                        print_status(f"[WATCHER] Key error for {g.get('label', '?')}: {e}")
                        continue

                    t = threading.Thread(
                        target=run_game,
                        args=(g, game_pk, results, bundle, uploader),
                        name=g.get("label", gk),
                        daemon=False,
                    )
                    t.start()
                    threads.append(t)
                    print_status(f"[WATCHER] Hot-added: {g.get('label', gk)} (alloc=${g['allocation']:.2f})")

        except Exception as e:
            print_status(f"[WATCHER] Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="CBB game runner for Kalshi (RA strategy, TMA/TMA 80/200 t=10)")
    ap.add_argument("--bundle", default="", help="Path to todays_ncaam_bundle.json (optional)")
    ap.add_argument("--games-json", default="", help="Path to games config JSON (optional)")
    ap.add_argument("--cloud", action="store_true", help="Cloud/worker mode: R2 sync, no interactive prompt")
    args = ap.parse_args()

    cloud_mode = args.cloud

    # --- Cloud mode setup ---
    if cloud_mode:
        setup_workdir, R2Uploader, StopFlag, uploader_loop_fn, load_private_key_for_worker = _load_cloud_helpers()
        setup_workdir()

    print("\n" + "=" * 80)
    print("  CBB RUNNER — RA-ONLY (TMA/TMA 80/200 t=10) + MAKER ENTRIES + MP20")
    print("  ESPN game clock enabled per game")
    print("  Strategy: Rolling Average sniper (~1 trade/game)")
    if cloud_mode:
        print("  MODE: CLOUD (R2 sync enabled)")
    print("=" * 80)

    # --- Key loading ---
    if cloud_mode:
        api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
        if not api_key:
            print_status("Set KALSHI_API_KEY_ID in environment.")
            return 1
        try:
            private_key = load_private_key_for_worker()
            print_status("Private key loaded (cloud)")
        except Exception as e:
            print_status(f"Key error: {e}")
            return 1
    else:
        api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if not api_key or not key_path:
            print_status("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
            return 1
        try:
            private_key = _load_private_key(key_path)
            print_status("Private key loaded")
        except Exception as e:
            print_status(f"Key error: {e}")
            return 1

    # --- Load games config ---
    games = None

    # Cloud mode: download cbb_games.json from R2, then try env vars
    if cloud_mode:
        try:
            # Temporary uploader just for initial config download
            _init_uploader = R2Uploader(local_logs_dir=LOG_ROOT)
            _r2_download_cbb_games(_init_uploader)
            _init_uploader.download_config(Path("strategy_config.json"))
        except Exception as e:
            print_status(f"[R2] Initial config download failed: {e}")

        games = load_games_from_env()
        if games:
            print_status(f"Loaded {len(games)} games from env")

    # CLI --games-json flag
    if games is None and args.games_json:
        p = Path(args.games_json)
        if p.exists():
            try:
                games = json.loads(p.read_text(encoding="utf-8"))
                print_status(f"Loaded {len(games)} games from {args.games_json}")
            except Exception as e:
                print_status(f"Invalid games JSON: {e}")
                return 1
        else:
            print_status(f"Games JSON not found: {args.games_json}")
            return 1

    # Try cbb_games.json file
    if games is None:
        games = load_games_from_file()
        if games:
            print_status(f"Loaded {len(games)} games from cbb_games.json")

    # Fallback to hardcoded GAMES
    if games is None:
        games = list(GAMES)
        if games:
            print_status(f"Using {len(games)} hardcoded games")

    # Load bundle if provided
    bundle = None
    bundle_path = args.bundle or os.getenv("CBB_BUNDLE_PATH", "").strip()
    if bundle_path:
        bp = Path(bundle_path)
        if bp.exists():
            try:
                bundle = json.loads(bp.read_text(encoding="utf-8"))
                print_status(f"Loaded ncaam bundle from {bundle_path}")
            except Exception as e:
                print_status(f"Invalid bundle JSON: {e}")

    if not games:
        print_status("No games configured!")
        print_status("  Add games to GAMES list in cbb_runner.py, or")
        print_status("  Create cbb_games.json, or")
        print_status("  Pass --games-json <path>")
        return 1

    # Balance check + allocation
    try:
        resp = _get(private_key, "/trade-api/v2/portfolio/balance")
        balance_cents = int(resp.get("balance", 0))
        balance = balance_cents / 100.0
        print_status(f"Balance: ${balance:.2f}")

        per_game_allocation = balance / max(1, len(games))
        for g in games:
            g["allocation"] = float(g.get("allocation") or per_game_allocation)

        print_status(f"Default per-game allocation: ${per_game_allocation:.2f}")
    except Exception as e:
        print_status(f"Balance check failed: {e}")
        return 1

    print_status("\nGAMES:")
    for i, g in enumerate(games, 1):
        print_status(
            f"  {i}. {g['label']} | team={g['team_name']} | "
            f"ESPN={g.get('espn_team', '?')} vs {g.get('espn_opponent', '?')} | "
            f"p_win={g.get('model_p_win', 0.50):.0%} | alloc=${g['allocation']:.2f}"
        )

    # --- Interactive prompt (local only) ---
    if not cloud_mode:
        print("\nPress ENTER to start (Ctrl+C to abort)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nAborted")
            return 0

    print("=" * 80 + "\n")

    # --- R2 uploader (cloud mode) ---
    stop = None
    uploader_thread = None
    uploader = None
    if cloud_mode:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        uploader = R2Uploader(local_logs_dir=LOG_ROOT)
        stop = StopFlag()

        def _handle_sig(_sig, _frame):
            print_status("Received shutdown signal. Stopping...")
            stop.set()

        signal.signal(signal.SIGTERM, _handle_sig)
        signal.signal(signal.SIGINT, _handle_sig)

        def _cbb_uploader_loop(stop_flag, upl, interval):
            """CBB-specific uploader loop: syncs cbb_games.json instead of tennis_matches.json."""
            print_status(f"[R2] CBB uploader thread started (interval={interval}s)")
            while not stop_flag.is_set():
                try:
                    upl.download_config(Path("strategy_config.json"))
                    _r2_download_cbb_games(upl)
                    upl.write_index()
                    upl.upload_changed()
                except Exception as e:
                    print_status(f"[R2] Uploader error: {e}")
                time.sleep(max(5, int(interval)))
            # final flush
            try:
                upl.write_index({"final_flush": True})
                upl.upload_changed(max_files=5000)
            except Exception:
                pass
            print_status("[R2] CBB uploader thread stopped")

        uploader_thread = threading.Thread(
            target=_cbb_uploader_loop,
            args=(stop, uploader, int(os.getenv("R2_SYNC_INTERVAL_SECS") or "20")),
            name="r2_uploader",
            daemon=True,
        )
        uploader_thread.start()

    # --- Launch game threads ---
    results: Dict[str, Any] = {}
    threads: List[threading.Thread] = []

    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()

    # Track running game keys for hot-add watcher
    running_keys: set = set()
    running_keys_lock = threading.Lock()

    for game in games:
        gk = game.get("game_key", game.get("label", ""))
        running_keys.add(gk)

        game_pk = _load_private_key(key_path)
        t = threading.Thread(
            target=run_game,
            args=(game, game_pk, results, bundle, uploader),
            name=game["label"],
            daemon=False,
        )
        t.start()
        threads.append(t)
        print_status(f"Started: {game['label']}")

    # --- Start game watcher (hot-add new games from cbb_games.json) ---
    watcher_thread = threading.Thread(
        target=game_watcher_loop,
        args=(running_keys, running_keys_lock, results, threads,
              key_path, bundle, cloud_mode, per_game_allocation, stop, uploader),
        name="game_watcher",
        daemon=True,
    )
    watcher_thread.start()
    print_status("[WATCHER] Game watcher started (checks every 60s for new games)")

    # --- Wait for threads ---
    if cloud_mode and stop:
        while not stop.is_set():
            alive = any(t.is_alive() for t in threads)
            if not alive:
                break
            time.sleep(2)
        if stop.is_set():
            print_status("Stop flag set. Waiting briefly for threads to wrap up...")
            for t in threads:
                t.join(timeout=5)
    else:
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print_status("\nInterrupted — games continue until market close")

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
            net = float(stats.get("net_pnl", 0))
            game_net += net
            print(
                f"  {strat_name}: "
                f"{stats.get('trades', 0)} trades "
                f"({stats.get('wins', 0)}W-{stats.get('losses', 0)}L) | "
                f"Locks:{stats.get('locks', 0)} Stops:{stats.get('stops', 0)} | "
                f"Net:{net:.1f}c (${net/100:.2f}) | "
                f"Fees:{float(stats.get('fees', 0)):.1f}c"
            )

        print(f"  GAME TOTAL: {game_net:.1f}c (${game_net/100:.2f})")
        total_net += game_net

    print(f"\n{'=' * 80}")
    print(f"PORTFOLIO NET: {total_net:.1f}c (${total_net/100:.2f})")
    print(f"{'=' * 80}\n")

    # Save aggregate summary
    try:
        ts = utc_now().strftime("%Y%m%d_%H%M%S")
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        agg_path = LOG_ROOT / f"cbb_aggregate_summary_{ts}.json"
        agg = {
            "sport": "cbb",
            "strategy": "rolling_avg",
            "ra_params": "tma_tma_80_200_t10",
            "updated_utc": utc_now().isoformat(),
            "portfolio_net_cents": total_net,
            "results": results,
        }
        agg_path.write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
        print_status(f"Summary: {agg_path}")
    except Exception as e:
        print_status(f"Failed to write summary: {e}")

    # --- Stop R2 uploader + final flush ---
    if cloud_mode and stop:
        stop.set()
        try:
            uploader_thread.join(timeout=20)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
