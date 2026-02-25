# tennis_runner.py
# ATP Tennis match runner — Rolling Average (MA crossover) strategy
#
# Differences from CBB runner:
#   - No ESPN clock (tennis doesn't have reliable free live score APIs with win prob)
#   - Uses Kalshi close_time as sole clock source
#   - Market discovery via KXATPMATCH series
#   - Matches configured with player codes instead of team codes
#   - RA-only strategy (no MR) — data collection phase
#
# Usage:
#   python tennis_runner.py                          # uses MATCHES list below
#   python tennis_runner.py --matches-json tennis_matches.json
#   python tennis_runner.py --cloud                  # cloud mode with R2 sync
#
# Env vars (same as CBB):
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

from production_strategies import (
    RollingAverageStrategy,
    ExposureTracker,
    GameRunner,
)

import re

# ============================================================================
# CONFIGURATION
# ============================================================================

TENNIS_SERIES = ["KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH"]

MATCHES_JSON_FILE = REPO_ROOT / "tennis_matches.json"

# Tennis-optimal RA params from grid search (EMA/TMA 40/160 t=12)
TENNIS_RA_OVERRIDES = {
    "strategy_type": "ema_tma",
    "fast_window": 40,
    "slow_window": 160,
    "threshold_cents": 12,
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

# Today's matches — fallback if tennis_matches.json not found
# player_code must match the Kalshi ticker suffix (e.g. "POP", "RUB", "MEN")
MATCHES = [
    {"label": "Rublev vs Humbert", "player_code": "RUB", "opponent_code": "HUM", "match_key": "26FEB25RUBHUM", "model_p_win": 0.50},
    {"label": "Blanchet vs Broom", "player_code": "BLA", "opponent_code": "BRO", "match_key": "26FEB25BLABRO", "series": "KXATPCHALLENGERMATCH", "model_p_win": 0.50},
    {"label": "Budkov Kjaer vs Nes", "player_code": "BUD", "opponent_code": "NES", "match_key": "26FEB25BUDNES", "series": "KXATPCHALLENGERMATCH", "model_p_win": 0.50},
    {"label": "Martineau vs Torra", "player_code": "MAR", "opponent_code": "TOR", "match_key": "26FEB25MARTOR", "series": "KXATPCHALLENGERMATCH", "model_p_win": 0.50},
    {"label": "Mensik vs Popyrin", "player_code": "MEN", "opponent_code": "POP", "match_key": "26FEB25MENPOP", "model_p_win": 0.50},
    {"label": "Ugo vs Hanfmann", "player_code": "UGO", "opponent_code": "HAN", "match_key": "26FEB24UGOHAN", "model_p_win": 0.50},
    {"label": "Rindknecht vs Draper", "player_code": "RIN", "opponent_code": "DRA", "match_key": "26FEB25RINDRA", "model_p_win": 0.50},
    {"label": "Mrva vs Moro", "player_code": "MRV", "opponent_code": "MMO", "match_key": "26FEB25MRVMMO", "series": "KXATPCHALLENGERMATCH", "model_p_win": 0.50},
    {"label": "Koueladjol vs Droguet", "player_code": "KOU", "opponent_code": "DRO", "match_key": "26FEB22KOUDRO", "series": "KXATPCHALLENGERMATCH", "model_p_win": 0.50},
]


# ============================================================================
# LOG DIRECTORY
# ============================================================================

LOG_ROOT = Path(os.getenv("KALSHI_LOG_ROOT", "kalshi-logs"))


def safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)


def build_match_log_dir(match_label: str, match_date: str,
                        series: str = "atp") -> Path:
    # Derive subfolder from series: KXWTAMATCH → wta, KXATPMATCH → atp
    if "WTA" in series.upper():
        subfolder = "wta"
    else:
        subfolder = "atp"
    return (
        LOG_ROOT
        / "kalshi"
        / "tennis"
        / subfolder
        / match_date
        / safe_name(match_label)
    )


# ============================================================================
# MARKET DISCOVERY
# ============================================================================

def find_ml_ticker_for_player(private_key, player_code: str, match_key: Optional[str] = None,
                              series_override: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Find the ML (match winner) market for a player by ticker suffix match.
    Searches all TENNIS_SERIES for an active market ending in -<PLAYER_CODE>.
    If match_key is provided, filters candidates whose ticker contains the match_key
    substring (e.g. "26FEB22NAVKOP") to disambiguate players with the same code.
    """
    series_list = [series_override] if series_override else TENNIS_SERIES
    markets = []
    for s in series_list:
        markets.extend(get_markets_in_series(private_key, s))
    code = player_code.upper()
    suffix = f"-{code}"

    candidates = [
        m for m in markets
        if (m.get("ticker") or "").upper().endswith(suffix)
    ]

    if not candidates:
        # Fallback: search titles
        code_lower = player_code.lower()
        for m in markets:
            title = (m.get("title") or "").lower()
            if code_lower in title:
                candidates.append(m)

    # Filter by match_key if provided (disambiguates e.g. NAV in Navone vs Nava)
    if match_key and candidates:
        mk_upper = match_key.upper()
        filtered = [m for m in candidates if mk_upper in (m.get("ticker") or "").upper()]
        if filtered:
            candidates = filtered

    if not candidates:
        raise RuntimeError(f"No ATP match market found for player code '{player_code}'" +
                           (f" with match_key '{match_key}'" if match_key else ""))

    # Prefer future markets
    now = utc_now()
    future = [m for m in candidates if parse_iso(m["close_time"]) > now]
    if future:
        candidates = future

    # Sort by close time (nearest first)
    candidates.sort(key=lambda m: parse_iso(m["close_time"]))
    chosen = candidates[0]
    return chosen["ticker"], chosen


def find_ml_ticker_from_bundle(bundle: Dict[str, Any], player_code: str, match_key: Optional[str] = None) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Resolve ML ticker from a tennis bundle JSON.
    Returns (ticker, match_info, bundle_match).
    If match_key is provided, looks up that specific match first for disambiguation.
    """
    matches = bundle.get("matches", {})
    code = player_code.upper()

    # Direct lookup by match_key if provided
    if match_key and match_key in matches:
        match = matches[match_key]
        outcomes = match.get("ml_outcomes", {})
        if code in outcomes:
            return outcomes[code], match, match

    # Fallback: scan all matches
    for mk, match in matches.items():
        outcomes = match.get("ml_outcomes", {})
        if code in outcomes:
            return outcomes[code], match, match

    raise RuntimeError(f"Player code '{player_code}' not found in tennis bundle" +
                       (f" (match_key='{match_key}')" if match_key else ""))


# ============================================================================
# PER-MATCH WORKER
# ============================================================================

def run_match(match_config: Dict[str, Any], private_key, results: Dict[str, Any],
              bundle: Optional[Dict[str, Any]] = None, uploader=None):
    label = match_config["label"]
    try:
        print_status(f"\n[{label}] Initializing...")

        player_code = match_config["player_code"]
        match_key = match_config.get("match_key")
        series_override = match_config.get("series")

        # Discover ML ticker
        if bundle and not series_override:
            ml_ticker, match_info, _ = find_ml_ticker_from_bundle(bundle, player_code, match_key)
            ml_market = fetch_market(private_key, ml_ticker)
        else:
            ml_ticker, ml_market = find_ml_ticker_for_player(private_key, player_code, match_key, series_override)

        print_status(f"[{label}] ML Market: {ml_ticker} — {ml_market.get('title', 'N/A')}")
        print_status(f"[{label}] Close time: {ml_market.get('close_time', 'N/A')}")

        preferred_side = "yes" if float(match_config.get("model_p_win", 0.50)) >= 0.50 else "no"

        allocation = float(match_config["allocation"])

        # Shared exposure tracker for this match
        exposure = ExposureTracker(max_exposure_dollars=allocation)

        # Log directory — derive series from ticker for correct subfolder
        match_date = utc_now().strftime("%Y-%m-%d")
        ticker_series = ml_ticker.split("-")[0] if ml_ticker else (series_override or "KXATPMATCH")
        log_dir = build_match_log_dir(label, match_date, series=ticker_series)
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

        # Strategy: RA-only (Rolling Average / MA crossover)
        # 100% allocation to RA — no MR for this test run
        strategies = [
            RollingAverageStrategy(
                max_capital=allocation,
                preferred_side=preferred_side,
                exposure_tracker=exposure,
            ),
        ]

        # Apply tennis RA defaults, then strategy_config.json overrides on top
        for strat in strategies:
            strat.update_params(TENNIS_RA_OVERRIDES)
            overrides = strategy_config_overrides.get(strat.name, {})
            if overrides:
                strat.update_params(overrides)

        st = strat.params.get("strategy_type", "ema_tma")
        fw = strat.params.get("fast_window", 40)
        sw = strat.params.get("slow_window", 160)
        tc = strat.params.get("threshold_cents", 12)
        print_status(
            f"[{label}] RA:{st} {fw}/{sw} t={tc} | ${allocation:.2f} | "
            f"Preferred:{preferred_side.upper()} | Maker entries + MP20"
        )

        # Run GameRunner (no ESPN clock — tennis uses Kalshi close time only)
        runner = GameRunner(
            game_label=label,
            ticker=ml_ticker,
            market=ml_market,
            strategies=strategies,
            private_key=private_key,
            espn_clock=None,      # No live clock for tennis
            log_dir=str(log_dir),
            maker_entries=True,
            maker_exits=True,
            min_entry_price=20,
            mq_params={
                "std_min": 1.5,      # tennis has flat holds between points
                "spread_max": 10.0,  # tennis books are naturally thinner
                "range_min": 3.0,    # smaller range still valid
            },
            base_strategy_overrides={"rolling_avg": TENNIS_RA_OVERRIDES},
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


def load_matches_from_env() -> Optional[List[Dict[str, Any]]]:
    """
    Load matches from env vars (cloud mode).
    Priority: MATCHES_JSON (inline) -> MATCHES_JSON_PATH (file path)
    """
    inline = os.getenv("MATCHES_JSON", "").strip()
    if inline:
        try:
            data = json.loads(inline)
            if isinstance(data, list):
                return data
        except Exception as e:
            print_status(f"[MATCHES] Invalid MATCHES_JSON: {e}")

    path = os.getenv("MATCHES_JSON_PATH", "").strip()
    if path:
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception as e:
                print_status(f"[MATCHES] Invalid JSON file {path}: {e}")
        else:
            print_status(f"[MATCHES] MATCHES_JSON_PATH not found: {path}")

    return None


def load_matches_from_file() -> Optional[List[Dict[str, Any]]]:
    """Load matches from tennis_matches.json (local or R2-synced).
    Check cwd first (R2-synced copy from uploader), then repo root."""
    for path in [Path("tennis_matches.json"), MATCHES_JSON_FILE]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return data
            except Exception:
                pass
    return None


# ============================================================================
# MATCH WATCHER — hot-add new matches mid-session
# ============================================================================

def match_watcher_loop(running_keys: set, running_keys_lock: threading.Lock,
                       results: Dict[str, Any], threads: List[threading.Thread],
                       private_key_path: str, bundle: Optional[Dict[str, Any]],
                       cloud_mode: bool, balance_per_match: float,
                       stop_event: Optional[Any] = None, uploader=None):
    """
    Periodically re-reads tennis_matches.json and launches threads for new matches.
    Runs in main thread or as a daemon thread.
    """
    check_interval = 60  # seconds
    while True:
        if stop_event and stop_event.is_set():
            break
        time.sleep(check_interval)

        try:
            new_matches = load_matches_from_file()
            if not new_matches:
                continue

            with running_keys_lock:
                for m in new_matches:
                    mk = m.get("match_key", m.get("label", ""))
                    if mk in running_keys:
                        continue

                    # New match found — launch thread
                    running_keys.add(mk)
                    m["allocation"] = float(m.get("allocation") or balance_per_match)

                    try:
                        match_pk = _load_private_key(private_key_path)
                    except Exception as e:
                        print_status(f"[WATCHER] Key error for {m.get('label', '?')}: {e}")
                        continue

                    t = threading.Thread(
                        target=run_match,
                        args=(m, match_pk, results, bundle, uploader),
                        name=m.get("label", mk),
                        daemon=False,
                    )
                    t.start()
                    threads.append(t)
                    print_status(f"[WATCHER] Hot-added: {m.get('label', mk)} (alloc=${m['allocation']:.2f})")

        except Exception as e:
            print_status(f"[WATCHER] Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="ATP Tennis match runner for Kalshi (RA strategy)")
    ap.add_argument("--bundle", default="", help="Path to todays_tennis_bundle.json (optional)")
    ap.add_argument("--matches-json", default="", help="Path to matches config JSON (optional)")
    ap.add_argument("--cloud", action="store_true", help="Cloud/worker mode: R2 sync, no interactive prompt")
    args = ap.parse_args()

    cloud_mode = args.cloud

    # --- Cloud mode setup ---
    if cloud_mode:
        setup_workdir, R2Uploader, StopFlag, uploader_loop_fn, load_private_key_for_worker = _load_cloud_helpers()
        setup_workdir()

    print("\n" + "=" * 80)
    print("  ATP TENNIS RUNNER — RA-ONLY (EMA/TMA crossover) + MAKER ENTRIES + MP20")
    print("  No ESPN clock — using Kalshi close time only")
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

    # --- Load matches config ---
    matches = None

    # Cloud mode: try env vars first
    if cloud_mode:
        matches = load_matches_from_env()
        if matches:
            print_status(f"Loaded {len(matches)} matches from env")

    # CLI --matches-json flag
    if matches is None and args.matches_json:
        p = Path(args.matches_json)
        if p.exists():
            try:
                matches = json.loads(p.read_text(encoding="utf-8"))
                print_status(f"Loaded {len(matches)} matches from {args.matches_json}")
            except Exception as e:
                print_status(f"Invalid matches JSON: {e}")
                return 1
        else:
            print_status(f"Matches JSON not found: {args.matches_json}")
            return 1

    # Try tennis_matches.json file
    if matches is None:
        matches = load_matches_from_file()
        if matches:
            print_status(f"Loaded {len(matches)} matches from tennis_matches.json")

    # Fallback to hardcoded MATCHES
    if matches is None:
        matches = list(MATCHES)
        print_status(f"Using {len(matches)} hardcoded matches")

    # Load bundle if provided
    bundle = None
    bundle_path = args.bundle or os.getenv("TENNIS_BUNDLE_PATH", "").strip()
    if bundle_path:
        bp = Path(bundle_path)
        if bp.exists():
            try:
                bundle = json.loads(bp.read_text(encoding="utf-8"))
                print_status(f"Loaded tennis bundle from {bundle_path}")
            except Exception as e:
                print_status(f"Invalid bundle JSON: {e}")

    if not matches:
        print_status("No matches configured!")
        return 1

    # Balance check + allocation
    try:
        resp = _get(private_key, "/trade-api/v2/portfolio/balance")
        balance_cents = int(resp.get("balance", 0))
        balance = balance_cents / 100.0
        print_status(f"Balance: ${balance:.2f}")

        per_match_allocation = balance / max(1, len(matches))
        for m in matches:
            m["allocation"] = float(m.get("allocation") or per_match_allocation)

        print_status(f"Default per-match allocation: ${per_match_allocation:.2f}")
    except Exception as e:
        print_status(f"Balance check failed: {e}")
        return 1

    print_status("\nMATCHES:")
    for i, m in enumerate(matches, 1):
        print_status(
            f"  {i}. {m['label']} | player={m['player_code']} vs {m.get('opponent_code', '?')} | "
            f"p_win={m.get('model_p_win', 0.50):.0%} | alloc=${m['allocation']:.2f}"
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

        uploader_thread = threading.Thread(
            target=uploader_loop_fn,
            args=(stop, uploader, int(os.getenv("R2_SYNC_INTERVAL_SECS") or "20")),
            name="r2_uploader",
            daemon=True,
        )
        uploader_thread.start()

    # --- Launch match threads ---
    results: Dict[str, Any] = {}
    threads: List[threading.Thread] = []

    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()

    # Track running match keys for hot-add watcher
    running_keys: set = set()
    running_keys_lock = threading.Lock()

    for match in matches:
        mk = match.get("match_key", match.get("label", ""))
        running_keys.add(mk)

        match_pk = _load_private_key(key_path)
        t = threading.Thread(
            target=run_match,
            args=(match, match_pk, results, bundle, uploader),
            name=match["label"],
            daemon=False,
        )
        t.start()
        threads.append(t)
        print_status(f"Started: {match['label']}")

    # --- Start match watcher (hot-add new matches from tennis_matches.json) ---
    watcher_thread = threading.Thread(
        target=match_watcher_loop,
        args=(running_keys, running_keys_lock, results, threads,
              key_path, bundle, cloud_mode, per_match_allocation, stop, uploader),
        name="match_watcher",
        daemon=True,
    )
    watcher_thread.start()
    print_status("[WATCHER] Match watcher started (checks every 60s for new matches)")

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
            print_status("\nInterrupted — matches continue until market close")

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
        match_net = 0.0

        for strat_name, stats in strategies.items():
            net = float(stats.get("net_pnl", 0))
            match_net += net
            print(
                f"  {strat_name}: "
                f"{stats.get('trades', 0)} trades "
                f"({stats.get('wins', 0)}W-{stats.get('losses', 0)}L) | "
                f"Locks:{stats.get('locks', 0)} Stops:{stats.get('stops', 0)} | "
                f"Net:{net:.1f}c (${net/100:.2f}) | "
                f"Fees:{float(stats.get('fees', 0)):.1f}c"
            )

        print(f"  MATCH TOTAL: {match_net:.1f}c (${match_net/100:.2f})")
        total_net += match_net

    print(f"\n{'=' * 80}")
    print(f"PORTFOLIO NET: {total_net:.1f}c (${total_net/100:.2f})")
    print(f"{'=' * 80}\n")

    # Save aggregate summary
    try:
        ts = utc_now().strftime("%Y%m%d_%H%M%S")
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        agg_path = LOG_ROOT / f"tennis_aggregate_summary_{ts}.json"
        agg = {
            "sport": "tennis",
            "strategy": "rolling_avg",
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
