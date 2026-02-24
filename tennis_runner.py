# tennis_runner.py
# ATP Tennis match runner — mirrors tonight_runner_cloud.py for CBB
#
# Differences from CBB runner:
#   - No ESPN clock (tennis doesn't have reliable free live score APIs with win prob)
#   - Uses Kalshi close_time as sole clock source
#   - Market discovery via KXATPMATCH series
#   - Matches configured with player codes instead of team codes
#
# Usage:
#   python tennis_runner.py                          # uses MATCHES list below
#   python tennis_runner.py --bundle todays_tennis_bundle.json  # from bundle
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
    MeanReversionStrategy,
    ExposureTracker,
    GameRunner,
)

import re

# ============================================================================
# CONFIGURATION
# ============================================================================

TENNIS_SERIES = ["KXATPMATCH", "KXATPCHALLENGERMATCH"]

# Today's matches — edit before running
# player_code must match the Kalshi ticker suffix (e.g. "POP", "RUB", "MEN")
MATCHES = [
    {
        "label": "Tabilo vs Barrios Vera",
        "player_code": "TAB",
        "opponent_code": "BAR",
        "match_key": "26FEB22TABBAR",
        "model_p_win": 0.50,
    },
    {
        "label": "Cerundolo vs Garin",
        "player_code": "CER",
        "opponent_code": "GAR",
        "match_key": "26FEB22CERGAR",
        "model_p_win": 0.50,
    },
    {
        "label": "Passaro vs Vallejo",
        "player_code": "PAS",
        "opponent_code": "VAL",
        "match_key": "26FEB24PASVAL",
        "model_p_win": 0.50,
    },
    {
        "label": "La Serna vs Aboian",
        "player_code": "LA",
        "opponent_code": "ABO",
        "match_key": "26FEB24LAABO",
        "series": "KXATPCHALLENGERMATCH",
        "model_p_win": 0.50,
    },
]


# ============================================================================
# LOG DIRECTORY
# ============================================================================

LOG_ROOT = Path(os.getenv("KALSHI_LOG_ROOT", "kalshi-logs"))


def safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)


def build_match_log_dir(match_label: str, match_date: str) -> Path:
    return (
        LOG_ROOT
        / "kalshi"
        / "tennis"
        / "atp"
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
              bundle: Optional[Dict[str, Any]] = None):
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

        # Log directory
        match_date = utc_now().strftime("%Y-%m-%d")
        log_dir = build_match_log_dir(label, match_date)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Load strategy config overrides
        strategy_config_overrides = {}
        try:
            cfg_path = REPO_ROOT / "strategy_config.json"
            if cfg_path.exists():
                strategy_config_overrides = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            print_status(f"[{label}] Warning: failed to load strategy_config.json: {e}")

        # Strategy: MR-only with tennis-specific tuning
        #
        # Tennis vs CBB differences that affect MR:
        #  - Prices move in step-functions (flat during holds, jumps on breaks/sets)
        #    → shorter lookback so the mean tracks regime shifts faster
        #  - No ESPN clock gating → lower warmup to start trading sooner
        #  - Baseline volatility is spikier (break points can move 10-15c instantly)
        #    → wider std multipliers to avoid over-triggering on normal tennis swings
        #  - Capital split across 7+ matches vs 3-4 CBB games
        #    → fewer max positions per match
        TENNIS_MR_OVERRIDES = {
            "lookback": 80,              # 80 samples (~4 min) vs CBB 120 (~6 min)
            "warmup_samples": 80,        # faster warmup (no ESPN gate)
            "max_positions": 3,          # fewer positions per match (more matches running)
            "high_vol_std_mult": 2.0,    # wider threshold (tennis spikes are normal)
            "low_vol_std_mult": 3.0,     # wider low-vol threshold
            "dead_lookback": 50,         # 50 samples (~2.5 min) — tennis has longer flat
                                         # stretches during service holds before big jumps
            "dead_min_move": 2.0,        # 2c vs CBB 3c — more tolerant of quiet periods
            "revert_min_cents": 4.0,     # 4c vs CBB 2c — hold for bigger reversion;
                                         # tennis swings are larger so 2c bounce is noise
            "mean_shift_exit_cents": 5.0,  # 5c vs CBB 3c — shorter lookback makes mean
                                         # drift faster, so raise the "thesis collapsed"
                                         # threshold to avoid premature exits
        }

        strategies = [
            MeanReversionStrategy(
                max_capital=allocation,
                preferred_side=preferred_side,
                exposure_tracker=exposure,
            ),
        ]

        # Apply tennis defaults, then strategy_config.json overrides on top
        for strat in strategies:
            strat.update_params(TENNIS_MR_OVERRIDES)
            overrides = strategy_config_overrides.get(strat.name, {})
            if overrides:
                strat.update_params(overrides)

        print_status(
            f"[{label}] MR:${allocation:.2f} | "
            f"Preferred:{preferred_side.upper()} | "
            f"Maker entries + MP20"
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
            base_strategy_overrides={"mean_reversion": TENNIS_MR_OVERRIDES},
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
    Priority: MATCHES_JSON (inline) → MATCHES_JSON_PATH (file path)
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


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="ATP Tennis match runner for Kalshi")
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
    print("  ATP TENNIS RUNNER — MR-ONLY + MAKER ENTRIES + MP20")
    print("  No ESPN clock — using Kalshi close time only")
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

    # Fallback to hardcoded MATCHES
    if matches is None:
        matches = list(MATCHES)

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

    for match in matches:
        match_pk = _load_private_key(key_path)
        t = threading.Thread(
            target=run_match,
            args=(match, match_pk, results, bundle),
            name=match["label"],
            daemon=False,
        )
        t.start()
        threads.append(t)
        print_status(f"Started: {match['label']}")

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
