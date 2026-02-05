# combo_multigame_driver.py

import os
import threading
from typing import Dict, Any, List, Tuple, Optional
from espn_game_clock import EspnGameClock
import json


from combo_vnext import (
    _load_private_key,
    _get,
    get_markets_in_series,
    parse_iso,
    utc_now,
    run_combo_for_one_market,
    print_status,
    SERIES_TICKER,
)

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def date_token_to_yyyymmdd(date_token: str) -> str:
    # "26FEB04" -> "20260204"
    yy = int(date_token[0:2])
    mon = _MONTHS[date_token[2:5].upper()]
    dd = int(date_token[5:7])
    return f"{2000+yy:04d}{mon:02d}{dd:02d}"

BUNDLE_JSON_PATH = os.getenv("BUNDLE_JSON_PATH", "todays_ncaam_bundle.json")


# Optional: if you want bundle support later, you can also import:
# from combo_vnext import load_ncaam_bundle, resolve_ml_ticker_from_bundle


# ======================
# USER CONFIG: GAMES
# ======================
# Edit this list before running.
#
# Each entry:
#   - label: just for logs / filenames
#   - team_name: name (or substring) of the side your model prob is for
#   - p_win: your model's probability that team_name's side wins (0–1)
#   - capital_frac: fraction of account bankroll to allocate to this game (0–1)
#
# The sum of capital_frac should be <= 1.0 for safety.

GAME_CONFIGS: List[Dict[str, Any]] = [
    # EXAMPLE:
    # {
    #     "label": "Game1_UNLV",
    #     "team_name": "UNLV",
    #     "p_win": 0.57,
    #     "capital_frac": 0.30,
    # },
    # {
    #     "label": "Game2_SaintLouis",
    #     "team_name": "Saint Louis",
    #     "p_win": 0.62,
    #     "capital_frac": 0.25,
    # },

        {
        "label": "Iowa at Washington",
        "team_name":  "Iowa",
        "p_win": 0.58,
        "capital_frac": 0.33,
    },
    {
        "label": "Utah St. at New Mexico",
        "team_name": "USU",
        "p_win": 0.54,
        "capital_frac": 0.33,
    },
        {
        "label": "Washington St. at Oregon St.",
        "team_name": "WSU",
        "p_win": 0.6,
        "capital_frac": 0.33,
    },

]


# ======================
# PORTFOLIO / BANKROLL
# ======================

def get_available_bankroll_dollars(private_key) -> float:
    """
    Fetch available bankroll from Kalshi via the official Get Balance endpoint.

    Docs: GET /portfolio/balance
    Response JSON:
      - balance: available balance in cents
      - portfolio_value: total portfolio value in cents
      - updated_ts: unix timestamp
    """
    # Correct endpoint: /trade-api/v2/portfolio/balance
    resp = _get(private_key, "/trade-api/v2/portfolio/balance")

    # Helpful for the first run / debugging
    print_status(f"Balance response: {resp}")

    if "balance" not in resp and "portfolio_value" not in resp:
        raise RuntimeError(
            "Unexpected response from /portfolio/balance; "
            "expected 'balance' and/or 'portfolio_value' keys."
        )

    # Use available cash balance in cents by default.
    balance_raw = resp.get("balance")

    if balance_raw is None:
        # Fallback: if for some reason only portfolio_value is present,
        # you could choose to use that instead (less ideal for trading).
        balance_raw = resp.get("portfolio_value")

    # Handle both int and string representations
    try:
        balance_cents = int(balance_raw)
    except (TypeError, ValueError):
        raise RuntimeError(f"Could not parse balance value from /portfolio/balance: {balance_raw!r}")

    dollars = balance_cents / 100.0
    print_status(f"Using available balance from /portfolio/balance: {balance_cents}c -> ${dollars:.2f}")
    return dollars


# ======================
# MARKET RESOLUTION
# ======================

def resolve_ticker_for_team(
    private_key,
    team_name: str,
    series_ticker: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve the ML market ticker for a given team within a Kalshi series.

    IMPORTANT: For NCAAM ML winner markets, each game has two markets like:

        KXNCAAMBGAME-26FEB04BYUOKST-BYU   -> "BYU wins?"
        KXNCAAMBGAME-26FEB04BYUOKST-OKST -> "Oklahoma St. wins?"

    Here we treat `team_name` as the team code (BYU, OKST, CLEM, STAN, etc.)
    and FIRST try to match the ticker suffix, e.g. "-BYU". This guarantees
    we get the correct side instead of a random side whose title happens
    to contain the substring.

    If no suffix match is found (e.g. you pass "Clemson" instead of "CLEM"),
    we fall back to the previous "team_name in title" behavior.
    """
    markets = get_markets_in_series(private_key, series_ticker)
    team_code = team_name.upper()

    # 1) Prefer strict ticker-suffix match: ...-BYU, ...-CLEM, etc.
    suffix = f"-{team_code}"
    candidates: List[Dict[str, Any]] = [
        m for m in markets
        if (m.get("ticker") or "").upper().endswith(suffix)
    ]

    # 2) Fallback: substring in title (handles "Clemson", "Oklahoma State", etc.)
    if not candidates:
        team_l = team_name.lower()
        for m in markets:
            title = (m.get("title") or "").lower()
            if team_l in title:
                candidates.append(m)

    if not candidates:
        raise RuntimeError(
            f"No markets found in series {series_ticker} for team '{team_name}'"
        )

    # If multiple, prefer ones that are still in the future.
    now = utc_now()
    future = [m for m in candidates if parse_iso(m["close_time"]) > now]
    if future:
        candidates = future

    # Pick the earliest closing among remaining candidates.
    candidates.sort(key=lambda m: parse_iso(m["close_time"]))
    chosen = candidates[0]

    print_status(
        f"Resolved team '{team_name}' to ticker={chosen.get('ticker')} | "
        f"title={chosen.get('title')} | close_time={chosen.get('close_time')}"
    )

    return chosen["ticker"], chosen


# ======================
# WORKER PER GAME
# ======================

def start_game_worker(
    label: str,
    team_name: str,
    p_win: float,
    capital_frac: float,
    total_bankroll_dollars: float,
    series_ticker: str,
    bundle: Optional[Dict[str, Any]] = None,
):
    """
    Run combo_vnext on a single game in its own thread with a per-game capital cap.
    """
    # Each thread loads its own private key for safety
    api_key_id = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
    priv_path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()
    if not api_key_id or not priv_path:
        raise RuntimeError("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set in env")

    private_key = _load_private_key(priv_path)

    # Capital allocation for this game
    game_bankroll = total_bankroll_dollars * capital_frac

    # Convert model prob -> fair prices in cents
    fair_yes_cents = int(round(p_win * 100))
    fair_no_cents = int(round((1.0 - p_win) * 100))

    # Resolve ticker & market
    ticker, market = resolve_ticker_for_team(private_key, team_name, series_ticker)

    # --- ESPN clock wiring (bundle + ticker-derived game_key) ---
    clock_cb = None
    try:
        if bundle:
            parts = (ticker or "").split("-")
            # Expected: KXNCAAMBGAME-26FEB04BYUOKST-BYU  -> parts[1]=game_key, parts[-1]=YES team code
            if len(parts) >= 3:
                game_key = parts[1]
                yes_team_code = parts[-1].upper()

                games = bundle.get("games") or {}
                game = games.get(game_key)

                if game:
                    date_token = game.get("date_token") or game_key[:7]
                    yyyymmdd = date_token_to_yyyymmdd(date_token)

                    ml_outcomes = game.get("ml_outcomes") or {}
                    codes = [c.upper() for c in ml_outcomes.keys() if c]

                    opponent_code = None
                    if len(codes) == 2 and yes_team_code in codes:
                        opponent_code = codes[0] if codes[1] == yes_team_code else codes[1]

                    clock = EspnGameClock(
                        yyyymmdd=yyyymmdd,
                        team_code=yes_team_code,
                        opponent_code=opponent_code,
                        cache_ttl_secs=10,
                    )
                    clock_cb = clock.get_secs_to_game_end

                    print_status(
                        f"[{label}] ESPN clock armed: game_key={game_key} date={yyyymmdd} "
                        f"team={yes_team_code} opp={opponent_code}"
                    )
                else:
                    print_status(
                        f"[{label}] ESPN clock not armed: bundle missing game_key={game_key} "
                        f"(ticker={ticker}) -> using Kalshi close_time fallback"
                    )
    except Exception as e:
        print_status(f"[{label}] ESPN clock init failed: {e} -> using Kalshi close_time fallback")
    # --- end ESPN clock wiring ---


    print_status(
        f"[{label}] Starting worker: team={team_name}, p_win={p_win:.3f}, "
        f"fair_yes={fair_yes_cents}c, capital_frac={capital_frac:.3f}, "
        f"per-game bankroll=${game_bankroll:.2f}"
    )

    # Run the combo strategy with per-game collateral cap
    run_combo_for_one_market(
        private_key=private_key,
        ticker=ticker,
        fair_yes_cents=fair_yes_cents,
        fair_no_cents=fair_no_cents,
        market=market,
        log_label=f"{label}_{ticker}",
        max_collateral_dollars=game_bankroll,
        get_secs_to_game_end=clock_cb,
    )

    print_status(f"[{label}] Worker finished: {ticker}")


# ======================
# MAIN
# ======================

def main() -> None:
    if not GAME_CONFIGS:
        print("GAME_CONFIGS is empty. Edit combo_multigame_driver.py and add your games.", flush=True)
        return

    # Quick sanity check on capital fractions
    total_frac = sum(float(g["capital_frac"]) for g in GAME_CONFIGS)
    if total_frac > 1.0 + 1e-6:
        raise RuntimeError(
            f"Sum of capital_frac in GAME_CONFIGS is {total_frac:.3f} (>1.0). "
            "Reduce allocations so they fit within your bankroll."
        )

    # Load key once just to query bankroll
    api_key_id = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
    priv_path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()
    if not api_key_id or not priv_path:
        raise RuntimeError("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set in env")

    private_key = _load_private_key(priv_path)
    bankroll = get_available_bankroll_dollars(private_key)
    print_status(f"Total available bankroll ≈ ${bankroll:.2f}")

    bundle = None
    if os.path.exists(BUNDLE_JSON_PATH):
        try:
            with open(BUNDLE_JSON_PATH, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            print_status(f"Loaded bundle: {BUNDLE_JSON_PATH}")
        except Exception as e:
            print_status(f"Failed to load bundle {BUNDLE_JSON_PATH}: {e} (ESPN clock will be disabled)")
            bundle = None
    else:
        print_status(f"Bundle not found at {BUNDLE_JSON_PATH} (ESPN clock will be disabled)")


    # Kick off a thread per game
    threads: List[threading.Thread] = []
    for cfg in GAME_CONFIGS:
        label = str(cfg["label"])
        team_name = str(cfg["team_name"])
        p_win = float(cfg["p_win"])
        capital_frac = float(cfg["capital_frac"])

        t = threading.Thread(
            target=start_game_worker,
            kwargs={
                "label": label,
                "team_name": team_name,
                "p_win": p_win,
                "capital_frac": capital_frac,
                "total_bankroll_dollars": bankroll,
                "series_ticker": SERIES_TICKER,
                "bundle": bundle,
            },
            name=label,
            daemon=False,
        )
        t.start()
        threads.append(t)

    # Wait for all games to finish (they'll each stop near their market's close)
    for t in threads:
        t.join()

    print_status("All game workers have finished.")


if __name__ == "__main__":
    main()
