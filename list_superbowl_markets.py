# list_superbowl_markets.py
#
# Find/print Kalshi Super Bowl (NFL) market tickers.
#
# Features:
#   - Attempts to discover NFL/Super Bowl series tickers via /trade-api/v2/series (if supported)
#   - Falls back to user-provided --series CSV (or env NFL_SERIES_CSV)
#   - Filters markets by:
#       (a) Kalshi date token in ticker/event_ticker (e.g., 26FEB08)
#       (b) keyword match in title/subtitle (e.g., "Super Bowl", "Patriots", "Seahawks")
#   - Prints a compact list of matching markets with ticker/title/close_time
#
# Usage:
#   python list_superbowl_markets.py
#   python list_superbowl_markets.py --date 2026-02-08
#   python list_superbowl_markets.py --contains "Super Bowl" --contains "Patriots"
#   python list_superbowl_markets.py --series "KXNFLGAME,KXNFLSPREAD,KXNFLTOTAL"
#   python list_superbowl_markets.py --status open
#
import os
import time
import json
import base64
import argparse
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

load_dotenv()

ENV = (os.getenv("KALSHI_ENV") or "DEMO").upper()  # DEMO or PROD
BASE_URL = "https://demo-api.kalshi.co" if ENV == "DEMO" else "https://api.elections.kalshi.com"

API_KEY_ID = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
PRIVATE_KEY_PATH = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()

DEFAULT_TZ = "America/New_York"

# Optional fallback if series discovery isn't available:
# You can set this in .env, or pass --series
DEFAULT_NFL_SERIES_CSV = (os.getenv("NFL_SERIES_CSV") or "").strip()

# Default keyword filters for SB night (add/remove as you want)
DEFAULT_KEYWORDS = [
    "super bowl",
    "superbowl",
    "patriots",
    "seahawks",
]


# -----------------------
# AUTH / HTTP
# -----------------------
def _now_ms() -> str:
    return str(int(time.time() * 1000))


def _load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign_request(private_key, ts_ms: str, method: str, path: str) -> str:
    path_no_q = path.split("?")[0]
    msg = f"{ts_ms}{method.upper()}{path_no_q}".encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def _auth_headers(private_key, method: str, path: str) -> Dict[str, str]:
    ts_ms = _now_ms()
    signature = _sign_request(private_key, ts_ms, method, path)
    return {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json",
    }


def _get(private_key, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = BASE_URL + path
    headers = _auth_headers(private_key, "GET", path)
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"GET {path} failed: {r.status_code} {r.text}")
    return r.json()


# -----------------------
# DATE TOKEN
# -----------------------
def kalshi_ticker_date_token(day_str: Optional[str], tz_name: str) -> str:
    """
    Convert YYYY-MM-DD (or now in tz) to Kalshi's YYMONDD used in tickers.
    Example: 2026-02-08 -> 26FEB08
    """
    tz = ZoneInfo(tz_name)
    if day_str:
        d = datetime.strptime(day_str, "%Y-%m-%d").date()
    else:
        d = datetime.now(tz).date()

    yy = f"{d.year % 100:02d}"
    mon = d.strftime("%b").upper()
    dd = f"{d.day:02d}"
    return f"{yy}{mon}{dd}"


# -----------------------
# SERIES DISCOVERY (best-effort)
# -----------------------
def fetch_all_series(private_key) -> List[Dict[str, Any]]:
    """
    Try to list all series. If Kalshi doesn't expose this endpoint in your environment,
    we'll catch and return empty.
    """
    out: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    # NOTE: endpoint path is best-effort. If it errors, we fallback.
    path = "/trade-api/v2/series"

    while True:
        params: Dict[str, Any] = {"limit": 200}
        if cursor:
            params["cursor"] = cursor

        resp = _get(private_key, path, params=params)
        chunk = resp.get("series", []) or resp.get("data", []) or []
        out.extend(chunk)

        cursor = resp.get("cursor")
        if not cursor:
            break

    return out


def infer_nfl_series_tickers(series: List[Dict[str, Any]]) -> List[str]:
    """
    Heuristic: keep series that look NFL-ish or SB-ish.
    We rely on ticker/title/description containing NFL / Super Bowl.
    """
    keep: List[str] = []
    for s in series:
        tk = (s.get("ticker") or "").strip()
        title = (s.get("title") or "").strip()
        desc = (s.get("description") or "").strip()

        hay = f"{tk} {title} {desc}".lower()
        if ("nfl" in hay) or ("super bowl" in hay) or ("superbowl" in hay) or ("sb" in hay):
            if tk:
                keep.append(tk)

    # Dedup, stable order
    seen = set()
    out: List[str] = []
    for tk in keep:
        if tk not in seen:
            seen.add(tk)
            out.append(tk)
    return out


# -----------------------
# FETCH EVENTS + MARKETS
# -----------------------
def fetch_events_with_nested_markets(
    private_key,
    *,
    series_ticker: str,
    status: str = "open",
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {
            "limit": 200,
            "with_nested_markets": True,
            "series_ticker": series_ticker,
        }
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        resp = _get(private_key, "/trade-api/v2/events", params=params)
        events.extend(resp.get("events", []) or [])

        cursor = resp.get("cursor")
        if not cursor:
            break

    return events


def extract_markets(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for ev in events:
        for m in (ev.get("markets") or []):
            t = (m.get("ticker") or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(m)
    return out


def compact_market_obj(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ticker": (m.get("ticker") or "").strip(),
        "event_ticker": (m.get("event_ticker") or "").strip(),
        "title": (m.get("title") or "").strip(),
        "subtitle": (m.get("subtitle") or "").strip(),
        "status": (m.get("status") or "").strip(),
        "close_time": m.get("close_time", ""),
    }


def game_key_from_any_ticker(ticker: str) -> str:
    parts = ticker.split("-")
    return parts[1].strip() if len(parts) >= 2 else ""


def market_matches_filters(m: Dict[str, Any], day_token: str, keywords_lc: List[str]) -> bool:
    et = (m.get("event_ticker") or "").lower()
    tk = (m.get("ticker") or "").lower()
    title = (m.get("title") or "").lower()
    subtitle = (m.get("subtitle") or "").lower()

    # Date token match (strong signal)
    if day_token and (day_token.lower() in et or day_token.lower() in tk):
        return True

    # Keyword match
    hay = f"{tk} {et} {title} {subtitle}"
    return any(k in hay for k in keywords_lc)


def print_matches(markets: List[Dict[str, Any]]) -> None:
    if not markets:
        print("\nNo matches found.\n")
        return

    # Sort: game_key then title
    def _k(m: Dict[str, Any]) -> Tuple[str, str, str]:
        t = (m.get("ticker") or "")
        return (game_key_from_any_ticker(t), (m.get("title") or ""), t)

    markets_sorted = sorted(markets, key=_k)

    print("\nMATCHING MARKETS\n" + "=" * 80)
    last_gk = None
    for m in markets_sorted:
        cm = compact_market_obj(m)
        gk = game_key_from_any_ticker(cm["ticker"])
        if gk != last_gk:
            print("\n" + "-" * 80)
            print(f"GAME_KEY: {gk}")
            last_gk = gk

        print(f"Ticker:   {cm['ticker']}")
        print(f"Title:    {cm['title']}")
        if cm.get("subtitle"):
            print(f"Subtitle: {cm['subtitle']}")
        if cm.get("close_time"):
            print(f"Close:    {cm['close_time']}")
        print("")


# -----------------------
# MAIN
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Find Kalshi Super Bowl/NFL market tickers.")
    ap.add_argument("--date", default="", help="YYYY-MM-DD (defaults to today in tz)")
    ap.add_argument("--tz", default=DEFAULT_TZ, help=f"Timezone (default {DEFAULT_TZ})")
    ap.add_argument("--status", default="open", help="Event status filter (default open)")
    ap.add_argument(
        "--series",
        default="",
        help="Comma-separated series tickers. If empty, tries to auto-discover NFL-ish series via /trade-api/v2/series.",
    )
    ap.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Keyword filter (repeatable). Example: --contains 'Super Bowl' --contains 'Patriots'",
    )
    ap.add_argument(
        "--dump",
        default="",
        help="Optional: write matching compact markets to this JSON file path",
    )
    args = ap.parse_args()

    if not API_KEY_ID:
        raise SystemExit("Missing KALSHI_API_KEY_ID (set in .env)")
    if not PRIVATE_KEY_PATH:
        raise SystemExit("Missing KALSHI_PRIVATE_KEY_PATH (set in .env)")
    if not os.path.exists(PRIVATE_KEY_PATH):
        raise SystemExit(f"Private key file not found: {PRIVATE_KEY_PATH}")

    private_key = _load_private_key(PRIVATE_KEY_PATH)

    day_token = kalshi_ticker_date_token(args.date or None, args.tz)
    print(f"ENV={ENV} BASE_URL={BASE_URL}")
    print(f"Using ticker date token: {day_token}")
    print(f"Status filter: {args.status}")

    # Build keywords list
    keywords = args.contains[:] if args.contains else DEFAULT_KEYWORDS[:]
    keywords_lc = [k.lower().strip() for k in keywords if k and k.strip()]
    print(f"Keyword filters: {keywords_lc}")

    # Determine series list
    series_list: List[str] = []
    if args.series.strip():
        series_list = [s.strip() for s in args.series.split(",") if s.strip()]
        print(f"Using user-provided series: {series_list}")
    elif DEFAULT_NFL_SERIES_CSV:
        series_list = [s.strip() for s in DEFAULT_NFL_SERIES_CSV.split(",") if s.strip()]
        print(f"Using NFL_SERIES_CSV from env: {series_list}")
    else:
        print("No --series provided. Attempting series auto-discovery via /trade-api/v2/series ...")
        try:
            all_series = fetch_all_series(private_key)
            inferred = infer_nfl_series_tickers(all_series)
            series_list = inferred
            print(f"Discovered {len(series_list)} NFL-ish series tickers.")
            for s in series_list[:30]:
                print(f"  - {s}")
            if len(series_list) > 30:
                print("  ... (truncated)")
        except Exception as e:
            print(f"Series discovery failed: {e}")
            print("Provide series tickers via --series or set NFL_SERIES_CSV in .env.")
            series_list = []

    if not series_list:
        print("\nNo series tickers available to query.")
        print("Try one of these:")
        print("  1) Set NFL_SERIES_CSV in .env (comma-separated)")
        print("  2) Run: python list_superbowl_markets.py --series 'KX...,...'")
        return

    # Fetch markets across the selected series
    all_markets: List[Dict[str, Any]] = []
    total_events = 0

    for series in series_list:
        try:
            print(f"\nFetching events for series={series} ...")
            events = fetch_events_with_nested_markets(private_key, series_ticker=series, status=args.status)
            mkts = extract_markets(events)
            total_events += len(events)
            all_markets.extend(mkts)
            print(f"  events={len(events)} markets={len(mkts)}")
        except Exception as e:
            print(f"  series={series} failed: {e}")

    print(f"\nTotals: events={total_events} markets={len(all_markets)}")

    # Filter markets
    matches = [m for m in all_markets if market_matches_filters(m, day_token=day_token, keywords_lc=keywords_lc)]
    print(f"Matches: {len(matches)}")

    print_matches(matches)

    # Optional dump
    if args.dump.strip():
        compact = [compact_market_obj(m) for m in matches]
        with open(args.dump.strip(), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "env": ENV,
                        "base_url": BASE_URL,
                        "date_token": day_token,
                        "created_utc": datetime.utcnow().isoformat() + "Z",
                        "keywords": keywords_lc,
                        "status": args.status,
                        "series": series_list,
                    },
                    "markets": compact,
                },
                f,
                indent=2,
            )
        print(f"\nWrote matches JSON: {args.dump.strip()}")


if __name__ == "__main__":
    main()
