# list_cbb_markets.py
#
# Save today's NCAAM markets (ML + Spread + Total) grouped per game.
#
# Goals:
#   1) Game-level teams (away/home) clearly searchable
#   2) Tickers saved consistently + indexed so downstream code can find things fast
#
# Strategy:
#   - Fetch events (GAME, SPREAD, TOTAL) with nested markets
#   - Filter "today" using YYMONDD embedded in tickers (e.g. 26FEB04)
#   - Group by game_key = second dash-chunk in ticker: SERIES-<GAME_KEY>-...
#   - At game level, store:
#       teams: away/home names, (and codes when available)
#       matchup, search_text
#       tickers: ml/spread/total lists
#       ml.by_team_code mapping (e.g. {"ARMY": "...-ARMY", "COLG": "...-COLG"})
#       spread.by_team_code mapping (team_code -> list of tickers)
#       total.lines parsed list (line -> tickers)
#       spread.lines parsed list (best-effort; always keep raw tickers)
#
# Usage:
#   python list_cbb_markets.py
#   python list_cbb_markets.py --date 2026-02-04
#   python list_cbb_markets.py --path todays_ncaam_bundle.json
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
DEFAULT_SERIES_CSV = "KXNCAAMBGAME,KXNCAAMBSPREAD,KXNCAAMBTOTAL"
DEFAULT_FILE = "todays_ncaam_bundle.json"


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
def today_kalshi_ticker_date(day_str: Optional[str], tz_name: str) -> str:
    """
    Convert YYYY-MM-DD (or now in tz) to Kalshi's YYMONDD used in tickers.
    Example: 2026-02-04 -> 26FEB04
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
# FETCH
# -----------------------
def fetch_events_with_nested_markets(
    private_key,
    *,
    events_series_ticker: str,
    status: str = "open",
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {
            "limit": 200,
            "with_nested_markets": True,  # boolean
            "series_ticker": events_series_ticker,
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


def market_matches_day(m: Dict[str, Any], day_token: str) -> bool:
    et = (m.get("event_ticker") or "")
    tk = (m.get("ticker") or "")
    return (day_token in et) or (day_token in tk)


# -----------------------
# PARSING / GROUPING
# -----------------------

def infer_away_home_codes_from_game_key(game_key: str, day_token: str, ml_outcomes: Dict[str, str]) -> Tuple[str, str]:
    """
    Infer (away_code, home_code) using:
      - game_key like: 26FEB04GTCAL
      - day_token like: 26FEB04
      - ml_outcomes keys like: {"GT": "...", "CAL": "..."} (2 outcomes)

    Returns ("", "") if we can't confidently infer.
    """
    if not game_key or not day_token:
        return "", ""

    codes = [c for c in (ml_outcomes or {}).keys() if c]
    if len(codes) != 2:
        return "", ""

    # game token is what's left after the date token
    if not game_key.startswith(day_token):
        return "", ""
    token = game_key[len(day_token):]  # e.g. "GTCAL"
    a, b = codes[0], codes[1]

    if token == a + b:
        return a, b
    if token == b + a:
        return b, a

    return "", ""


def series_from_ticker(ticker: str) -> str:
    return (ticker.split("-", 1)[0] if "-" in ticker else ticker).strip()


def game_key_from_any_ticker(ticker: str) -> str:
    parts = ticker.split("-")
    return parts[1].strip() if len(parts) >= 2 else ""


def market_kind(m: Dict[str, Any]) -> str:
    tk = (m.get("ticker") or "").strip()
    s = series_from_ticker(tk).upper()
    if "SPREAD" in s:
        return "spread"
    if "TOTAL" in s:
        return "total"
    return "ml"


def compact_market_obj(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ticker": (m.get("ticker") or "").strip(),
        "event_ticker": (m.get("event_ticker") or "").strip(),
        "title": (m.get("title") or "").strip(),
        "subtitle": (m.get("subtitle") or "").strip(),
        "status": (m.get("status") or "").strip(),
        "close_time": m.get("close_time", ""),
    }


# Replace your existing _TEAM_TITLE_PATTERNS + parse_teams_from_title with this:

_TEAM_TITLE_PATTERNS = [
    # "Furman at East Tennessee St. Winner?"
    # "Utah St. at New Mexico Total Points?"
    # "Team A at Team B ..." (stop before trailing market question text)
    re.compile(
        r"^(?P<away>.+?)\s+at\s+(?P<home>.+?)\s*(?:"
        r"Winner\?|Total Points\?|Point Spread\?|wins by over.*?\?|Over/Under.*?\?|Over\s+\d+(?:\.\d+)?\??|Under\s+\d+(?:\.\d+)?\??"
        r")?\s*$",
        re.IGNORECASE,
    ),
    # Fallback: "Team A vs Team B ..."
    re.compile(
        r"^(?P<away>.+?)\s+vs\.?\s+(?P<home>.+?)\s*(?:"
        r"Winner\?|Total Points\?|Point Spread\?|wins by over.*?\?|Over/Under.*?\?|Over\s+\d+(?:\.\d+)?\??|Under\s+\d+(?:\.\d+)?\??"
        r")?\s*$",
        re.IGNORECASE,
    ),
]

def parse_teams_from_title(title: str) -> Tuple[str, str]:
    """
    Robustly parse 'Away at Home' from Kalshi market titles without truncating multi-word names.
    """
    t = (title or "").strip()
    if not t:
        return "", ""

    # Normalize common trailing punctuation
    t = t.rstrip()

    for rx in _TEAM_TITLE_PATTERNS:
        m = rx.match(t)
        if m:
            away = (m.group("away") or "").strip()
            home = (m.group("home") or "").strip()

            # Final cleanup: strip dangling punctuation
            away = away.rstrip(" ?:-").strip()
            home = home.rstrip(" ?:-").strip()
            return away, home

    return "", ""


def outcome_code_from_ticker(ticker: str) -> str:
    parts = (ticker or "").split("-")
    return parts[-1].strip() if len(parts) >= 3 else ""


# Best-effort line parsing for totals/spreads.
# NOTE: even if parsing fails, we still keep raw tickers and market objects.
_RX_NUM = re.compile(r"(-?\d+(?:\.\d+)?)")


def try_parse_total_line(m: Dict[str, Any]) -> Optional[float]:
    """
    Totals often encode the number in the ticker suffix (e.g. ...-146 or ...-149)
    or in subtitle/title. We try ticker last chunk first, then subtitle/title numbers.
    """
    ticker = (m.get("ticker") or "").strip()
    parts = ticker.split("-")
    if len(parts) >= 3:
        last = parts[-1]
        if last.isdigit():
            return float(last)

    subtitle = (m.get("subtitle") or "")
    title = (m.get("title") or "")
    for text in (subtitle, title):
        mm = _RX_NUM.search(text)
        if mm:
            try:
                return float(mm.group(1))
            except ValueError:
                pass
    return None


def try_parse_spread_line(m: Dict[str, Any]) -> Optional[float]:
    """
    Spreads are trickier. Often subtitle/title includes something like:
      "wins by over 4.5 Points" or "wins by over 9.5 Points"
    We'll parse the first number we see in subtitle/title.
    """
    subtitle = (m.get("subtitle") or "")
    title = (m.get("title") or "")

    for text in (subtitle, title):
        mm = _RX_NUM.search(text)
        if mm:
            try:
                return float(mm.group(1))
            except ValueError:
                pass
    return None


def build_game_bundle(markets: List[Dict[str, Any]], day_token: str) -> Dict[str, Any]:
    games: Dict[str, Any] = {}

    for raw in markets:
        ticker = (raw.get("ticker") or "").strip()
        if not ticker:
            continue

        game_key = game_key_from_any_ticker(ticker)
        if not game_key:
            continue

        kind = market_kind(raw)
        m = compact_market_obj(raw)

        away_name, home_name = parse_teams_from_title(m["title"])

        g = games.setdefault(
            game_key,
            {
                "game_key": game_key,
                "date_token": day_token,

                # Search-first fields
                "teams": {
                    "away_name": away_name,
                    "home_name": home_name,
                    "away_code": "",  # filled from ML when possible
                    "home_code": "",
                },
                "matchup": f"{away_name} at {home_name}" if away_name and home_name else "",
                "search_text": "",

                # Consistent ticker storage
                "tickers": {"ml": [], "spread": [], "total": []},

                # Richer indexes for fast retrieval
                "ml": {"by_team_code": {}},
                "ml_outcomes": {},
                "spread": {"by_team_code": {}, "lines": []},  # lines: [{team_code,line,ticker}]
                "total": {"lines": []},  # lines: [{line,ticker}]

                # Keep raw compact markets too
                "markets": {
                    "ml": {"event_tickers": [], "markets": []},
                    "spread": {"event_tickers": [], "markets": []},
                    "total": {"event_tickers": [], "markets": []},
                },

                # Representative title (nice-to-have)
                "title": m["title"],
            },
        )

        # If teams missing, attempt to populate from any market title
        if (not g["teams"]["away_name"] and not g["teams"]["home_name"]) and m["title"]:
            a2, h2 = parse_teams_from_title(m["title"])
            if a2 and h2:
                g["teams"]["away_name"], g["teams"]["home_name"] = a2, h2
                g["matchup"] = f"{a2} at {h2}"

        # Prefer a "Winner?" title if present; else keep whatever we have
        if (not g.get("title")) or ("Winner?" not in g.get("title", "") and "Winner?" in m["title"]):
            if m["title"]:
                g["title"] = m["title"]

        # Track event ticker
        et = m["event_ticker"]
        if et and et not in g["markets"][kind]["event_tickers"]:
            g["markets"][kind]["event_tickers"].append(et)

        # Store compact market
        g["markets"][kind]["markets"].append(m)

        # Store ticker in top-level lists
        if m["ticker"] and m["ticker"] not in g["tickers"][kind]:
            g["tickers"][kind].append(m["ticker"])

        # --- ML outcome map (code -> ticker) ---
        if kind == "ml" and m["ticker"]:
            oc = outcome_code_from_ticker(m["ticker"])
            if oc and oc not in g["ml_outcomes"]:
                g["ml_outcomes"][oc] = m["ticker"]

        # Build indexes
        if kind == "ml":
            oc = outcome_code_from_ticker(m["ticker"])
            if oc and oc not in g["ml"]["by_team_code"]:
                g["ml"]["by_team_code"][oc] = m["ticker"]

        elif kind == "spread":
            # outcome code for spread is NOT always last chunk; but commonly includes team code prefix.
            # We'll still use last chunk as a "code bucket" and also parse line from title/subtitle.
            oc = outcome_code_from_ticker(m["ticker"])
            if oc:
                g["spread"]["by_team_code"].setdefault(oc, [])
                if m["ticker"] not in g["spread"]["by_team_code"][oc]:
                    g["spread"]["by_team_code"][oc].append(m["ticker"])

            line = try_parse_spread_line(m)
            if line is not None:
                g["spread"]["lines"].append(
                    {
                        "team_code": oc,
                        "line": line,
                        "ticker": m["ticker"],
                    }
                )

        elif kind == "total":
            line = try_parse_total_line(m)
            if line is not None:
                g["total"]["lines"].append({"line": line, "ticker": m["ticker"]})

    # Cleanup / dedupe / sort, plus fill in away/home team codes from ML tickers when possible
    for game_key, g in games.items():
        # Dedup + sort tickers
        for kind in ("ml", "spread", "total"):
            g["tickers"][kind] = sorted(set(g["tickers"][kind]))
            g["markets"][kind]["event_tickers"] = sorted(set(g["markets"][kind]["event_tickers"]))

            # Dedup compact markets by ticker
            seen = set()
            uniq = []
            for mm in g["markets"][kind]["markets"]:
                t = mm.get("ticker") or ""
                if t and t not in seen:
                    seen.add(t)
                    uniq.append(mm)
            uniq.sort(key=lambda x: (x.get("event_ticker") or "", x.get("ticker") or ""))
            g["markets"][kind]["markets"] = uniq

        # Sort spread by_team_code lists
        for tc in list(g["spread"]["by_team_code"].keys()):
            g["spread"]["by_team_code"][tc] = sorted(set(g["spread"]["by_team_code"][tc]))

        # Sort parsed lines
        g["total"]["lines"] = sorted(
            {f'{x["ticker"]}': x for x in g["total"]["lines"]}.values(),
            key=lambda x: (x["line"], x["ticker"]),
        )
        g["spread"]["lines"] = sorted(
            {f'{x["ticker"]}': x for x in g["spread"]["lines"]}.values(),
            key=lambda x: (x.get("team_code") or "", x["line"], x["ticker"]),
        )

        # Fill away/home codes if we can infer from ML outcomes:
        # ML outcomes are usually 2 codes. We'll map them to away/home by matching title text.
        ml_codes = list(g["ml"]["by_team_code"].keys())
        away_name = g["teams"]["away_name"]
        home_name = g["teams"]["home_name"]

        # If exactly 2 codes and we have team names, do a best-effort:
        # choose the code that appears in the ML ticker suffix; we can't map to names perfectly
        # without a team-code dictionary, but often the home team's code is the one used in ticker suffix
        # and names are present.
        if len(ml_codes) == 2:
            # If one code is a substring of away/home names (rare), use it; else just store in deterministic order
            a_code = ""
            h_code = ""
            lower_away = (away_name or "").lower()
            lower_home = (home_name or "").lower()
            for c in ml_codes:
                if c.lower() in lower_away:
                    a_code = c
                if c.lower() in lower_home:
                    h_code = c
            if not a_code or not h_code:
                # deterministic: sort codes and assign (still useful for lookup)
                codes_sorted = sorted(ml_codes)
                a_code = a_code or codes_sorted[0]
                h_code = h_code or codes_sorted[1]

            g["teams"]["away_code"] = g["teams"]["away_code"] or a_code
            g["teams"]["home_code"] = g["teams"]["home_code"] or h_code

        # Search text
        g["search_text"] = " ".join(
            [
                g["teams"]["away_name"],
                g["teams"]["home_name"],
                g.get("matchup", ""),
                g.get("title", ""),
                g.get("game_key", ""),
                " ".join(g["tickers"]["ml"]),
                " ".join(g["tickers"]["spread"][:3]),  # small sample to keep string smaller
                " ".join(g["tickers"]["total"][:3]),
            ]
        ).strip()

    # Sort games by matchup/title then key
    games_sorted = dict(
        sorted(
            games.items(),
            key=lambda kv: (
                kv[1].get("matchup") or kv[1].get("title") or "",
                kv[0],
            ),
        )
    )

    return {
        "meta": {
            "env": ENV,
            "base_url": BASE_URL,
            "date_token": day_token,
            "created_utc": datetime.utcnow().isoformat() + "Z",
        },
        "games": games_sorted,
    }


def print_game_summary(bundle: Dict[str, Any]) -> None:
    games = bundle.get("games", {}) or {}
    dt = bundle.get("meta", {}).get("date_token", "")
    print(f"\nSummary for {dt}: {len(games)} games\n")

    for _, g in games.items():
        away = g.get("teams", {}).get("away_name", "")
        home = g.get("teams", {}).get("home_name", "")
        label = f"{away} at {home}" if away and home else (g.get("title") or g.get("game_key"))

        ml_n = len(g.get("tickers", {}).get("ml", []))
        sp_n = len(g.get("tickers", {}).get("spread", []))
        to_n = len(g.get("tickers", {}).get("total", []))

        print(f"{label} | ML={ml_n}  Spread={sp_n}  Total={to_n}")


def write_json(path: str, bundle: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nWrote bundle JSON: {path}")


# -----------------------
# MAIN
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Save today's Kalshi NCAAM ML+Spread+Total grouped per game.")
    ap.add_argument("--date", default="", help="YYYY-MM-DD (defaults to today in tz)")
    ap.add_argument("--tz", default=DEFAULT_TZ, help=f"Timezone (default {DEFAULT_TZ})")
    ap.add_argument(
        "--series",
        default=DEFAULT_SERIES_CSV,
        help="Comma-separated series tickers to fetch (default: GAME, SPREAD, TOTAL)",
    )
    ap.add_argument("--status", default="open", help="Event status filter (default open)")
    ap.add_argument("--path", default=DEFAULT_FILE, help="Output JSON path")
    args = ap.parse_args()

    if not API_KEY_ID:
        raise SystemExit("Missing KALSHI_API_KEY_ID (set in .env)")
    if not PRIVATE_KEY_PATH:
        raise SystemExit("Missing KALSHI_PRIVATE_KEY_PATH (set in .env)")
    if not os.path.exists(PRIVATE_KEY_PATH):
        raise SystemExit(f"Private key file not found: {PRIVATE_KEY_PATH}")

    private_key = _load_private_key(PRIVATE_KEY_PATH)

    day_token = today_kalshi_ticker_date(args.date or None, args.tz)
    series_list = [s.strip() for s in args.series.split(",") if s.strip()]

    print(f"ENV={ENV} BASE_URL={BASE_URL}")
    print(f"Filtering to ticker date token: {day_token}")
    print(f"Series to fetch: {', '.join(series_list)}")

    all_markets: List[Dict[str, Any]] = []
    total_events = 0

    for series in series_list:
        print(f"\nFetching EVENTS series={series} status={args.status} with nested markets...")
        events = fetch_events_with_nested_markets(private_key, events_series_ticker=series, status=args.status)
        mkts = extract_markets(events)
        total_events += len(events)
        all_markets.extend(mkts)
        print(f"  fetched events={len(events)} markets={len(mkts)}")

    todays = [m for m in all_markets if market_matches_day(m, day_token)]

    print(f"\nTotals across series: events={total_events} markets={len(all_markets)}")
    print(f"Today's markets for {day_token}: {len(todays)}")

    bundle = build_game_bundle(todays, day_token=day_token)
    print_game_summary(bundle)
    write_json(args.path, bundle)


if __name__ == "__main__":
    main()
