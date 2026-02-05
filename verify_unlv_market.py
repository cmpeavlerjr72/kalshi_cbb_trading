# verify_unlv_market.py
# Read-only sanity check (PROD/DEMO):
# 1) Loads .env
# 2) Authenticated call to list open markets (optionally scoped to series_ticker)
# 3) Finds the UNLV market (and tries to confirm opponent Fresno via subtitle/title)
# 4) Fetches PUBLIC orderbook for the found ticker
# 5) Prints best bids + implied asks + a few candidate matches for debugging
#
# Usage (PowerShell):
#   pip install python-dotenv requests cryptography
#   & "C:/Python 3/python.exe" c:/Users/cmpea/Kalshi/verify_unlv_market.py
#
# .env should contain:
#   KALSHI_ENV=PROD   (or DEMO)
#   KALSHI_API_KEY_ID=...
#   KALSHI_PRIVATE_KEY_PATH=C:\Users\cmpea\Kalshi\kalshi_private_key.pem

import os
import time
import base64
import requests
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


# =========================
# CONFIG (edit if needed)
# =========================
TEAM_PRIMARY = "unlv"
TEAM_OPP = "fresno"        # catches "Fresno", "Fresno St", "Fresno State"
SERIES_TICKER = "KXNCAAMBGAME"  # from your UI URL; can set to None to search all open markets
MAX_PRINT_CANDIDATES = 12       # how many candidate lines to print

# =========================
# LOAD .env
# =========================
load_dotenv()

ENV = os.getenv("KALSHI_ENV", "PROD").upper()
BASE_URL = "https://demo-api.kalshi.co" if ENV == "DEMO" else "https://api.elections.kalshi.com"

API_KEY_ID = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
PRIVATE_KEY_PATH = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()


# =========================
# AUTH HELPERS
# =========================
def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def sign_request(private_key, ts_ms: str, method: str, path: str) -> str:
    # IMPORTANT: sign path without query params
    path_no_q = path.split("?")[0]
    msg = f"{ts_ms}{method.upper()}{path_no_q}".encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")

def auth_get(private_key, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ts_ms = str(int(time.time() * 1000))
    sig = sign_request(private_key, ts_ms, "GET", path)
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }
    r = requests.get(BASE_URL + path, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def public_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(BASE_URL + path, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# =========================
# MARKET SEARCH
# =========================
def list_open_markets(private_key) -> List[Dict[str, Any]]:
    """
    Pull open markets. If SERIES_TICKER is set, use it; otherwise pull all open.
    Note: /markets is paginated; we iterate cursors until exhausted (simple + safe).
    """
    markets: List[Dict[str, Any]] = []
    cursor = None

    while True:
        params: Dict[str, Any] = {"status": "open", "limit": 200}
        if SERIES_TICKER:
            params["series_ticker"] = SERIES_TICKER
        if cursor:
            params["cursor"] = cursor

        resp = auth_get(private_key, "/trade-api/v2/markets", params=params)
        batch = resp.get("markets", []) or []
        markets.extend(batch)

        cursor = resp.get("cursor")
        if not cursor:
            break

    return markets

def market_text(m: Dict[str, Any]) -> str:
    return f"{m.get('title','')} {m.get('subtitle','')}".strip()

def score_candidate(m: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Heuristic scoring:
    - Must mention TEAM_PRIMARY somewhere (title/subtitle)
    - Bonus if mentions TEAM_OPP
    - Bonus if title looks like a team-specific "Will X win?" style (contains TEAM_PRIMARY)
    Returns a tuple used for sorting (higher is better).
    """
    txt = market_text(m).lower()
    has_primary = int(TEAM_PRIMARY in txt)
    has_opp = int(TEAM_OPP in txt)
    # "will unlv win" / "unlv wins" often indicates the specific side market
    title = (m.get("title") or "").lower()
    title_primary = int(TEAM_PRIMARY in title)
    return (has_primary, has_opp, title_primary)

def find_best_market(markets: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return best match and a list of top candidates (for printing).
    """
    # Filter anything mentioning primary team anywhere
    candidates = [m for m in markets if TEAM_PRIMARY in market_text(m).lower()]
    if not candidates:
        return None, []

    # Sort by our heuristic, then by soonest close_time (earlier = usually tonight)
    candidates.sort(
        key=lambda m: (
            score_candidate(m)[0],
            score_candidate(m)[1],
            score_candidate(m)[2],
            # earlier close times are "sooner" games; lexical ISO sort is ok
            # but we want earliest first, so use as secondary sort after score by reversing score later
            m.get("close_time", ""),
        ),
        reverse=True,
    )

    # After reversing, close_time might now be in reverse direction; we’ll refine:
    top = candidates[:MAX_PRINT_CANDIDATES]

    # Choose best by: highest score, and among equal scores pick earliest close_time
    def key_best(m):
        s = score_candidate(m)
        return (-s[0], -s[1], -s[2], m.get("close_time", "9999"))

    best = sorted(candidates, key=key_best)[0]
    return best, top


# =========================
# ORDERBOOK PARSING
# =========================
def best_bid(levels: List[List[int]]) -> Optional[int]:
    """
    Orderbook levels are [[price, qty], ...], sorted ascending by price.
    Best bid is the last element's price.
    """
    if not levels:
        return None
    return int(levels[-1][0])

def print_orderbook_snapshot(ticker: str):
    ob = public_get(f"/trade-api/v2/markets/{ticker}/orderbook")
    yes = ob.get("orderbook", {}).get("yes", []) or []
    no = ob.get("orderbook", {}).get("no", []) or []

    best_yes = best_bid(yes)
    best_no = best_bid(no)

    print("\nORDERBOOK SNAPSHOT")
    print(f"Best YES bid: {best_yes}¢")
    print(f"Best NO bid:  {best_no}¢")
    if best_no is not None:
        print(f"Implied YES ask: {100 - best_no}¢")
    if best_yes is not None:
        print(f"Implied NO ask:  {100 - best_yes}¢")

    # Quick spread estimate (YES side)
    if best_yes is not None and best_no is not None:
        yes_ask = 100 - best_no
        spread = yes_ask - best_yes
        print(f"Implied YES spread: {spread}¢")


def main():
    if not API_KEY_ID:
        raise RuntimeError("Missing KALSHI_API_KEY_ID (set in .env)")
    if not PRIVATE_KEY_PATH:
        raise RuntimeError("Missing KALSHI_PRIVATE_KEY_PATH (set in .env)")
    if not os.path.exists(PRIVATE_KEY_PATH):
        raise RuntimeError(f"Private key file not found: {PRIVATE_KEY_PATH}")

    private_key = load_private_key(PRIVATE_KEY_PATH)

    print(f"\nENV: {ENV}")
    print(f"BASE_URL: {BASE_URL}")
    print(f"Series filter: {SERIES_TICKER or '(none)'}")
    print(f"Looking for: primary='{TEAM_PRIMARY}', opponent hint='{TEAM_OPP}'\n")

    markets = list_open_markets(private_key)
    print(f"Fetched {len(markets)} open markets.\n")

    best, top = find_best_market(markets)

    if not best:
        print("❌ No markets found mentioning UNLV in title/subtitle.")
        print("Tip: set SERIES_TICKER=None to search across ALL open markets.")
        return

    print("TOP CANDIDATES (debug):")
    for m in top:
        txt = market_text(m)
        s = score_candidate(m)
        print(f"— ticker={m.get('ticker')} | score={s} | closes={m.get('close_time')} | {txt}")

    print("\n✅ SELECTED MARKET")
    print(f"Ticker: {best.get('ticker')}")
    print(f"Title:  {best.get('title')}")
    print(f"Subtitle: {best.get('subtitle')}")
    print(f"Closes: {best.get('close_time')}")

    print_orderbook_snapshot(best["ticker"])


if __name__ == "__main__":
    main()
