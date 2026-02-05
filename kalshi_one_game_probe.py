# kalshi_one_game_probe.py
#
# Single-game strategy for UNLV @ Fresno St. "Winner?" market.
#
# Strategy:
# - Model: UNLV win prob = 51%, Fresno = 49%.
# - Fair YES ≈ 51c, fair NO ≈ 49c.
# - We:
#   1) Open a small directional position (1 contract) ONLY when market price is
#      significantly mispriced vs our fair (strong edge, not tiny).
#   2) If later we can buy the opposite side so YES+NO total cost <= 100 - MIN_LOCKED_PROFIT_CENTS,
#      we lock in a risk-free pair and hold to settlement.
#   3) Track:
#        - Locked pair profit (at settlement, risk-free)
#        - Realized P/L from stop-outs
#        - MTM P/L on any open position
#   4) Risk control for a ~$10 account:
#        - Only 1 open directional position at a time (size=1)
#        - At most MAX_PAIRS pairs
#        - Soft collateral cap
#        - STOP LOSS: if MTM loss on open pos ≥ STOP_LOSS_CENTS, we sell out at bid,
#          record realized loss, and NEVER trade that side again in this game
#
# NOTE: This is a toy strategy for experimentation only.

import os
import time
import json
import base64
import datetime as dt
from typing import Dict, Any, Optional, List, Tuple, Set

import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import csv


# =========================
# CONFIG + .env LOADING
# =========================

# Load .env just like verify_unlv_market.py
load_dotenv()

# These env vars should be set in .env:
#   KALSHI_ENV=PROD or DEMO
#   KALSHI_API_KEY_ID=...
#   KALSHI_PRIVATE_KEY_PATH=C:\Users\cmpea\Kalshi\kalshi_private_key.pem
ENV = (os.getenv("KALSHI_ENV") or "DEMO").upper()  # DEMO or PROD

BASE_URL = "https://demo-api.kalshi.co" if ENV == "DEMO" else "https://api.elections.kalshi.com"
API_KEY_ID = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
PRIVATE_KEY_PATH = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()

# Market search: men's CBB game series + team hints
SERIES_TICKER = "KXNCAAMBGAME"
TEAM_A = "UNLV"
TEAM_B = "Fresno"  # "Fresno St" / "Fresno State" should match

# Model: UNLV 51%, Fresno 49%
P_MODEL_UNLV = 0.51
FAIR_YES_CENTS = int(round(P_MODEL_UNLV * 100))           # 51
FAIR_NO_CENTS = int(round((1.0 - P_MODEL_UNLV) * 100))    # 49

# Trading logic parameters
# How mispriced must the market be vs our fair before we open a position?
# e.g. with FAIR_NO=49c and ENTRY_EDGE=4, we can buy NO at prices <= 45c
ENTRY_EDGE_CENTS = 4

# Minimum locked profit per YES+NO pair (risk-free at settlement)
MIN_LOCKED_PROFIT_CENTS = 6     # e.g. YES+NO total <= 94c → 6c/pair ≈ 12% on ~50c

MAX_PAIRS = 3                   # max number of juicy pairs to accumulate

# Stop-loss: if MTM loss on open position reaches this (in cents), exit and never trade that side again
STOP_LOSS_CENTS = 15            # e.g. bought YES @ 55c; if bid drops to 40c → -15c, we stop out

# Take-profit: if MTM gain on open position reaches this (in cents), exit and bank the profit
TAKE_PROFIT_CENTS = 8           # e.g. buy at 24, sell at 32 → +8c

# Bankroll control for ~$10 account
BANKROLL_DOLLARS = 10.0
MAX_COLLATERAL_DOLLARS = 8.0    # don't tie up more than ~$8 in positions/pairs

# Order sizing: always 1 contract at a time for now
ORDER_QTY = 1

# Timing parameters
ENTRY_TIMEOUT_SECS = 20         # how long to wait for order fill before giving up
POLL_LOOP_SECS = 6              # main loop sleep between checks
STOP_TRADING_BEFORE_CLOSE_SECS = 60  # stop opening/locking this long before close


# =========================
# AUTH + HTTP HELPERS
# (match verify_unlv_market.py)
# =========================

def _now_ms() -> str:
    return str(int(time.time() * 1000))


def _sign_request(private_key, ts_ms: str, method: str, path: str) -> str:
    """
    Sign the path WITHOUT query params using PSS, exactly like verify_unlv_market.py.
    """
    path_no_q = path.split("?")[0]
    msg = f"{ts_ms}{method.upper()}{path_no_q}".encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def _load_private_key(path: str):
    with open(path, "rb") as f:
        data = f.read()
    return serialization.load_pem_private_key(data, password=None)


def _auth_headers(private_key, method: str, path: str) -> Dict[str, str]:
    """
    Use the same header names as verify_unlv_market.py:
      - KALSHI-ACCESS-KEY
      - KALSHI-ACCESS-TIMESTAMP
      - KALSHI-ACCESS-SIGNATURE
    """
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
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {path} failed: {resp.status_code} {resp.text}")
    return resp.json()


def _post(private_key, path: str, body: Dict[str, Any]) -> Any:
    url = BASE_URL + path
    headers = _auth_headers(private_key, "POST", path)
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"POST {path} failed: {resp.status_code} {resp.text}")
    return resp.json()


def _delete(private_key, path: str) -> Any:
    url = BASE_URL + path
    headers = _auth_headers(private_key, "DELETE", path)
    resp = requests.delete(url, headers=headers, timeout=20)
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"DELETE {path} failed: {resp.status_code} {resp.text}")
    return resp.text and resp.json() or {}


# =========================
# UTIL
# =========================

def parse_iso(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def print_status(msg: str) -> None:
    now = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def init_log_file() -> str:
    """
    Create a timestamped CSV log file in a local 'logs' directory and
    write the header row. Returns the full path to the log file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"kalshi_unlv_probe_{timestamp}.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t_utc",
                "ticker",
                "yes_bid_c",
                "yes_ask_c",
                "no_bid_c",
                "no_ask_c",
                "open_side",
                "open_price_c",
                "open_qty",
                "pairs_count",
                "locked_pnl_c",
                "realized_pnl_c",
                "mtm_pnl_c",
                "total_pnl_c",
                "stopped_out_yes",
                "stopped_out_no",
                "secs_to_close",
            ]
        )

    print_status(f"Logging snapshots to: {log_path}")
    return log_path


def append_log_row(
    log_path: str,
    ticker: str,
    prices: Dict[str, Optional[float]],
    state: "StrategyState",
    locked_cents: int,
    mtm_cents: int,
    total_cents: int,
    secs_to_close: float,
) -> None:
    """
    Append one snapshot row to the CSV log.
    """
    t_utc = utc_now().isoformat()

    yes_bid = prices.get("best_yes_bid")
    yes_ask = prices.get("imp_yes_ask")
    no_bid = prices.get("best_no_bid")
    no_ask = prices.get("imp_no_ask")

    open_side = state.open_side or ""
    open_price = state.open_price if state.open_price is not None else ""
    open_qty = state.open_qty

    stopped_out_yes = "yes" in state.stop_out_sides
    stopped_out_no = "no" in state.stop_out_sides

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                t_utc,
                ticker,
                yes_bid if yes_bid is not None else "",
                yes_ask if yes_ask is not None else "",
                no_bid if no_bid is not None else "",
                no_ask if no_ask is not None else "",
                open_side,
                open_price,
                open_qty,
                len(state.pairs),
                locked_cents,
                state.realized_pnl_cents,
                mtm_cents,
                total_cents,
                int(stopped_out_yes),
                int(stopped_out_no),
                int(secs_to_close),
            ]
        )



# =========================
# MARKET DISCOVERY
# =========================

def find_unlv_market(private_key) -> Tuple[str, Dict[str, Any]]:
    """
    Find the UNLV @ Fresno St men's CBB winner market.

    Mirrors verify_unlv_market.py:
    - No min/max close_ts filters (search ALL open markets in the series)
    - Score candidates based on mentions of UNLV and Fresno in title/subtitle
    - Pick the best-scoring candidate
    """
    TEAM_PRIMARY = TEAM_A.lower()   # "unlv"
    TEAM_OPP = TEAM_B.lower()       # "fresno"

    def market_text(m: Dict[str, Any]) -> str:
        return f"{m.get('title', '')} {m.get('subtitle', '')}".strip()

    def score_candidate(m: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Heuristic scoring:
        - Must mention TEAM_PRIMARY somewhere (title/subtitle)
        - Bonus if mentions TEAM_OPP
        - Bonus if title looks like a team-specific "UNLV at Fresno St. Winner?" type
        """
        txt = market_text(m).lower()
        has_primary = int(TEAM_PRIMARY in txt)
        has_opp = int(TEAM_OPP in txt)
        title = (m.get("title") or "").lower()
        title_primary = int(TEAM_PRIMARY in title)
        return (has_primary, has_opp, title_primary)

    # Pull ALL open markets in this series (no time filters)
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {"status": "open", "limit": 200}
        if SERIES_TICKER:
            params["series_ticker"] = SERIES_TICKER
        if cursor:
            params["cursor"] = cursor

        resp = _get(private_key, "/trade-api/v2/markets", params=params)
        batch = resp.get("markets", []) or []
        markets.extend(batch)

        cursor = resp.get("cursor")
        if not cursor:
            break

    if not markets:
        raise RuntimeError("No open markets returned for this series_ticker.")

    # Filter to markets that at least mention UNLV somewhere
    candidates = [m for m in markets if TEAM_PRIMARY in market_text(m).lower()]
    if not candidates:
        raise RuntimeError("Could not find any open markets mentioning UNLV in title/subtitle.")

    # Sort by heuristic score, highest first
    candidates.sort(key=score_candidate, reverse=True)
    chosen = candidates[0]

    print_status(
        f"Found market: {chosen.get('ticker')} | "
        f"{chosen.get('title')} | closes={chosen.get('close_time')}"
    )

    return chosen["ticker"], chosen


# =========================
# ORDERBOOK HELPERS
# =========================

def fetch_orderbook(ticker: str) -> Dict[str, Any]:
    # public orderbook endpoint
    url = BASE_URL + f"/trade-api/v2/markets/{ticker}/orderbook"
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Orderbook fetch failed: {resp.status_code} {resp.text}")
    return resp.json()


def best_bid(ob: Dict[str, Any], side: str) -> Optional[int]:
    # ob["orderbook"]["yes"] / ["no"] are arrays of [price, qty] sorted ascending
    levels = (ob.get("orderbook") or {}).get(side, []) or []
    if not levels:
        return None
    return int(levels[-1][0])  # last = highest price


def derive_prices(ob: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Derive best YES/NO bids and asks using binary complement:
      yes_ask = 100 - best_no_bid
      no_ask  = 100 - best_yes_bid
    """
    by = best_bid(ob, "yes")
    bn = best_bid(ob, "no")

    if by is None or bn is None:
        return {
            "best_yes_bid": by,
            "best_no_bid": bn,
            "imp_yes_ask": None,
            "imp_no_ask": None,
            "mid_yes": None,
            "mid_no": None,
        }

    imp_yes_ask = 100 - bn
    imp_no_ask = 100 - by
    mid_yes = (by + imp_yes_ask) / 2.0
    mid_no = (bn + imp_no_ask) / 2.0

    return {
        "best_yes_bid": by,
        "best_no_bid": bn,
        "imp_yes_ask": imp_yes_ask,
        "imp_no_ask": imp_no_ask,
        "mid_yes": mid_yes,
        "mid_no": mid_no,
    }


# =========================
# ORDER HELPERS
# =========================

def place_limit_buy(private_key, ticker: str, side: str, price_cents: int, qty: int) -> str:
    """
    Places a limit BUY order on YES or NO.
    Returns Kalshi order_id.
    """
    body = {
        "ticker": ticker,
        "type": "limit",
        "action": "buy",
        "side": side,     # "yes" or "no"
        "count": int(qty),
    }
    if side == "yes":
        body["yes_price"] = int(price_cents)
    else:
        body["no_price"] = int(price_cents)

    resp = _post(private_key, "/trade-api/v2/portfolio/orders", body)
    order = resp.get("order", {})
    oid = order.get("order_id")
    if not oid:
        raise RuntimeError(f"Create buy order failed: {resp}")
    return oid


def place_limit_sell(private_key, ticker: str, side: str, price_cents: int, qty: int) -> str:
    """
    Places a limit SELL order on YES or NO (to exit a long position).
    Returns order_id.
    """
    body = {
        "ticker": ticker,
        "type": "limit",
        "action": "sell",
        "side": side,
        "count": int(qty),
    }
    if side == "yes":
        body["yes_price"] = int(price_cents)
    else:
        body["no_price"] = int(price_cents)

    resp = _post(private_key, "/trade-api/v2/portfolio/orders", body)
    order = resp.get("order", {})
    oid = order.get("order_id")
    if not oid:
        raise RuntimeError(f"Create sell order failed: {resp}")
    return oid


def get_order(private_key, order_id: str) -> Dict[str, Any]:
    return _get(private_key, f"/trade-api/v2/portfolio/orders/{order_id}")


def cancel_order(private_key, order_id: str) -> None:
    _delete(private_key, f"/trade-api/v2/portfolio/orders/{order_id}")


def wait_for_fill_or_timeout(
    private_key,
    order_id: str,
    max_wait_secs: int = ENTRY_TIMEOUT_SECS,
    poll_secs: int = 2,
) -> int:
    """
    Poll fills for this order until it fills or times out.
    Returns total filled count for the given order_id.

    We avoid GET /portfolio/orders/{order_id} (which can 404)
    and instead rely on GET /portfolio/fills?order_id=... which always
    returns 200 with a list of fills (possibly empty).
    """
    start = time.time()
    last_fill = 0

    while True:
        # Query fills filtered by this order_id
        try:
            resp = _get(
                private_key,
                "/trade-api/v2/portfolio/fills",
                params={"order_id": order_id, "limit": 50},
            )
        except RuntimeError as e:
            print_status(f"Error fetching fills for order {order_id}: {e}")
            resp = {"fills": []}

        fills = resp.get("fills", []) or []

        # Sum up fill counts (for our usage this will be 0 or 1)
        filled = 0
        for f in fills:
            try:
                filled += int(f.get("count", 0))
            except (TypeError, ValueError):
                continue

        if filled != last_fill:
            print_status(f"Order {order_id}: filled={filled}")
            last_fill = filled

        if filled > 0:
            # For our 1-lot orders, any positive fill means we're done.
            return filled

        if time.time() - start >= max_wait_secs:
            print_status(f"Order {order_id}: timeout, cancelling.")
            try:
                cancel_order(private_key, order_id)
            except Exception as e:
                print_status(f"Cancel error (ignored): {e}")
            return 0

        time.sleep(poll_secs)


# =========================
# STATE CONTAINER
# =========================

class StrategyState:
    """
    Track open unpaired position, locked pairs, realized P/L, and per-side stop-outs.
    """

    def __init__(self):
        # One open unpaired position max
        self.open_side: Optional[str] = None   # "yes" or "no"
        self.open_price: Optional[int] = None  # cents
        self.open_qty: int = 0

        # Locked pairs: list of (yes_price_cents, no_price_cents, qty)
        self.pairs: List[Tuple[int, int, int]] = []

        # Realized P/L in cents from stop-outs & take-profits (can be positive or negative)
        self.realized_pnl_cents: int = 0

        # Sides we have stopped out of; we won't open new positions on them
        self.stop_out_sides: Set[str] = set()

    def collateral_used_dollars(self) -> float:
        """
        Approximate collateral usage:
        - Each unpaired position ~1.0
        - Each pair ~1.0
        """
        used = 0.0
        if self.open_side and self.open_qty > 0:
            used += 1.0
        used += len(self.pairs) * 1.0
        return used

    def locked_profit_cents(self) -> int:
        """
        Sum of risk-free profit from all locked pairs at settlement.
        For each pair (py, pn): profit = 100 - py - pn.
        """
        total = 0
        for (py, pn, qty) in self.pairs:
            total += qty * (100 - py - pn)
        return total

    def mark_to_market_cents(self, best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> int:
        """
        Rough mark-to-market for any open one-sided position.
        - If we are long YES at open_price: PnL ≈ (best_yes_bid - open_price) * qty
        - If long NO: similarly with NO.
        """
        if not self.open_side or self.open_price is None or self.open_qty <= 0:
            return 0

        if self.open_side == "yes":
            if best_yes_bid is None:
                return 0
            return (best_yes_bid - self.open_price) * self.open_qty
        else:
            if best_no_bid is None:
                return 0
            return (best_no_bid - self.open_price) * self.open_qty

    def total_pnl_cents(self, best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> int:
        return self.locked_profit_cents() + self.realized_pnl_cents + self.mark_to_market_cents(best_yes_bid, best_no_bid)


# =========================
# STRATEGY LOGIC
# =========================

def maybe_open_new_position(
    private_key,
    ticker: str,
    state: StrategyState,
    prices: Dict[str, Optional[float]],
) -> None:
    """
    If flat (no open one-sided position) and not at collateral/pair limits,
    look for an undervalued side and buy 1 contract.

    Logic:
    - Use the side's ASK price derived from the opposite bid:
        YES_ask = 100 - best_no_bid
        NO_ask  = 100 - best_yes_bid
    - Place a BUY at that ask (or not at all), as long as this buy price is
      still cheap vs our fair (by ENTRY_EDGE_CENTS).
    - This means we are using the price people are actually selling for.
    """
    if state.open_side is not None:
        return  # already have an open unpaired position

    if len(state.pairs) >= MAX_PAIRS:
        return  # don't add more riskless pairs than we planned

    if state.collateral_used_dollars() >= MAX_COLLATERAL_DOLLARS:
        return  # don't overuse small bankroll

    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]
    imp_yes_ask = prices["imp_yes_ask"]
    imp_no_ask = prices["imp_no_ask"]

    chosen_side: Optional[str] = None
    chosen_price: Optional[int] = None

    max_yes_price = FAIR_YES_CENTS - ENTRY_EDGE_CENTS
    max_no_price = FAIR_NO_CENTS - ENTRY_EDGE_CENTS

    # Prefer NO first (that's what we're focusing on), then YES if NO isn't attractive.
    # --- Try NO first ---
    if (
        imp_no_ask is not None
        and "no" not in state.stop_out_sides
    ):
        ask_no_price = int(imp_no_ask)
        if ask_no_price <= max_no_price:
            chosen_side = "no"
            chosen_price = max(1, min(ask_no_price, 99))  # keep внутри 1..99

    # --- If we didn't choose NO, consider YES symmetrically ---
    if (
        chosen_side is None
        and imp_yes_ask is not None
        and "yes" not in state.stop_out_sides
    ):
        ask_yes_price = int(imp_yes_ask)
        if ask_yes_price <= max_yes_price:
            chosen_side = "yes"
            chosen_price = max(1, min(ask_yes_price, 99))

    if chosen_side is None or chosen_price is None:
        # Nothing is cheap enough right now
        return

    print_status(
        f"Opening position: BUY {chosen_side.upper()} 1 @ {chosen_price}c "
        f"(fair: YES={FAIR_YES_CENTS}c NO={FAIR_NO_CENTS}c, "
        f"best_yes_bid={best_yes_bid}c best_no_bid={best_no_bid}c, "
        f"yes_ask={imp_yes_ask}c no_ask={imp_no_ask}c)"
    )

    order_id = place_limit_buy(private_key, ticker, chosen_side, chosen_price, ORDER_QTY)
    filled = wait_for_fill_or_timeout(private_key, order_id, ENTRY_TIMEOUT_SECS, 2)

    if filled >= 1:
        state.open_side = chosen_side
        state.open_price = chosen_price
        state.open_qty = filled
        print_status(f"Open position established: {chosen_side.upper()} x{filled} @ {chosen_price}c")
    else:
        print_status("Open attempt did not fill, staying flat.")


def maybe_lock_pair(private_key, ticker: str, state: StrategyState, prices: Dict[str, Optional[float]]) -> None:
    """
    If we have an open one-sided position (YES or NO),
    see if we can buy the opposite side cheaply enough to lock
    risk-free profit >= MIN_LOCKED_PROFIT_CENTS per contract.
    """
    if state.open_side is None or state.open_price is None or state.open_qty <= 0:
        return

    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]
    imp_yes_ask = prices["imp_yes_ask"]
    imp_no_ask = prices["imp_no_ask"]

    if best_yes_bid is None or best_no_bid is None or imp_yes_ask is None or imp_no_ask is None:
        return

    open_side = state.open_side
    py = state.open_price
    qty = state.open_qty

    if open_side == "yes":
        # We already own YES at py.
        # To lock a pair, we need to buy NO at pn such that:
        #   py + pn <= 100 - MIN_LOCKED_PROFIT_CENTS
        max_no_price = 100 - MIN_LOCKED_PROFIT_CENTS - py
        if max_no_price <= 0:
            return
        if imp_no_ask <= max_no_price:
            target_price = int(min(imp_no_ask, max_no_price))
            print_status(
                f"Locking pair: already LONG YES @ {py}c, buying NO @ {target_price}c "
                f"to lock ≥{MIN_LOCKED_PROFIT_CENTS}c/contract."
            )
            order_id = place_limit_buy(private_key, ticker, "no", target_price, qty)
            filled = wait_for_fill_or_timeout(private_key, order_id, ENTRY_TIMEOUT_SECS, 2)
            if filled >= 1:
                state.pairs.append((py, target_price, filled))
                state.open_side = None
                state.open_price = None
                state.open_qty = 0
                print_status(
                    f"Pair locked: YES {py}c + NO {target_price}c, qty={filled} "
                    f"→ profit at settlement = {100 - py - target_price}c/contract."
                )
    else:
        # We own NO at py, want to buy YES to complete pair.
        max_yes_price = 100 - MIN_LOCKED_PROFIT_CENTS - py
        if max_yes_price <= 0:
            return
        if imp_yes_ask <= max_yes_price:
            target_price = int(min(imp_yes_ask, max_yes_price))
            print_status(
                f"Locking pair: already LONG NO @ {py}c, buying YES @ {target_price}c "
                f"to lock ≥{MIN_LOCKED_PROFIT_CENTS}c/contract."
            )
            order_id = place_limit_buy(private_key, ticker, "yes", target_price, qty)
            filled = wait_for_fill_or_timeout(private_key, order_id, ENTRY_TIMEOUT_SECS, 2)
            if filled >= 1:
                state.pairs.append((target_price, py, filled))
                state.open_side = None
                state.open_price = None
                state.open_qty = 0
                print_status(
                    f"Pair locked: YES {target_price}c + NO {py}c, qty={filled} "
                    f"→ profit at settlement = {100 - target_price - py}c/contract."
                )


def maybe_stop_out(private_key, ticker: str, state: StrategyState, prices: Dict[str, Optional[float]]) -> None:
    """
    If our open position is losing more than STOP_LOSS_CENTS (MTM),
    sell it at the current bid, record realized loss, and ban that side
    from further entries this game.
    """
    if state.open_side is None or state.open_price is None or state.open_qty <= 0:
        return

    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]

    mtm = state.mark_to_market_cents(best_yes_bid, best_no_bid)

    # If MTM loss is at least STOP_LOSS_CENTS per contract, exit.
    if mtm > -STOP_LOSS_CENTS * state.open_qty:
        return

    side = state.open_side
    px_open = state.open_price
    qty = state.open_qty

    # Choose exit price: hit current bid on that side.
    if side == "yes":
        if best_yes_bid is None:
            return
        exit_px = best_yes_bid
    else:
        if best_no_bid is None:
            return
        exit_px = best_no_bid

    print_status(
        f"STOP-LOSS: exiting {side.upper()} x{qty} opened @ {px_open}c; "
        f"selling @ {exit_px}c (MTM={mtm}c <= -{STOP_LOSS_CENTS}c)"
    )

    order_id = place_limit_sell(private_key, ticker, side, exit_px, qty)
    filled = wait_for_fill_or_timeout(private_key, order_id, ENTRY_TIMEOUT_SECS, 2)

    if filled > 0:
        realized = (exit_px - px_open) * filled
        state.realized_pnl_cents += realized
        print_status(
            f"STOP-LOSS filled: {side.upper()} x{filled} @ {exit_px}c, "
            f"realized P/L={realized}c"
        )
        # Clear position and ban this side from future entries
        state.open_side = None
        state.open_price = None
        state.open_qty = 0
        state.stop_out_sides.add(side)
    else:
        print_status("STOP-LOSS sell did not fill; keeping position for now.")


def maybe_take_profit(
    private_key,
    ticker: str,
    state: StrategyState,
    prices: Dict[str, Optional[float]],
) -> None:
    """
    If our open position is up at least TAKE_PROFIT_CENTS (MTM),
    sell it at the current bid, realize the gain, and go flat.
    This lets us "trade around" the swings: buy low, sell high, repeat.
    """
    if state.open_side is None or state.open_price is None or state.open_qty <= 0:
        return

    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]

    mtm = state.mark_to_market_cents(best_yes_bid, best_no_bid)

    # Require MTM profit ≥ TAKE_PROFIT_CENTS per contract
    if mtm < TAKE_PROFIT_CENTS * state.open_qty:
        return

    side = state.open_side
    px_open = state.open_price
    qty = state.open_qty

    # Exit at current bid on that side
    if side == "yes":
        if best_yes_bid is None:
            return
        exit_px = best_yes_bid
    else:
        if best_no_bid is None:
            return
        exit_px = best_no_bid

    print_status(
        f"TAKE-PROFIT: exiting {side.upper()} x{qty} opened @ {px_open}c; "
        f"selling @ {exit_px}c (MTM={mtm}c ≥ {TAKE_PROFIT_CENTS}c)"
    )

    order_id = place_limit_sell(private_key, ticker, side, exit_px, qty)
    filled = wait_for_fill_or_timeout(private_key, order_id, ENTRY_TIMEOUT_SECS, 2)

    if filled > 0:
        realized = (exit_px - px_open) * filled
        state.realized_pnl_cents += realized
        print_status(
            f"TAKE-PROFIT filled: {side.upper()} x{filled} @ {exit_px}c, "
            f"realized P/L={realized}c"
        )
        # Clear the position; we do NOT ban this side, so we can trade it again later
        state.open_side = None
        state.open_price = None
        state.open_qty = 0
    else:
        print_status("TAKE-PROFIT sell did not fill; keeping position for now.")


# =========================
# MAIN
# =========================

def main():
    if not API_KEY_ID:
        raise SystemExit("Missing KALSHI_API_KEY_ID (set in .env)")
    if not PRIVATE_KEY_PATH:
        raise SystemExit("Missing KALSHI_PRIVATE_KEY_PATH (set in .env)")
    if not os.path.exists(PRIVATE_KEY_PATH):
        raise SystemExit(f"Private key file not found: {PRIVATE_KEY_PATH}")

    private_key = _load_private_key(PRIVATE_KEY_PATH)

    print_status(f"ENV={ENV} BASE_URL={BASE_URL}")
    print_status(f"Model: UNLV win prob = {P_MODEL_UNLV:.3f} → fair YES={FAIR_YES_CENTS}c, NO={FAIR_NO_CENTS}c")
    print_status("Finding UNLV @ Fresno St market...")

    ticker, market = find_unlv_market(private_key)
    close_time = parse_iso(market["close_time"])

    print_status(f"Selected market: {ticker} | {market.get('title')} | closes={market['close_time']}")

    state = StrategyState()

    # Initialize CSV logging for this run
    log_path = init_log_file()


    # Main loop: run until just before close
    while True:
        now = utc_now()
        secs_to_close = (close_time - now).total_seconds()

        if secs_to_close <= STOP_TRADING_BEFORE_CLOSE_SECS:
            print_status("Within stop-trading window before close. Ending trading loop.")
            break

        # Fetch orderbook and derive prices
        ob = fetch_orderbook(ticker)
        prices = derive_prices(ob)

        best_yes_bid = prices["best_yes_bid"]
        best_no_bid = prices["best_no_bid"]
        imp_yes_ask = prices["imp_yes_ask"]
        imp_no_ask = prices["imp_no_ask"]

        locked_cents = state.locked_profit_cents()
        mtm_cents = state.mark_to_market_cents(best_yes_bid, best_no_bid)
        total_cents = state.total_pnl_cents(best_yes_bid, best_no_bid)

        print_status(
            f"Market snapshot: YES_bid={best_yes_bid}c YES_ask={imp_yes_ask}c "
            f"NO_bid={best_no_bid}c NO_ask={imp_no_ask}c | "
            f"Pairs={len(state.pairs)} | Open={state.open_side or 'none'} "
            f"(qty={state.open_qty}, px={state.open_price}) | "
            f"P/L (locked={locked_cents}c, realized={state.realized_pnl_cents}c, "
            f"mtm={mtm_cents}c, total={total_cents}c)"
        )

        # Log this snapshot to CSV for offline analysis
        append_log_row(
            log_path=log_path,
            ticker=ticker,
            prices=prices,
            state=state,
            locked_cents=locked_cents,
            mtm_cents=mtm_cents,
            total_cents=total_cents,
            secs_to_close=secs_to_close,
        )


        # 1) Stop-loss check first — if the game is running away against us, cut it.
        maybe_stop_out(private_key, ticker, state, prices)

        # 2) Take-profit: if we're up enough on the open pos, cash it and go flat.
        maybe_take_profit(private_key, ticker, state, prices)

        # 3) Try to lock in a pair if we still have an open position
        maybe_lock_pair(private_key, ticker, state, prices)

        # 4) If flat, maybe open a new position on the side that's cheap vs our 51/49,
        #    unless we've previously stopped-out that side.
        if state.open_side is None:
            maybe_open_new_position(private_key, ticker, state, prices)

        time.sleep(POLL_LOOP_SECS)

    # Final status at end of loop (pre-settlement)
    ob = fetch_orderbook(ticker)
    prices = derive_prices(ob)
    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]
    locked_cents = state.locked_profit_cents()
    mtm_cents = state.mark_to_market_cents(best_yes_bid, best_no_bid)
    total_cents = state.total_pnl_cents(best_yes_bid, best_no_bid)

    print("\n=== FINAL STRATEGY SNAPSHOT (pre-settlement) ===")
    print(f"Market: {ticker} | {market.get('title')}")
    print(f"Close time (UTC): {market['close_time']}")
    print(f"Pairs locked: {len(state.pairs)}")
    for i, (py, pn, qty) in enumerate(state.pairs, start=1):
        print(f"  Pair {i}: YES {py}c + NO {pn}c, qty={qty}, locked={100 - py - pn}c/contract")

    print(f"Open position: side={state.open_side} qty={state.open_qty} px={state.open_price}")
    print(f"Locked pair profit: {locked_cents}c = ${locked_cents/100:.2f}")
    print(f"Realized P/L from stop-outs: {state.realized_pnl_cents}c = ${state.realized_pnl_cents/100:.2f}")
    print(f"Open MTM P/L: {mtm_cents}c = ${mtm_cents/100:.2f}")
    print(f"Total P/L (pre-fees, pre-settlement): {total_cents}c = ${total_cents/100:.2f}")
    print("NOTE: Pairs are risk-free at settlement (ignoring fees). "
          "Stop-outs cap losses on runaway games, and we ban that side afterward.")


if __name__ == "__main__":
    main()
