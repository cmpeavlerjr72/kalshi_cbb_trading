# kalshi_one_game_probe.py
# Improved version - tuned for real Kalshi mid-major CBB trading (fees, liquidity, edge bias)

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

load_dotenv()

ENV = (os.getenv("KALSHI_ENV") or "DEMO").upper()
BASE_URL = "https://demo-api.kalshi.co" if ENV == "DEMO" else "https://api.elections.kalshi.com"
API_KEY_ID = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
PRIVATE_KEY_PATH = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()

SERIES_TICKER = "KXNCAAMBGAME"
TEAM_A = "UNLV"
TEAM_B = "Fresno"

# === YOUR MODEL ===
P_MODEL_UNLV = 0.51
FAIR_YES_CENTS = int(round(P_MODEL_UNLV * 100))   # 51
FAIR_NO_CENTS = int(round((1.0 - P_MODEL_UNLV) * 100))  # 49

# === TUNED PARAMETERS (recommended for live trading) ===
ENTRY_EDGE_CENTS = 5
MIN_LOCKED_PROFIT_CENTS = 8      # ~6-8¢ net after fees
MAX_PAIRS = 4
STOP_LOSS_CENTS = 20
TAKE_PROFIT_CENTS = 12
BANKROLL_DOLLARS = 10.0
MAX_COLLATERAL_DOLLARS = 7.0
MIN_LIQUIDITY_AT_LEVEL = 1        # min contracts at target price or better

STRONG_SIDE = "yes"               # UNLV = YES (your model lean)
WEAK_SIDE_EDGE_BUFFER = 21        # extra caution on Fresno side

ORDER_QTY_BASE = 1
ENTRY_TIMEOUT_SECS = 20
POLL_LOOP_SECS = 5
STOP_TRADING_BEFORE_CLOSE_SECS = 300  # 5 minutes


# =========================
# AUTH + HELPERS (unchanged)
# =========================
def _now_ms(): return str(int(time.time() * 1000))

def _sign_request(private_key, ts_ms: str, method: str, path: str) -> str:
    path_no_q = path.split("?")[0]
    msg = f"{ts_ms}{method.upper()}{path_no_q}".encode("utf-8")
    sig = private_key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return base64.b64encode(sig).decode("utf-8")

def _load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def _auth_headers(private_key, method: str, path: str) -> Dict[str, str]:
    ts_ms = _now_ms()
    signature = _sign_request(private_key, ts_ms, method, path)
    return {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json",
    }

def _get(private_key, path: str, params=None):
    url = BASE_URL + path
    headers = _auth_headers(private_key, "GET", path)
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    if resp.status_code != 200: raise RuntimeError(f"GET {path} failed: {resp.text}")
    return resp.json()

def _post(private_key, path: str, body: dict):
    url = BASE_URL + path
    headers = _auth_headers(private_key, "POST", path)
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
    if resp.status_code not in (200, 201): raise RuntimeError(f"POST {path} failed: {resp.text}")
    return resp.json()

def _delete(private_key, path: str):
    url = BASE_URL + path
    headers = _auth_headers(private_key, "DELETE", path)
    resp = requests.delete(url, headers=headers, timeout=20)
    if resp.status_code not in (200, 204): raise RuntimeError(f"DELETE {path} failed: {resp.text}")
    return resp.json() if resp.text else {}


# =========================
# UTIL
# =========================
def parse_iso(ts: str): return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
def utc_now(): return dt.datetime.now(dt.timezone.utc)

def print_status(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# CSV Logging
def init_log_file() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"grok_kalshi_unlv_probe_{timestamp}.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["t_utc","ticker","yes_bid_c","yes_ask_c","no_bid_c","no_ask_c","open_side","open_price_c","open_qty","pairs_count","locked_pnl_c","realized_pnl_c","mtm_pnl_c","total_pnl_c","stopped_out_yes","stopped_out_no","secs_to_close"])
    print_status(f"Logging to: {log_path}")
    return log_path

def append_log_row(log_path, ticker, prices, state, locked_cents, mtm_cents, total_cents, secs_to_close):
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            utc_now().isoformat(), ticker,
            prices.get("best_yes_bid"), prices.get("imp_yes_ask"),
            prices.get("best_no_bid"), prices.get("imp_no_ask"),
            state.open_side or "", state.open_price or "", state.open_qty,
            len(state.pairs), locked_cents, state.realized_pnl_cents, mtm_cents, total_cents,
            int("yes" in state.stop_out_sides), int("no" in state.stop_out_sides),
            int(secs_to_close)
        ])


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
def fetch_orderbook(ticker: str) -> Dict:
    url = BASE_URL + f"/trade-api/v2/markets/{ticker}/orderbook"
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200: raise RuntimeError(resp.text)
    return resp.json()

def derive_prices(ob: Dict) -> Dict:
    yes_levels = (ob.get("orderbook") or {}).get("yes", []) or []
    no_levels = (ob.get("orderbook") or {}).get("no", []) or []

    # Sort descending by price (best first)
    yes_levels = sorted(yes_levels, key=lambda x: x[0], reverse=True)
    no_levels = sorted(no_levels, key=lambda x: x[0], reverse=True)

    best_yes_bid = int(yes_levels[0][0]) if yes_levels else None
    best_no_bid = int(no_levels[0][0]) if no_levels else None

    imp_yes_ask = 100 - best_no_bid if best_no_bid is not None else None
    imp_no_ask = 100 - best_yes_bid if best_yes_bid is not None else None

    return {
        "best_yes_bid": best_yes_bid, "best_no_bid": best_no_bid,
        "imp_yes_ask": imp_yes_ask, "imp_no_ask": imp_no_ask,
        "yes_levels": yes_levels, "no_levels": no_levels
    }

def has_sufficient_liquidity(levels: List, target_price: int, min_qty: int = MIN_LIQUIDITY_AT_LEVEL) -> bool:
    cum = 0
    for price, qty in levels:
        if price >= target_price:
            cum += qty
            if cum >= min_qty: return True
    return False


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
# UPDATED STRATEGY LOGIC
# =========================
def maybe_open_new_position(private_key, ticker: str, state: StrategyState, prices: Dict) -> None:
    if state.open_side or len(state.pairs) >= MAX_PAIRS or state.collateral_used_dollars() >= MAX_COLLATERAL_DOLLARS:

        return

    imp_yes_ask = prices["imp_yes_ask"]
    imp_no_ask = prices["imp_no_ask"]
    yes_levels = prices["yes_levels"]
    no_levels = prices["no_levels"]

    candidates = []

    # 1. Strong side (UNLV YES) - normal edge
    if imp_yes_ask and "yes" not in state.stop_out_sides:
        price = int(imp_yes_ask)
        if price <= FAIR_YES_CENTS - ENTRY_EDGE_CENTS:
            edge = FAIR_YES_CENTS - price
            candidates.append(("yes", price, edge, yes_levels))

    # 2. Weak side (Fresno NO) - stricter edge
    if imp_no_ask and "no" not in state.stop_out_sides:
        price = int(imp_no_ask)
        if price <= FAIR_NO_CENTS - ENTRY_EDGE_CENTS - WEAK_SIDE_EDGE_BUFFER:
            edge = FAIR_NO_CENTS - price
            candidates.append(("no", price, edge, no_levels))

    if not candidates: return
    candidates.sort(key=lambda x: x[2], reverse=True)  # best edge first
    side, price, edge, levels = candidates[0]

    if not has_sufficient_liquidity(levels, price):
        print_status(f"Skipped {side.upper()} @ {price}c — insufficient liquidity")
        return

    if edge >= 10:
        qty = 3
    elif edge >= 7:
        qty = 2
    else:
        qty = 1

    print_status(f"Opening BUY {side.upper()} {qty}x @ {price}c (edge {edge}¢)")
    order_id = place_limit_buy(private_key, ticker, side, price, qty)
    filled = wait_for_fill_or_timeout(private_key, order_id)

    if filled > 0:
        state.open_side, state.open_price, state.open_qty = side, price, filled
        print_status(f"Opened {side.upper()} {filled}x @ {price}c")


def maybe_lock_pair(private_key, ticker: str, state: StrategyState, prices: Dict) -> None:
    if not state.open_side or state.open_qty <= 0: return

    imp_yes_ask = prices["imp_yes_ask"]
    imp_no_ask = prices["imp_no_ask"]
    yes_levels = prices["yes_levels"]
    no_levels = prices["no_levels"]

    py = state.open_price
    qty = state.open_qty

    if state.open_side == "yes":
        max_pn = 100 - MIN_LOCKED_PROFIT_CENTS - py
        if imp_no_ask > max_pn or max_pn < 1: return
        target = int(min(imp_no_ask, max_pn))
        if not has_sufficient_liquidity(no_levels, target): return
        side_to_buy = "no"
        levels = no_levels
    else:
        max_py = 100 - MIN_LOCKED_PROFIT_CENTS - py
        if imp_yes_ask > max_py or max_py < 1: return
        target = int(min(imp_yes_ask, max_py))
        if not has_sufficient_liquidity(yes_levels, target): return
        side_to_buy = "yes"
        levels = yes_levels

    print_status(f"Locking: LONG {state.open_side.upper()} @{py}c → BUY {side_to_buy.upper()} @{target}c")
    order_id = place_limit_buy(private_key, ticker, side_to_buy, target, qty)
    filled = wait_for_fill_or_timeout(private_key, order_id)

    if filled > 0:
        if state.open_side == "yes":
            state.pairs.append((py, target, filled))
        else:
            state.pairs.append((target, py, filled))
        state.open_side = state.open_price = state.open_qty = None
        print_status(f"Pair locked! Profit = {100 - py - target}¢ per contract")


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
    log_path = init_log_file()

    while True:
        now = utc_now()
        secs_to_close = (close_time - now).total_seconds()
        if secs_to_close <= STOP_TRADING_BEFORE_CLOSE_SECS: break

        ob = fetch_orderbook(ticker)
        prices = derive_prices(ob)

        locked_cents = state.locked_profit_cents()
        mtm_cents = state.mark_to_market_cents(prices["best_yes_bid"], prices["best_no_bid"])
        total_cents = state.total_pnl_cents(prices["best_yes_bid"], prices["best_no_bid"])

        print_status(f"YES {prices['best_yes_bid']}/{prices['imp_yes_ask']} | NO {prices['best_no_bid']}/{prices['imp_no_ask']} | "
                     f"Open: {state.open_side or 'flat'} | Pairs: {len(state.pairs)} | Total PnL: {total_cents}¢")

        append_log_row(log_path, ticker, prices, state, locked_cents, mtm_cents, total_cents, secs_to_close)

        maybe_stop_out(private_key, ticker, state, prices)
        maybe_take_profit(private_key, ticker, state, prices)
        maybe_lock_pair(private_key, ticker, state, prices)

        if state.open_side is None:
            maybe_open_new_position(private_key, ticker, state, prices)

        time.sleep(POLL_LOOP_SECS)

    # Final summary...
    print("\n=== FINAL SNAPSHOT ===")
    # ... your existing final print block ...

if __name__ == "__main__":
    main()