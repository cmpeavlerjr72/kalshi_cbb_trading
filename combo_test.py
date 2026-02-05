# kalshi_one_game_probe_combo.py
# Drop-in "best of Grok + GPT-5.1" for a single CBB game market on Kalshi
#
# Key upgrades vs both originals:
# - Correct liquidity checks (must check *opposite* book for implied ask fills)
# - VWAP fill prices (PnL + pair locking based on actual fills)
# - More aggressive exits (sell through bid a bit to actually fill)
# - Optional maker-then-taker entry (small edge -> try to improve price first)
# - Safer market selection heuristics + optional ticker override

import os
import time
import json
import base64
import csv
import datetime as dt
from typing import Dict, Any, Optional, List, Tuple, Set

import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# =========================
# CONFIG + .env LOADING
# =========================

load_dotenv()

ENV = (os.getenv("KALSHI_ENV") or "DEMO").upper()  # DEMO or PROD
BASE_URL = "https://demo-api.kalshi.co" if ENV == "DEMO" else "https://api.elections.kalshi.com"

API_KEY_ID = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
PRIVATE_KEY_PATH = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()

# Optional: bypass discovery if you already know the ticker
TICKER_OVERRIDE = (os.getenv("KALSHI_TICKER_OVERRIDE") or "").strip()

# Market search
SERIES_TICKER = os.getenv("KALSHI_SERIES_TICKER") or "KXNCAAMBGAME"
TEAM_A = os.getenv("KALSHI_TEAM_A") or "UNLV"
TEAM_B = os.getenv("KALSHI_TEAM_B") or "Fresno"

# === YOUR MODEL (edit for the game) ===
P_MODEL_TEAM_A = float(os.getenv("P_MODEL_TEAM_A") or "0.51")  # TEAM_A win prob
FAIR_YES_CENTS = int(round(P_MODEL_TEAM_A * 100))              # YES = TEAM_A wins
FAIR_NO_CENTS = int(round((1.0 - P_MODEL_TEAM_A) * 100))

# === STRATEGY PARAMS (tune quickly) ===
ENTRY_EDGE_CENTS = int(os.getenv("ENTRY_EDGE_CENTS") or "5")          # minimum edge vs fair to enter
MIN_LOCKED_PROFIT_CENTS = int(os.getenv("MIN_LOCKED_PROFIT_CENTS") or "8")

MAX_PAIRS = int(os.getenv("MAX_PAIRS") or "6")                        # allow more pairs for "profit seeking"
MAX_COLLATERAL_DOLLARS = float(os.getenv("MAX_COLLATERAL_DOLLARS") or "8.0")

# Stop / take profit are in cents per contract (MTM). Exits attempt to fill quickly.
STOP_LOSS_CENTS = int(os.getenv("STOP_LOSS_CENTS") or "25")
TAKE_PROFIT_CENTS = int(os.getenv("TAKE_PROFIT_CENTS") or "10")
EXIT_CROSS_CENTS = int(os.getenv("EXIT_CROSS_CENTS") or "2")          # sell 1-2c through bid for fills

# Liquidity
MIN_LIQUIDITY_CONTRACTS = int(os.getenv("MIN_LIQUIDITY_CONTRACTS") or "1")

# Timing
ENTRY_TIMEOUT_SECS = int(os.getenv("ENTRY_TIMEOUT_SECS") or "18")
POLL_LOOP_SECS = float(os.getenv("POLL_LOOP_SECS") or "4.0")
STOP_TRADING_BEFORE_CLOSE_SECS = int(os.getenv("STOP_TRADING_BEFORE_CLOSE_SECS") or "300")

# Optional maker-then-taker entry behavior (small edges)
ENABLE_MAKER_THEN_TAKER = (os.getenv("ENABLE_MAKER_THEN_TAKER") or "1") == "1"
MAKER_IMPROVE_CENTS = int(os.getenv("MAKER_IMPROVE_CENTS") or "1")    # try 1c better than implied ask
MAKER_WAIT_SECS = int(os.getenv("MAKER_WAIT_SECS") or "6")

# Edge-based sizing (simple)
MAX_ORDER_QTY = int(os.getenv("MAX_ORDER_QTY") or "3")

# =========================
# AUTH + HTTP HELPERS
# =========================

def _now_ms() -> str:
    return str(int(time.time() * 1000))

def _load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def _sign_request(private_key, ts_ms: str, method: str, path: str) -> str:
    # IMPORTANT: sign the path without query params
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
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# =========================
# CSV LOGGING
# =========================

def init_log_file() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"kalshi_one_game_combo_{timestamp}.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "t_utc","ticker",
            "yes_bid_c","yes_ask_c","no_bid_c","no_ask_c",
            "open_side","open_vwap_c","open_qty",
            "pairs_count","locked_pnl_c","realized_pnl_c","mtm_pnl_c","total_pnl_c",
            "stopped_out_yes","stopped_out_no","secs_to_close"
        ])
    print_status(f"Logging to: {log_path}")
    return log_path

def append_log_row(log_path: str, ticker: str, prices: Dict[str, Any], state: "StrategyState",
                   locked_cents: int, mtm_cents: int, total_cents: int, secs_to_close: float) -> None:
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            utc_now().isoformat(), ticker,
            prices.get("best_yes_bid",""), prices.get("imp_yes_ask",""),
            prices.get("best_no_bid",""), prices.get("imp_no_ask",""),
            state.open_side or "",
            state.open_vwap_c if state.open_vwap_c is not None else "",
            state.open_qty,
            len(state.pairs),
            locked_cents,
            state.realized_pnl_cents,
            mtm_cents,
            total_cents,
            int("yes" in state.stop_out_sides),
            int("no" in state.stop_out_sides),
            int(secs_to_close),
        ])

# =========================
# MARKET DISCOVERY
# =========================

def _market_text(m: Dict[str, Any]) -> str:
    return f"{m.get('title','')} {m.get('subtitle','')}".strip().lower()

def _looks_like_winner_market(txt: str) -> bool:
    # heuristic filters to avoid spreads/totals/multi-game
    bad = ["wins by", "over ", "under ", "points", "total", "spread", "handicap", "multi", "parlay"]
    if any(b in txt for b in bad):
        return False
    good = ["winner", "wins", "win"]
    return any(g in txt for g in good)

def find_game_market(private_key) -> Tuple[str, Dict[str, Any]]:
    team_a = TEAM_A.lower()
    team_b = TEAM_B.lower()

    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {"status": "open", "limit": 200, "series_ticker": SERIES_TICKER}
        if cursor:
            params["cursor"] = cursor
        resp = _get(private_key, "/trade-api/v2/markets", params=params)
        batch = resp.get("markets", []) or []
        markets.extend(batch)
        cursor = resp.get("cursor")
        if not cursor:
            break

    if not markets:
        raise RuntimeError("No open markets returned for series_ticker.")

    # candidates must include team A mention; bonus if includes team B and looks like winner market
    candidates = []
    for m in markets:
        txt = _market_text(m)
        if team_a not in txt:
            continue
        score = 0
        score += 5 if team_a in txt else 0
        score += 3 if team_b in txt else 0
        score += 3 if _looks_like_winner_market(txt) else 0
        # extra: title mention matters more
        title = (m.get("title") or "").lower()
        score += 2 if team_a in title else 0
        score += 1 if team_b in title else 0
        candidates.append((score, m))

    if not candidates:
        raise RuntimeError(f"No candidate markets mention {TEAM_A}.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]

    print_status(f"Selected market: {chosen.get('ticker')} | {chosen.get('title')} | closes={chosen.get('close_time')}")
    return chosen["ticker"], chosen

# =========================
# ORDERBOOK HELPERS
# =========================

def fetch_orderbook(ticker: str) -> Dict[str, Any]:
    url = BASE_URL + f"/trade-api/v2/markets/{ticker}/orderbook"
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Orderbook fetch failed: {resp.status_code} {resp.text}")
    return resp.json()

def derive_prices(ob: Dict[str, Any]) -> Dict[str, Any]:
    yes_levels = (ob.get("orderbook") or {}).get("yes", []) or []
    no_levels = (ob.get("orderbook") or {}).get("no", []) or []

    # Sort descending (best bid first)
    yes_levels = sorted(yes_levels, key=lambda x: x[0], reverse=True)
    no_levels = sorted(no_levels, key=lambda x: x[0], reverse=True)

    best_yes_bid = int(yes_levels[0][0]) if yes_levels else None
    best_no_bid = int(no_levels[0][0]) if no_levels else None

    # implied asks via complement
    imp_yes_ask = (100 - best_no_bid) if best_no_bid is not None else None
    imp_no_ask = (100 - best_yes_bid) if best_yes_bid is not None else None

    return {
        "best_yes_bid": best_yes_bid,
        "best_no_bid": best_no_bid,
        "imp_yes_ask": imp_yes_ask,
        "imp_no_ask": imp_no_ask,
        "yes_levels": yes_levels,
        "no_levels": no_levels,
    }

def _cum_qty_at_or_above(levels: List[List[int]], price_threshold: int) -> int:
    # levels are [[price, qty], ...] with price desc
    cum = 0
    for p, q in levels:
        if int(p) >= int(price_threshold):
            cum += int(q)
        else:
            break
    return cum

def has_fill_liquidity_for_implied_buy(prices: Dict[str, Any], side_to_buy: str, buy_price_c: int, min_qty: int) -> bool:
    """
    CRITICAL: For a BUY at implied ask, you must check liquidity on the OPPOSITE book.

    - If buying YES at price p, you fill against NO bids priced >= (100 - p).
    - If buying NO at price q, you fill against YES bids priced >= (100 - q).
    """
    yes_levels = prices["yes_levels"]
    no_levels = prices["no_levels"]

    if side_to_buy == "yes":
        needed_no_bid = 100 - int(buy_price_c)
        return _cum_qty_at_or_above(no_levels, needed_no_bid) >= min_qty
    else:
        needed_yes_bid = 100 - int(buy_price_c)
        return _cum_qty_at_or_above(yes_levels, needed_yes_bid) >= min_qty

# =========================
# ORDER HELPERS
# =========================

def place_limit_buy(private_key, ticker: str, side: str, price_cents: int, qty: int) -> str:
    body = {"ticker": ticker, "type": "limit", "action": "buy", "side": side, "count": int(qty)}
    if side == "yes":
        body["yes_price"] = int(price_cents)
    else:
        body["no_price"] = int(price_cents)

    resp = _post(private_key, "/trade-api/v2/portfolio/orders", body)
    oid = (resp.get("order") or {}).get("order_id")
    if not oid:
        raise RuntimeError(f"Create buy order failed: {resp}")
    return oid

def place_limit_sell(private_key, ticker: str, side: str, price_cents: int, qty: int) -> str:
    body = {"ticker": ticker, "type": "limit", "action": "sell", "side": side, "count": int(qty)}
    if side == "yes":
        body["yes_price"] = int(price_cents)
    else:
        body["no_price"] = int(price_cents)

    resp = _post(private_key, "/trade-api/v2/portfolio/orders", body)
    oid = (resp.get("order") or {}).get("order_id")
    if not oid:
        raise RuntimeError(f"Create sell order failed: {resp}")
    return oid

def cancel_order(private_key, order_id: str) -> None:
    _delete(private_key, f"/trade-api/v2/portfolio/orders/{order_id}")

def fetch_fills_for_order(private_key, order_id: str) -> List[Dict[str, Any]]:
    resp = _get(private_key, "/trade-api/v2/portfolio/fills", params={"order_id": order_id, "limit": 200})
    return resp.get("fills", []) or []

def fills_vwap_cents(fills: List[Dict[str, Any]], side: str) -> Optional[float]:
    """
    Try to compute VWAP from fill objects. Field names can vary;
    we defensively look for yes_price/no_price/price.
    """
    tot_qty = 0
    tot_px = 0.0
    for f in fills:
        try:
            q = int(f.get("count", 0))
        except (TypeError, ValueError):
            continue
        if q <= 0:
            continue

        px = None
        if side == "yes":
            px = f.get("yes_price", None)
        else:
            px = f.get("no_price", None)
        if px is None:
            px = f.get("price", None)

        try:
            px_f = float(px)
        except (TypeError, ValueError):
            continue

        tot_qty += q
        tot_px += px_f * q

    if tot_qty <= 0:
        return None
    return tot_px / tot_qty

def wait_for_fill_or_timeout(private_key, order_id: str, side: str,
                             max_wait_secs: int = ENTRY_TIMEOUT_SECS, poll_secs: float = 2.0) -> Tuple[int, Optional[float]]:
    """
    Poll fills for this order until it fills or times out.
    Returns (filled_qty, vwap_price_cents).
    """
    start = time.time()
    last_qty = -1

    while True:
        fills = []
        try:
            fills = fetch_fills_for_order(private_key, order_id)
        except Exception as e:
            print_status(f"fill poll error (ignored): {e}")

        filled = 0
        for f in fills:
            try:
                filled += int(f.get("count", 0))
            except (TypeError, ValueError):
                pass

        if filled != last_qty:
            print_status(f"Order {order_id}: filled={filled}")
            last_qty = filled

        if filled > 0:
            vwap = fills_vwap_cents(fills, side)
            return filled, vwap

        if time.time() - start >= max_wait_secs:
            print_status(f"Order {order_id}: timeout, cancelling.")
            try:
                cancel_order(private_key, order_id)
            except Exception as e:
                print_status(f"Cancel error (ignored): {e}")
            return 0, None

        time.sleep(poll_secs)

# =========================
# STATE
# =========================

class StrategyState:
    def __init__(self):
        self.open_side: Optional[str] = None        # "yes"/"no"
        self.open_vwap_c: Optional[float] = None    # actual fill VWAP
        self.open_qty: int = 0

        # locked pairs stored as (yes_vwap_c, no_vwap_c, qty)
        self.pairs: List[Tuple[float, float, int]] = []

        self.realized_pnl_cents: int = 0
        self.stop_out_sides: Set[str] = set()

    def collateral_used_dollars(self) -> float:
        used = 0.0
        if self.open_side and self.open_qty > 0:
            used += 1.0
        used += len(self.pairs) * 1.0
        return used

    def locked_profit_cents(self) -> int:
        tot = 0
        for (py, pn, qty) in self.pairs:
            tot += int(round(qty * (100 - py - pn)))
        return tot

    def mark_to_market_cents(self, best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> int:
        if not self.open_side or self.open_vwap_c is None or self.open_qty <= 0:
            return 0
        if self.open_side == "yes":
            if best_yes_bid is None:
                return 0
            return int(round((best_yes_bid - self.open_vwap_c) * self.open_qty))
        else:
            if best_no_bid is None:
                return 0
            return int(round((best_no_bid - self.open_vwap_c) * self.open_qty))

    def total_pnl_cents(self, best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> int:
        return self.locked_profit_cents() + self.realized_pnl_cents + self.mark_to_market_cents(best_yes_bid, best_no_bid)

# =========================
# STRATEGY LOGIC
# =========================

def edge_based_qty(edge_cents: int) -> int:
    # simple: bigger edge -> bigger size (still small)
    if edge_cents >= 12:
        return min(MAX_ORDER_QTY, 3)
    if edge_cents >= 8:
        return min(MAX_ORDER_QTY, 2)
    return 1

def maybe_open_new_position(private_key, ticker: str, state: StrategyState, prices: Dict[str, Any]) -> None:
    if state.open_side is not None:
        return
    if len(state.pairs) >= MAX_PAIRS:
        return
    if state.collateral_used_dollars() >= MAX_COLLATERAL_DOLLARS:
        return

    imp_yes_ask = prices["imp_yes_ask"]
    imp_no_ask = prices["imp_no_ask"]

    candidates = []

    # Buy YES if cheap vs fair
    if imp_yes_ask is not None and "yes" not in state.stop_out_sides:
        yes_px = int(imp_yes_ask)
        yes_edge = FAIR_YES_CENTS - yes_px
        if yes_edge >= ENTRY_EDGE_CENTS:
            candidates.append(("yes", yes_px, yes_edge))

    # Buy NO if cheap vs fair
    if imp_no_ask is not None and "no" not in state.stop_out_sides:
        no_px = int(imp_no_ask)
        no_edge = FAIR_NO_CENTS - no_px
        if no_edge >= ENTRY_EDGE_CENTS:
            candidates.append(("no", no_px, no_edge))

    if not candidates:
        return

    # take best edge
    candidates.sort(key=lambda x: x[2], reverse=True)
    side, taker_px, edge = candidates[0]
    qty = edge_based_qty(edge)

    # Liquidity check (opposite book)
    if not has_fill_liquidity_for_implied_buy(prices, side, taker_px, min_qty=max(MIN_LIQUIDITY_CONTRACTS, qty)):
        print_status(f"Skip entry: BUY {side.upper()} @{taker_px}c — insufficient opposite-book liquidity")
        return

    # Optional maker-then-taker: try 1c better for a few seconds if edge isn't huge
    maker_px = max(1, taker_px - MAKER_IMPROVE_CENTS)
    do_maker_first = ENABLE_MAKER_THEN_TAKER and edge <= (ENTRY_EDGE_CENTS + 3) and maker_px < taker_px

    if do_maker_first:
        print_status(f"Entry (maker try): BUY {side.upper()} {qty}x @ {maker_px}c (edge {edge}¢), then fallback to {taker_px}c")
        oid = place_limit_buy(private_key, ticker, side, maker_px, qty)
        filled, vwap = wait_for_fill_or_timeout(private_key, oid, side, max_wait_secs=MAKER_WAIT_SECS, poll_secs=2)
        if filled > 0:
            state.open_side, state.open_vwap_c, state.open_qty = side, (vwap if vwap is not None else float(maker_px)), filled
            print_status(f"Opened {side.upper()} {filled}x @ VWAP={state.open_vwap_c:.2f}c")
            return

        # fallback to taker price
        print_status(f"Entry fallback (taker): BUY {side.upper()} {qty}x @ {taker_px}c (edge {edge}¢)")
        oid = place_limit_buy(private_key, ticker, side, taker_px, qty)
        filled, vwap = wait_for_fill_or_timeout(private_key, oid, side)
        if filled > 0:
            state.open_side, state.open_vwap_c, state.open_qty = side, (vwap if vwap is not None else float(taker_px)), filled
            print_status(f"Opened {side.upper()} {filled}x @ VWAP={state.open_vwap_c:.2f}c")
        return

    # immediate entry
    print_status(f"Entry: BUY {side.upper()} {qty}x @ {taker_px}c (edge {edge}¢)")
    oid = place_limit_buy(private_key, ticker, side, taker_px, qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, side)
    if filled > 0:
        state.open_side, state.open_vwap_c, state.open_qty = side, (vwap if vwap is not None else float(taker_px)), filled
        print_status(f"Opened {side.upper()} {filled}x @ VWAP={state.open_vwap_c:.2f}c")

def maybe_lock_pair(private_key, ticker: str, state: StrategyState, prices: Dict[str, Any]) -> None:
    if state.open_side is None or state.open_vwap_c is None or state.open_qty <= 0:
        return

    imp_yes_ask = prices["imp_yes_ask"]
    imp_no_ask = prices["imp_no_ask"]
    if imp_yes_ask is None or imp_no_ask is None:
        return

    qty = state.open_qty
    open_side = state.open_side
    open_px = float(state.open_vwap_c)

    if open_side == "yes":
        # Need NO at pn such that open_px + pn <= 100 - MIN_LOCKED_PROFIT
        max_pn = 100 - MIN_LOCKED_PROFIT_CENTS - open_px
        if max_pn < 1:
            return
        target = int(min(imp_no_ask, max_pn))

        # Liquidity check for buying NO at implied ask uses YES book
        if not has_fill_liquidity_for_implied_buy(prices, "no", target, min_qty=max(MIN_LIQUIDITY_CONTRACTS, qty)):
            return

        print_status(f"Lock attempt: have YES @ {open_px:.2f}c, BUY NO {qty}x @ {target}c (lock ≥{MIN_LOCKED_PROFIT_CENTS}¢)")
        oid = place_limit_buy(private_key, ticker, "no", target, qty)
        filled, vwap = wait_for_fill_or_timeout(private_key, oid, "no")
        if filled > 0:
            no_px = float(vwap if vwap is not None else target)
            state.pairs.append((open_px, no_px, filled))
            per = 100 - open_px - no_px
            print_status(f"Pair locked: YES {open_px:.2f}c + NO {no_px:.2f}c => {per:.2f}¢/contract")
            state.open_side, state.open_vwap_c, state.open_qty = None, None, 0
    else:
        max_py = 100 - MIN_LOCKED_PROFIT_CENTS - open_px
        if max_py < 1:
            return
        target = int(min(imp_yes_ask, max_py))

        if not has_fill_liquidity_for_implied_buy(prices, "yes", target, min_qty=max(MIN_LIQUIDITY_CONTRACTS, qty)):
            return

        print_status(f"Lock attempt: have NO @ {open_px:.2f}c, BUY YES {qty}x @ {target}c (lock ≥{MIN_LOCKED_PROFIT_CENTS}¢)")
        oid = place_limit_buy(private_key, ticker, "yes", target, qty)
        filled, vwap = wait_for_fill_or_timeout(private_key, oid, "yes")
        if filled > 0:
            yes_px = float(vwap if vwap is not None else target)
            state.pairs.append((yes_px, open_px, filled))
            per = 100 - yes_px - open_px
            print_status(f"Pair locked: YES {yes_px:.2f}c + NO {open_px:.2f}c => {per:.2f}¢/contract")
            state.open_side, state.open_vwap_c, state.open_qty = None, None, 0

def maybe_exit_on_stop_or_profit(private_key, ticker: str, state: StrategyState, prices: Dict[str, Any]) -> None:
    if state.open_side is None or state.open_vwap_c is None or state.open_qty <= 0:
        return

    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]
    mtm = state.mark_to_market_cents(best_yes_bid, best_no_bid)

    # thresholds are per-contract; mtm is already qty-scaled
    stop_thresh = -STOP_LOSS_CENTS * state.open_qty
    take_thresh = TAKE_PROFIT_CENTS * state.open_qty

    if mtm > stop_thresh and mtm < take_thresh:
        return

    side = state.open_side
    qty = state.open_qty
    open_px = float(state.open_vwap_c)

    if side == "yes":
        if best_yes_bid is None:
            return
        # sell a bit below bid to fill
        exit_px = max(1, int(best_yes_bid) - EXIT_CROSS_CENTS)
    else:
        if best_no_bid is None:
            return
        exit_px = max(1, int(best_no_bid) - EXIT_CROSS_CENTS)

    tag = "TAKE-PROFIT" if mtm >= take_thresh else "STOP-LOSS"
    print_status(f"{tag}: exiting {side.upper()} {qty}x (open VWAP={open_px:.2f}c) @ {exit_px}c | mtm={mtm}c")

    oid = place_limit_sell(private_key, ticker, side, exit_px, qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, side)

    if filled > 0:
        exit_vwap = float(vwap if vwap is not None else exit_px)
        realized = int(round((exit_vwap - open_px) * filled))
        state.realized_pnl_cents += realized
        print_status(f"{tag} filled: {side.upper()} {filled}x @ VWAP={exit_vwap:.2f}c | realized={realized}c")

        # clear
        state.open_side, state.open_vwap_c, state.open_qty = None, None, 0

        # on stop-loss, ban this side
        if tag == "STOP-LOSS":
            state.stop_out_sides.add(side)
    else:
        print_status(f"{tag} sell did not fill; keeping position.")

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
    print_status(f"Teams: YES={TEAM_A} (fair {FAIR_YES_CENTS}c) vs NO={TEAM_B} (fair {FAIR_NO_CENTS}c)")
    print_status(f"Params: ENTRY_EDGE={ENTRY_EDGE_CENTS} MIN_LOCK={MIN_LOCKED_PROFIT_CENTS} STOP={STOP_LOSS_CENTS} TAKE={TAKE_PROFIT_CENTS}")

    if TICKER_OVERRIDE:
        ticker = TICKER_OVERRIDE
        market = _get(private_key, f"/trade-api/v2/markets/{ticker}")
        print_status(f"Using override ticker: {ticker} | {market.get('title')}")
    else:
        print_status("Finding market via discovery...")
        ticker, market = find_game_market(private_key)

    close_time = parse_iso(market["close_time"])
    print_status(f"Close time (UTC): {market['close_time']}")

    state = StrategyState()
    log_path = init_log_file()

    while True:
        now = utc_now()
        secs_to_close = (close_time - now).total_seconds()
        if secs_to_close <= STOP_TRADING_BEFORE_CLOSE_SECS:
            print_status("Within stop-trading window before close. Exiting loop.")
            break

        ob = fetch_orderbook(ticker)
        prices = derive_prices(ob)

        locked_cents = state.locked_profit_cents()
        mtm_cents = state.mark_to_market_cents(prices["best_yes_bid"], prices["best_no_bid"])
        total_cents = state.total_pnl_cents(prices["best_yes_bid"], prices["best_no_bid"])

        print_status(
            f"YES {prices['best_yes_bid']}/{prices['imp_yes_ask']} | "
            f"NO {prices['best_no_bid']}/{prices['imp_no_ask']} | "
            f"Open={state.open_side or 'flat'} "
            f"(qty={state.open_qty}, vwap={state.open_vwap_c if state.open_vwap_c is not None else ''}) | "
            f"Pairs={len(state.pairs)} | TotalPnL={total_cents}c"
        )

        append_log_row(log_path, ticker, prices, state, locked_cents, mtm_cents, total_cents, secs_to_close)

        # priority: exit -> lock -> open
        maybe_exit_on_stop_or_profit(private_key, ticker, state, prices)
        maybe_lock_pair(private_key, ticker, state, prices)
        if state.open_side is None:
            maybe_open_new_position(private_key, ticker, state, prices)

        time.sleep(POLL_LOOP_SECS)

    # final snapshot
    ob = fetch_orderbook(ticker)
    prices = derive_prices(ob)
    locked_cents = state.locked_profit_cents()
    mtm_cents = state.mark_to_market_cents(prices["best_yes_bid"], prices["best_no_bid"])
    total_cents = state.total_pnl_cents(prices["best_yes_bid"], prices["best_no_bid"])

    print("\n=== FINAL SNAPSHOT (pre-settlement) ===")
    print(f"Market: {ticker} | {market.get('title')}")
    print(f"Pairs locked: {len(state.pairs)} | Locked profit: {locked_cents}c (${locked_cents/100:.2f})")
    if state.pairs:
        for i, (py, pn, q) in enumerate(state.pairs, start=1):
            print(f"  Pair {i}: YES {py:.2f}c + NO {pn:.2f}c x{q} => {100 - py - pn:.2f}c/contract")
    print(f"Open: {state.open_side} qty={state.open_qty} vwap={state.open_vwap_c}")
    print(f"Realized: {state.realized_pnl_cents}c (${state.realized_pnl_cents/100:.2f})")
    print(f"MTM: {mtm_cents}c (${mtm_cents/100:.2f})")
    print(f"Total: {total_cents}c (${total_cents/100:.2f})")

if __name__ == "__main__":
    main()
