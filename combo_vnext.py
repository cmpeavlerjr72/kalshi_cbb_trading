# combo_vnext.py
# Next-gen version of kalshi_one_game_probe_combo.py
#
# Goals:
# - Preserve working behavior from combo_test.py (same client, fills, PnL calc, logging).
# - Add dynamic locking thresholds (earlier in game → demand higher lock, late → accept smaller).
# - Add "dead capital" logic: old, underwater positions first try a more permissive lock,
#   and if still stuck for too long, optionally exit to free capital.
# - Make the strategy easy to reuse for different games and for multi-game orchestration.

import os
import time
import json
import base64
import csv
import datetime as dt
from typing import Dict, Any, Optional, List, Tuple, Set, Callable

import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import threading
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

# Market search (fallback if we don't use the bundle)
SERIES_TICKER = os.getenv("KALSHI_SERIES_TICKER") or "KXNCAAMBGAME"
TEAM_A = os.getenv("KALSHI_TEAM_A")
TEAM_B = os.getenv("KALSHI_TEAM_B")

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
STOP_LOSS_CENTS = int(os.getenv("STOP_LOSS_CENTS") or "15")
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

# === Dynamic lock tuning ===
# "Early" = enough time that we want a slightly fatter lock
EARLY_LOCK_SECS = int(os.getenv("EARLY_LOCK_SECS") or str(30 * 60))   # 30 minutes
EARLY_LOCK_BUFFER_CENTS = int(os.getenv("EARLY_LOCK_BUFFER_CENTS") or "2")

# "Late" = close is near; we're willing to accept a bit less locked EV
LATE_LOCK_SECS = int(os.getenv("LATE_LOCK_SECS") or str(10 * 60))     # 10 minutes
LATE_LOCK_DISCOUNT_CENTS = int(os.getenv("LATE_LOCK_DISCOUNT_CENTS") or "2")

# === Dead-capital / time-in-trade parameters ===
# After this age (and if underwater), we start trying more permissive locks.
DEAD_MIN_AGE_SECS = int(os.getenv("DEAD_MIN_AGE_SECS") or str(15 * 60))    # 15 minutes
# After this age we are allowed to force-exit to free capital if still bad.
DEAD_MAX_AGE_SECS = int(os.getenv("DEAD_MAX_AGE_SECS") or str(45 * 60))    # 45 minutes
# "Meaningfully negative" threshold in cents per contract.
DEAD_MTM_CENTS = int(os.getenv("DEAD_MTM_CENTS") or "5")
# How much to relax the lock threshold (in cents) for dead-cap attempts.
DEAD_RELAX_LOCK_DELTA_CENTS = int(os.getenv("DEAD_RELAX_LOCK_DELTA_CENTS") or "2")
# Floor on the relaxed lock threshold.
DEAD_MIN_LOCK_CENTS = int(os.getenv("DEAD_MIN_LOCK_CENTS") or "2")

# === Bundle-based ticker resolution for NCAAM ===
# If both are set, we will use todays_ncaam_bundle.json to pick the ticker.
BUNDLE_JSON_PATH = (os.getenv("NCAAM_BUNDLE_JSON_PATH") or "").strip()
BUNDLE_GAME_KEY = (os.getenv("NCAAM_GAME_KEY") or "").strip()

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
    return resp.text

# =========================
# UTILITIES + LOGGING
# =========================

def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def parse_iso(s: str) -> dt.datetime:
    # Kalshi timestamps are ISO8601 with Z
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))

def print_status(msg: str) -> None:
    ts = utc_now().strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    # Don't bother tagging the main CLI; only worker threads.
    if thread_name and thread_name != "MainThread":
        prefix = f"[{thread_name}] "
    else:
        prefix = ""
    print(f"[{ts}] {prefix}{msg}", flush=True)

def init_log_file(label: str = "kalshi_combo_vnext") -> str:
    os.makedirs("logs", exist_ok=True)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("logs", f"{label}_{ts}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ts_utc",
                "ticker",
                "yes_best_bid",
                "yes_imp_ask",
                "no_best_bid",
                "no_imp_ask",
                "open_side",
                "open_qty",
                "open_vwap_c",
                "pairs_count",
                "locked_pnl_c",
                "mtm_pnl_c",
                "total_pnl_c",
                "secs_to_kalshi_close",
                "secs_to_game_end",
                "secs_to_close",
                "clock_source",
            ]
        )
    return path

def append_log_row(
    path: str,
    ticker: str,
    prices: Dict[str, Any],
    state: "StrategyState",
    locked_cents: int,
    mtm_cents: int,
    total_cents: int,
    secs_to_close: float,
) -> None:
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                utc_now().isoformat(),
                ticker,
                prices["best_yes_bid"],
                prices["imp_yes_ask"],
                prices["best_no_bid"],
                prices["imp_no_ask"],
                state.open_side or "",
                state.open_qty,
                state.open_vwap_c if state.open_vwap_c is not None else "",
                len(state.pairs),
                locked_cents,
                mtm_cents,
                total_cents,
                int(secs_to_close),
            ]
        )

# =========================
# MARKET DISCOVERY + BUNDLE
# =========================

def get_markets_in_series(private_key, series_ticker: str) -> List[Dict[str, Any]]:
    """
    Return all OPEN markets in a given series, using the same pattern that worked
    in combo_test.py: /trade-api/v2/markets?series_ticker=... with cursor paging.
    """
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {
            "status": "open",
            "limit": 200,
            "series_ticker": series_ticker,
        }
        if cursor:
            params["cursor"] = cursor

        resp = _get(private_key, "/trade-api/v2/markets", params=params)
        batch = resp.get("markets", []) or []
        markets.extend(batch)
        cursor = resp.get("cursor")

        if not cursor:
            break

    return markets

def find_game_market(private_key, team_a: str, team_b: str, series_ticker: str) -> Dict[str, Any]:
    """
    Fallback single-game discovery: search the series for a market whose title
    looks like TEAM_A vs TEAM_B (order-insensitive).
    """
    markets = get_markets_in_series(private_key, series_ticker)
    team_a_l = team_a.lower()
    team_b_l = team_b.lower()

    candidates = []
    for m in markets:
        title = (m.get("title") or "").lower()
        if team_a_l in title and team_b_l in title:
            candidates.append(m)

    if not candidates:
        raise RuntimeError(f"No markets found for {team_a} vs {team_b} in {series_ticker}")

    # If multiple, pick the one with the latest close_time.
    candidates.sort(key=lambda m: parse_iso(m["close_time"]))
    chosen = candidates[-1]
    print_status(f"Selected market from series {series_ticker}: {chosen.get('ticker')} | {chosen.get('title')}")
    return chosen

def load_ncaam_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_ml_ticker_from_bundle(bundle: Dict[str, Any], game_key: str) -> Tuple[str, Dict[str, Any]]:
    """
    Use todays_ncaam_bundle.json to resolve the ML (winner) ticker for a given game.

    We support a couple of plausible shapes, based on todays_ncaam_bundle.md:

      games[game_key]["tickers"]["ml"] -> "KXNCAAMBGAME-..."
    or:
      games[game_key]["markets"]["ml"]["ticker"] / ["market_ticker"]
    """
    games = bundle.get("games") or {}
    if game_key not in games:
        raise KeyError(f"Game key {game_key} not found in bundle")

    game = games[game_key]

    ticker = None

    tickers_block = game.get("tickers") or {}
    if isinstance(tickers_block, dict):
        ticker = tickers_block.get("ml") or tickers_block.get("moneyline")

    if ticker is None:
        markets_block = game.get("markets") or {}
        ml_block = markets_block.get("ml") or markets_block.get("moneyline") or {}
        if isinstance(ml_block, dict):
            ticker = (
                ml_block.get("ticker")
                or ml_block.get("market_ticker")
                or ml_block.get("ml_ticker")
            )

    if not ticker:
        raise RuntimeError(f"Could not resolve ML ticker for {game_key} from bundle structure")

    return ticker, game

def fetch_market(private_key, ticker: str) -> Dict[str, Any]:
    resp = _get(private_key, f"/trade-api/v2/markets/{ticker}")
    return resp.get("market") or resp

# =========================
# ORDERBOOK → PRICES
# =========================

def fetch_orderbook(ticker: str) -> Dict[str, Any]:
    resp = requests.get(
        f"{BASE_URL}/trade-api/v2/markets/{ticker}/orderbook",
        timeout=10,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Orderbook fetch failed: {resp.status_code} {resp.text}")
    return resp.json()

def _parse_levels(levels: Any) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for lvl in levels or []:
        try:
            px = int(lvl.get("price", 0))
            q = int(lvl.get("count", 0))
        except (TypeError, ValueError):
            continue
        if q <= 0 or px <= 0:
            continue
        out.append((px, q))
    # sort descending by price (bids) or ascending (asks) will be handled by caller
    return out

def derive_prices(ob: Dict[str, Any]) -> Dict[str, Any]:
    """
    Match the working logic from combo_test.py:
    - ob["orderbook"]["yes"] and ob["orderbook"]["no"] are lists of [price, qty]
    - We only really need best bids; implied asks come from 100 - opposite bid.
    """
    orderbook = (ob.get("orderbook") or {})
    yes_levels = orderbook.get("yes", []) or []
    no_levels = orderbook.get("no", []) or []

    # Sort descending by price => best bid first
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
        "best_yes_ask": None,  # not used but kept for completeness
        "best_no_ask": None,
        "imp_yes_ask": imp_yes_ask,
        "imp_no_ask": imp_no_ask,
        "yes_levels": yes_levels,  # [[price, qty], ...]
        "no_levels": no_levels,
    }

def _cum_qty_at_or_above(levels: List[List[int]], price_threshold: int) -> int:
    """
    Sum quantity for all price levels >= price_threshold.
    Levels are [[price, qty], ...] sorted by price desc.
    """
    cum = 0
    for p, q in levels:
        if int(p) >= int(price_threshold):
            cum += int(q)
        else:
            break
    return cum

def has_fill_liquidity_for_implied_buy(
    prices: Dict[str, Any], side_to_buy: str, buy_price_c: int, min_qty: int
) -> bool:
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

    if tot_qty == 0:
        return None
    return tot_px / tot_qty

def wait_for_fill_or_timeout(
    private_key,
    order_id: str,
    side: str,
    max_wait_secs: int = ENTRY_TIMEOUT_SECS,
    poll_secs: int = 2,
) -> Tuple[int, Optional[float]]:
    """
    Poll fills until:
      - we see at least 1 fill, or
      - timeout expires.

    Returns (filled_quantity, vwap_cents or None).
    """
    deadline = time.time() + max_wait_secs
    last_filled = 0
    while time.time() < deadline:
        fills = fetch_fills_for_order(private_key, order_id)
        filled_qty = sum(int(f.get("count", 0) or 0) for f in fills)
        if filled_qty > 0:
            vwap = fills_vwap_cents(fills, side)
            return filled_qty, vwap
        if filled_qty > last_filled:
            last_filled = filled_qty
        time.sleep(poll_secs)

    # timeout; try to cancel
    try:
        cancel_order(private_key, order_id)
    except Exception as e:
        print_status(f"Cancel order failed (non-fatal): {e}")

    fills = fetch_fills_for_order(private_key, order_id)
    filled_qty = sum(int(f.get("count", 0) or 0) for f in fills)
    if filled_qty > 0:
        vwap = fills_vwap_cents(fills, side)
        return filled_qty, vwap
    return 0, None

# =========================
# STATE + PNL
# =========================

class StrategyState:
    def __init__(self) -> None:
        self.open_side: Optional[str] = None           # "yes" / "no" / None
        self.open_vwap_c: Optional[float] = None       # avg price in cents for open side
        self.open_qty: int = 0
        self.open_opened_at: Optional[dt.datetime] = None  # when current leg was opened

        # List of (yes_px, no_px, qty) locked pairs
        self.pairs: List[Tuple[float, float, int]] = []

        # Realized PnL from stops / exits, in cents
        self.realized_pnl_cents: int = 0

        # Sides we've stop-out'd on (don't re-enter)
        self.stop_out_sides: Set[str] = set()

    def collateral_used_dollars(self) -> float:
        """
        Rough collateral usage: worst-case for an open YES leg is 100 - price,
        for open NO leg it's price. Each locked pair is ~bound; here we only
        approximate open exposure since pairs are already hedged.
        """
        coll = 0.0
        if self.open_side is not None and self.open_vwap_c is not None and self.open_qty > 0:
            if self.open_side == "yes":
                coll += (100 - self.open_vwap_c) * self.open_qty / 100.0
            else:
                coll += self.open_vwap_c * self.open_qty / 100.0
        return coll

    def locked_profit_cents(self) -> int:
        total = 0
        for py, pn, q in self.pairs:
            per = 100 - py - pn
            total += int(round(per * q))
        return total

    def mark_to_market_cents(self, best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> int:
        if self.open_side is None or self.open_vwap_c is None or self.open_qty <= 0:
            return 0

        if self.open_side == "yes":
            if best_yes_bid is None:
                return 0
            per = best_yes_bid - self.open_vwap_c
        else:
            if best_no_bid is None:
                return 0
            per = best_no_bid - self.open_vwap_c

        return int(round(per * self.open_qty))

    def total_pnl_cents(self, best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> int:
        realized = self.realized_pnl_cents
        locked = self.locked_profit_cents()
        mtm = self.mark_to_market_cents(best_yes_bid, best_no_bid)
        return realized + locked + mtm

def clear_open_position(state: StrategyState) -> None:
    state.open_side = None
    state.open_vwap_c = None
    state.open_qty = 0
    state.open_opened_at = None

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

def lock_threshold_cents(secs_to_close: float) -> int:
    """
    Dynamic minimum lock threshold in cents.

    - Far from close (>= EARLY_LOCK_SECS): demand a bit more than MIN_LOCKED_PROFIT.
    - In the middle: use MIN_LOCKED_PROFIT.
    - Very close to close (<= LATE_LOCK_SECS): allow slightly smaller locks.
    """
    base = MIN_LOCKED_PROFIT_CENTS
    if secs_to_close >= EARLY_LOCK_SECS:
        return base + EARLY_LOCK_BUFFER_CENTS
    if secs_to_close <= LATE_LOCK_SECS:
        return max(1, base - LATE_LOCK_DISCOUNT_CENTS)
    return base

def maybe_open_new_position(
    private_key,
    ticker: str,
    state: StrategyState,
    prices: Dict[str, Any],
    fair_yes_cents: int,
    fair_no_cents: int,
    max_collateral_dollars: float,
) -> None:
    if state.open_side is not None:
        return
    if len(state.pairs) >= MAX_PAIRS:
        return
    if state.collateral_used_dollars() >= max_collateral_dollars:
        return

    imp_yes_ask = prices["imp_yes_ask"]
    imp_no_ask = prices["imp_no_ask"]

    candidates: List[Tuple[str, int, int]] = []

    # Buy YES if cheap vs fair
    if imp_yes_ask is not None and "yes" not in state.stop_out_sides:
        yes_px = int(imp_yes_ask)
        yes_edge = fair_yes_cents - yes_px
        if yes_edge >= ENTRY_EDGE_CENTS:
            candidates.append(("yes", yes_px, yes_edge))

    # Buy NO if cheap vs fair
    if imp_no_ask is not None and "no" not in state.stop_out_sides:
        no_px = int(imp_no_ask)
        no_edge = fair_no_cents - no_px
        if no_edge >= ENTRY_EDGE_CENTS:
            candidates.append(("no", no_px, no_edge))

    if not candidates:
        return

    # take best edge
    candidates.sort(key=lambda x: x[2], reverse=True)
    side, taker_px, edge = candidates[0]
    qty = edge_based_qty(edge)

    # Liquidity check (opposite book)
    if not has_fill_liquidity_for_implied_buy(
        prices, side, taker_px, min_qty=max(MIN_LIQUIDITY_CONTRACTS, qty)
    ):
        print_status(f"Skip entry: BUY {side.upper()} @{taker_px}c — insufficient opposite-book liquidity")
        return

    # Optional maker-then-taker: try 1c better for a few seconds if edge isn't huge
    maker_px = max(1, taker_px - MAKER_IMPROVE_CENTS)
    do_maker_first = ENABLE_MAKER_THEN_TAKER and edge <= (ENTRY_EDGE_CENTS + 3) and maker_px < taker_px

    if do_maker_first:
        print_status(
            f"Entry (maker try): BUY {side.upper()} {qty}x @ {maker_px}c (edge {edge}¢), "
            f"then fallback to {taker_px}c"
        )
        oid = place_limit_buy(private_key, ticker, side, maker_px, qty)
        filled, vwap = wait_for_fill_or_timeout(private_key, oid, side, max_wait_secs=MAKER_WAIT_SECS, poll_secs=2)
        if filled > 0:
            state.open_side = side
            state.open_vwap_c = vwap if vwap is not None else float(maker_px)
            state.open_qty = filled
            state.open_opened_at = utc_now()
            print_status(f"Opened {side.upper()} {filled}x @ VWAP={state.open_vwap_c:.2f}c")
            return

        # fallback to taker price
        print_status(f"Entry fallback (taker): BUY {side.upper()} {qty}x @ {taker_px}c (edge {edge}¢)")
        oid = place_limit_buy(private_key, ticker, side, taker_px, qty)
        filled, vwap = wait_for_fill_or_timeout(private_key, oid, side)
        if filled > 0:
            state.open_side = side
            state.open_vwap_c = vwap if vwap is not None else float(taker_px)
            state.open_qty = filled
            state.open_opened_at = utc_now()
            print_status(f"Opened {side.upper()} {filled}x @ VWAP={state.open_vwap_c:.2f}c")
        return

    # immediate entry
    print_status(f"Entry: BUY {side.upper()} {qty}x @ {taker_px}c (edge {edge}¢)")
    oid = place_limit_buy(private_key, ticker, side, taker_px, qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, side)
    if filled > 0:
        state.open_side = side
        state.open_vwap_c = vwap if vwap is not None else float(taker_px)
        state.open_qty = filled
        state.open_opened_at = utc_now()
        print_status(f"Opened {side.upper()} {filled}x @ VWAP={state.open_vwap_c:.2f}c")

def maybe_lock_pair(
    private_key,
    ticker: str,
    state: StrategyState,
    prices: Dict[str, Any],
    min_lock_cents: int,
) -> bool:
    """
    Try to turn the current open leg into a locked YES/NO pair with at least
    `min_lock_cents` profit per contract.

    Returns True if a pair was successfully locked (and the open leg cleared),
    False otherwise.
    """
    if state.open_side is None or state.open_vwap_c is None or state.open_qty <= 0:
        return False

    imp_yes_ask = prices["imp_yes_ask"]
    imp_no_ask = prices["imp_no_ask"]
    if imp_yes_ask is None or imp_no_ask is None:
        return False

    qty = state.open_qty
    open_side = state.open_side
    open_px = float(state.open_vwap_c)

    if open_side == "yes":
        # Need NO at pn such that open_px + pn <= 100 - min_lock_cents
        max_pn = 100 - min_lock_cents - open_px
        if max_pn < 1:
            return False
        target = int(min(imp_no_ask, max_pn))

        # Liquidity check for buying NO at implied ask uses YES book
        if not has_fill_liquidity_for_implied_buy(
            prices, "no", target, min_qty=max(MIN_LIQUIDITY_CONTRACTS, qty)
        ):
            return False

        print_status(
            f"Lock attempt: have YES @ {open_px:.2f}c, BUY NO {qty}x @ {target}c "
            f"(lock ≥{min_lock_cents}¢)"
        )
        oid = place_limit_buy(private_key, ticker, "no", target, qty)
        filled, vwap = wait_for_fill_or_timeout(private_key, oid, "no")
        if filled > 0:
            no_px = float(vwap if vwap is not None else target)
            state.pairs.append((open_px, no_px, filled))
            per = 100 - open_px - no_px
            print_status(
                f"Pair locked: YES {open_px:.2f}c + NO {no_px:.2f}c => {per:.2f}¢/contract"
            )
            clear_open_position(state)
            return True
    else:
        # open_side == "no"
        max_py = 100 - min_lock_cents - open_px
        if max_py < 1:
            return False
        target = int(min(imp_yes_ask, max_py))

        if not has_fill_liquidity_for_implied_buy(
            prices, "yes", target, min_qty=max(MIN_LIQUIDITY_CONTRACTS, qty)
        ):
            return False

        print_status(
            f"Lock attempt: have NO @ {open_px:.2f}c, BUY YES {qty}x @ {target}c "
            f"(lock ≥{min_lock_cents}¢)"
        )
        oid = place_limit_buy(private_key, ticker, "yes", target, qty)
        filled, vwap = wait_for_fill_or_timeout(private_key, oid, "yes")
        if filled > 0:
            yes_px = float(vwap if vwap is not None else target)
            state.pairs.append((yes_px, open_px, filled))
            per = 100 - yes_px - open_px
            print_status(
                f"Pair locked: YES {yes_px:.2f}c + NO {open_px:.2f}c => {per:.2f}¢/contract"
            )
            clear_open_position(state)
            return True

    return False

def exit_position(
    private_key,
    ticker: str,
    state: StrategyState,
    prices: Dict[str, Any],
    tag: str,
    add_to_stop_set: bool,
) -> bool:
    """
    Sell out of the open leg at a slightly worse-than-bid price for fill probability.

    Returns True if exit succeeded (and state was updated), False otherwise.
    """
    if state.open_side is None or state.open_vwap_c is None or state.open_qty <= 0:
        return False

    side = state.open_side
    qty = state.open_qty
    open_px = float(state.open_vwap_c)
    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]

    if side == "yes":
        if best_yes_bid is None:
            return False
        exit_px = max(1, int(best_yes_bid) - EXIT_CROSS_CENTS)
    else:
        if best_no_bid is None:
            return False
        exit_px = max(1, int(best_no_bid) - EXIT_CROSS_CENTS)

    print_status(f"{tag}: SELL {side.upper()} {qty}x @ {exit_px}c (open {open_px:.2f}c)")
    oid = place_limit_sell(private_key, ticker, side, exit_px, qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, side)
    if filled <= 0:
        print_status(f"{tag}: no fills on exit order")
        return False

    exit_vwap = vwap if vwap is not None else float(exit_px)
    per_contract = (exit_vwap - open_px) if side == "yes" else (exit_vwap - open_px)
    pnl_cents = int(round(per_contract * filled))
    state.realized_pnl_cents += pnl_cents
    print_status(f"{tag}: realized {pnl_cents}c ({pnl_cents/100:.2f}$) on {filled}x")

    if add_to_stop_set:
        state.stop_out_sides.add(side)

    clear_open_position(state)
    return True

def maybe_stop_exit(
    private_key,
    ticker: str,
    state: StrategyState,
    prices: Dict[str, Any],
) -> bool:
    """
    Pure risk control: if MTM <= -STOP_LOSS_CENTS/contract, exit and mark side as stopped.
    """
    if state.open_side is None or state.open_vwap_c is None or state.open_qty <= 0:
        return False

    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]
    mtm = state.mark_to_market_cents(best_yes_bid, best_no_bid)

    stop_thresh = -STOP_LOSS_CENTS * state.open_qty
    if mtm > stop_thresh:
        return False

    return exit_position(
        private_key, ticker, state, prices, tag="STOP-LOSS", add_to_stop_set=True
    )

def maybe_take_profit_exit(
    private_key,
    ticker: str,
    state: StrategyState,
    prices: Dict[str, Any],
) -> bool:
    """
    Take-profit-only exit: if MTM >= TAKE_PROFIT_CENTS/contract, close the leg.
    """
    if state.open_side is None or state.open_vwap_c is None or state.open_qty <= 0:
        return False

    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]
    mtm = state.mark_to_market_cents(best_yes_bid, best_no_bid)

    take_thresh = TAKE_PROFIT_CENTS * state.open_qty
    if mtm < take_thresh:
        return False

    return exit_position(
        private_key, ticker, state, prices, tag="TAKE-PROFIT", add_to_stop_set=False
    )

def maybe_handle_dead_capital(
    private_key,
    ticker: str,
    state: StrategyState,
    prices: Dict[str, Any],
    secs_to_close: float,
) -> None:
    """
    Handle positions that have been open "too long" and are meaningfully negative MTM:

    1. If age >= DEAD_MIN_AGE_SECS and mtm/contract <= -DEAD_MTM_CENTS:
         - Try a more permissive lock (dynamic lock threshold minus DEAD_RELAX_LOCK_DELTA)
    2. If still open and (age >= DEAD_MAX_AGE_SECS or mtm/contract <= -STOP_LOSS_CENTS):
         - Exit to free capital (without permanently banning this side).
    """
    if state.open_side is None or state.open_vwap_c is None or state.open_qty <= 0:
        return
    if state.open_opened_at is None:
        return

    age_secs = (utc_now() - state.open_opened_at).total_seconds()
    if age_secs < DEAD_MIN_AGE_SECS:
        return

    best_yes_bid = prices["best_yes_bid"]
    best_no_bid = prices["best_no_bid"]
    mtm = state.mark_to_market_cents(best_yes_bid, best_no_bid)
    if state.open_qty <= 0:
        return

    per_contract = mtm / state.open_qty

    # Only act if meaningfully negative
    if per_contract > -DEAD_MTM_CENTS:
        return

    # Step 1: try a more permissive lock
    relaxed_lock = max(
        lock_threshold_cents(secs_to_close) - DEAD_RELAX_LOCK_DELTA_CENTS,
        DEAD_MIN_LOCK_CENTS,
    )
    if maybe_lock_pair(private_key, ticker, state, prices, relaxed_lock):
        print_status(
            f"DEAD-CAP: successfully locked position after {age_secs:.0f}s at relaxed "
            f"threshold {relaxed_lock}¢"
        )
        return

    # Still open? Optionally exit if very old or very bad.
    if state.open_side is None or state.open_qty <= 0:
        return

    if age_secs >= DEAD_MAX_AGE_SECS or per_contract <= -STOP_LOSS_CENTS:
        exited = exit_position(
            private_key,
            ticker,
            state,
            prices,
            tag="DEAD-CAP-EXIT",
            add_to_stop_set=False,
        )
        if exited:
            print_status(
                f"DEAD-CAP: exited stale leg after {age_secs:.0f}s with "
                f"{per_contract:.2f}¢/contract MTM to free capital"
            )

# =========================
# ONE-MARKET ORCHESTRATOR (REUSABLE)
# =========================

def run_combo_for_one_market(
    private_key,
    ticker: str,
    fair_yes_cents: int,
    fair_no_cents: int,
    market: Dict[str, Any],
    log_label: Optional[str] = None,
    max_collateral_dollars: Optional[float] = None,
    get_secs_to_game_end: Optional[Callable[[], Tuple[Optional[int], str]]] = None,
) -> None:
    """
    Core loop for a single market. This is the function to import and reuse
    for arbitrary games / markets.

    - `fair_yes_cents`, `fair_no_cents` come from your model.
    - `market` is the Kalshi market object (contains `close_time`, etc.).
    - `max_collateral_dollars` overrides the global MAX_COLLATERAL_DOLLARS if provided.
    """
    close_time = parse_iso(market["close_time"])
    state = StrategyState()
    label = log_label or f"kalshi_combo_vnext_{ticker}"
    log_path = init_log_file(label=label)

    # Per-game collateral cap
    per_game_collateral = (
        max_collateral_dollars if max_collateral_dollars is not None else MAX_COLLATERAL_DOLLARS
    )

    print_status(
        f"Starting combo_vnext on {ticker} | {market.get('title')} | "
        f"fair YES={fair_yes_cents}c, NO={fair_no_cents}c | "
        f"max_collateral=${per_game_collateral:.2f}"
    )

    while True:
        now = utc_now()
        kalshi_secs = (close_time - now).total_seconds()

        game_secs = None
        game_status = ""
        if get_secs_to_game_end is not None:
            try:
                game_secs, game_status = get_secs_to_game_end()
            except Exception:
                game_secs, game_status = None, "clock_exception"

        if game_secs is not None:
            secs_to_close = min(kalshi_secs, float(game_secs))
            clock_source = f"espn_min:{game_status}"
        else:
            secs_to_close = kalshi_secs
            clock_source = "kalshi_only"

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
            f"Pairs={len(state.pairs)} | TotalPnL={total_cents}c | "
            f"secs_to_close={int(secs_to_close)} | kalshi={int(kalshi_secs)} espn={game_secs} src={clock_source}"

        )

        append_log_row(log_path, ticker, prices, state, locked_cents, mtm_cents, total_cents, secs_to_close)

        # Decision order:
        # 1) Hard stop-loss
        if maybe_stop_exit(private_key, ticker, state, prices):
            time.sleep(POLL_LOOP_SECS)
            continue

        # 2) Try to lock the leg using dynamic threshold
        if state.open_side is not None:
            min_lock = lock_threshold_cents(secs_to_close)
            maybe_lock_pair(private_key, ticker, state, prices, min_lock)

        # 3) If still open, consider take-profit exit
        if state.open_side is not None:
            maybe_take_profit_exit(private_key, ticker, state, prices)

        # 4) Dead-capital handler (age + underwater)
        if state.open_side is not None:
            maybe_handle_dead_capital(private_key, ticker, state, prices, secs_to_close)

        # 5) If flat, consider new entries
        if state.open_side is None:
            maybe_open_new_position(
                private_key,
                ticker,
                state,
                prices,
                fair_yes_cents,
                fair_no_cents,
                max_collateral_dollars=per_game_collateral,
            )

        time.sleep(POLL_LOOP_SECS)

    # final snapshot (unchanged)...
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

# =========================
# CLI MAIN (single-game entrypoint)
# =========================

def main() -> None:
    if not API_KEY_ID or not PRIVATE_KEY_PATH:
        raise RuntimeError("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set in the environment")

    private_key = _load_private_key(PRIVATE_KEY_PATH)

    # 1) Ticker override (highest priority)
    if TICKER_OVERRIDE:
        ticker = TICKER_OVERRIDE
        market = fetch_market(private_key, ticker)
        print_status(f"Using TICKER_OVERRIDE={ticker}")
    # 2) Bundle-based NCAAM resolution
    elif BUNDLE_JSON_PATH and BUNDLE_GAME_KEY:
        bundle = load_ncaam_bundle(BUNDLE_JSON_PATH)
        ticker, game_obj = resolve_ml_ticker_from_bundle(bundle, BUNDLE_GAME_KEY)
        market = fetch_market(private_key, ticker)
        print_status(
            f"Resolved ticker from bundle: game_key={BUNDLE_GAME_KEY}, ticker={ticker}, "
            f"matchup={game_obj.get('matchup') or game_obj.get('title')}"
        )
    # 3) Fallback: search the series for TEAM_A vs TEAM_B
    else:
        market = find_game_market(private_key, TEAM_A, TEAM_B, SERIES_TICKER)
        ticker = market["ticker"]

    run_combo_for_one_market(
        private_key=private_key,
        ticker=ticker,
        fair_yes_cents=FAIR_YES_CENTS,
        fair_no_cents=FAIR_NO_CENTS,
        market=market,
        log_label=f"kalshi_combo_vnext_{ticker}",
    )

if __name__ == "__main__":
    main()
