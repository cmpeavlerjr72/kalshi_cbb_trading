# superbowl_spread_mr.py
#
# Super Bowl LIX - Spread mean reversion harness
# Seattle Seahawks vs New England Patriots
#
# Goals vs spread_beluic_test:
#   - Same "structural" idea (multi-ticker spread watcher)
#   - BUT:
#       * No explicit STOP LOSS
#       * No small TAKE_PROFIT cent target
#       * Exit logic is MR-style (revert toward rolling mean) + optional late flatten
#       * Higher default contract size (ORDER_QTY > 1)
#
# Uses NFLGameClock from superbowl_mr_strategy.py so we know roughly how
# close we are to the end of the game.

import os
import csv
import json
import time
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from dotenv import load_dotenv
load_dotenv()

from combo_vnext import (
    _load_private_key,
    utc_now,
    print_status,
    fetch_orderbook,
    derive_prices,
    place_limit_buy,
    place_limit_sell,
    wait_for_fill_or_timeout,
    has_fill_liquidity_for_implied_buy,
)

from production_strategies import calc_taker_fee
from superbowl_mr_strategy import NFLGameClock

# -------------------------
# SUPER BOWL SPREAD TICKERS
# -------------------------
#
# IMPORTANT: Replace these with the actual Kalshi tickers you want to trade.
# The idea is: cluster around SEA -4.5 (e.g. SEA -3.5, -4.5, -5.5),
# both Seahawks and Patriots sides, so mid has room to move.
#
# Example *placeholder* names — DO NOT USE AS-IS:
#   "KXSBSPREAD-26FEB08SEANE-SEA35",   # SEA -3.5
#   "KXSBSPREAD-26FEB08SEANE-SEA45",   # SEA -4.5 (current line)
#   "KXSBSPREAD-26FEB08SEANE-SEA55",   # SEA -5.5
#   "KXSBSPREAD-26FEB08SEANE-NE35",    # NE +3.5
#   "KXSBSPREAD-26FEB08SEANE-NE45",    # NE +4.5
#   "KXSBSPREAD-26FEB08SEANE-NE55",    # NE +5.5
#
# Paste the *real* tickers from your superbowl_markets.json output.

SPREAD_TICKERS: List[str] = [
    # TODO: replace all of these with the real tickers
    "REPLACE_ME_SPREAD_1",
    "REPLACE_ME_SPREAD_2",
    "REPLACE_ME_SPREAD_3",
    "REPLACE_ME_SPREAD_4",
    "REPLACE_ME_SPREAD_5",
    "REPLACE_ME_SPREAD_6",
]

# -------------------------
# CONFIG (env-overridable)
# -------------------------

# Capital / sizing (per *position*; we only hold one at a time)
MAX_COLLATERAL_DOLLARS = float(os.getenv("SB_SPREAD_MAX_COLLATERAL_DOLLARS", "25.0"))

# Higher default size vs BEL/UIC (which was ORDER_QTY=1)
ORDER_QTY = int(os.getenv("SB_SPREAD_ORDER_QTY", "3"))

# Polling
POLL_SECS = float(os.getenv("SB_SPREAD_POLL_SECS", "3.0"))

# Warmup / stats
WARMUP_SAMPLES = int(os.getenv("SB_SPREAD_WARMUP_SAMPLES", "25"))
WARMUP_MIN_RANGE = float(os.getenv("SB_SPREAD_WARMUP_MIN_RANGE", "3.0"))
LOOKBACK = int(os.getenv("SB_SPREAD_LOOKBACK", "30"))

MIN_RECENT_STD_OK = float(os.getenv("SB_SPREAD_MIN_RECENT_STD_OK", "1.5"))
MAX_AVG_SPREAD_OK = float(os.getenv("SB_SPREAD_MAX_AVG_SPREAD_OK", "10.0"))

# HARD book-quality gates (CURRENT snapshot)
MAX_ENTRY_SPREAD_C = float(os.getenv("SB_SPREAD_MAX_ENTRY_SPREAD_C", "8.0"))
MIN_BID_C = float(os.getenv("SB_SPREAD_MIN_BID_C", "5.0"))
MID_MIN = float(os.getenv("SB_SPREAD_MID_MIN", "30.0"))
MID_MAX = float(os.getenv("SB_SPREAD_MID_MAX", "70.0"))

# Maker entry behavior (we stay maker-only for now to keep fees sane)
MAKER_IMPROVE_CENTS = int(os.getenv("SB_SPREAD_MAKER_IMPROVE_CENTS", "1"))
MAKER_WAIT_SECS = int(os.getenv("SB_SPREAD_MAKER_WAIT_SECS", "8"))

# Exit behavior (no hard SL/TP; MR exit + optional flatten)
MR_TP_SIGMA = float(os.getenv("SB_SPREAD_MR_TP_SIGMA", "1.0"))  # how close to mean before we call it "reverted"
FINAL_FLATTEN_SECS = int(os.getenv("SB_SPREAD_FINAL_FLATTEN_SECS", "300"))  # 5 min
PREFERRED_SIDE = os.getenv("SB_SPREAD_PREFERRED_SIDE", "").lower() or None  # "yes" / "no" or ""

# Net-edge gating (tradeability check)
EDGE_BUFFER_C = float(os.getenv("SB_SPREAD_EDGE_BUFFER_C", "3.0"))
MIN_EDGE_C = float(os.getenv("SB_SPREAD_MIN_EDGE_C", "8.0"))

# Cooldown after MR *loss* (we keep this, but exits are MR-based, not cent TP/SL)
COOLDOWN_SECS = int(os.getenv("SB_SPREAD_COOLDOWN_SECS", "180"))

# Auto-stop (kill-switch for this run, *not* per-position stop loss)
MAX_RUNTIME_MINS = int(os.getenv("SB_SPREAD_MAX_RUNTIME_MINS", "240"))
MAX_NET_LOSS_CENTS = float(os.getenv("SB_SPREAD_MAX_NET_LOSS_CENTS", "150.0"))

# Console heartbeat (optional)
HEARTBEAT_SECS = int(os.getenv("SB_SPREAD_HEARTBEAT_SECS", "60"))

# -------------------------
# DATA STRUCTURES
# -------------------------

@dataclass
class OpenPosition:
    ticker: str
    side: str          # "yes" or "no"
    entry_price: float
    qty: int
    entry_time: str
    entry_fee_c: float
    lock_attempts: int = 0
    entry_reason: str = ""

@dataclass
class ClosedPosition:
    ticker: str
    side: str
    qty: int
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    exit_type: str
    entry_fee_c: float
    exit_fee_c: float
    gross_pnl_c: float
    net_pnl_c: float
    hold_secs: int
    detail: str = ""

class TickerStats:
    """
    Rolling stats per ticker: mids, spreads and last prices.
    MR logic uses recent_mean/recent_std on mids.
    """
    def __init__(self, maxlen: int = 240):
        self.mids = deque(maxlen=maxlen)
        self.spreads = deque(maxlen=maxlen)
        self.last_prices: Optional[Dict[str, Any]] = None

    def update(self, mid: Optional[float], spread: Optional[float], prices: Dict[str, Any]):
        if mid is not None:
            self.mids.append(mid)
        if spread is not None:
            self.spreads.append(spread)
        self.last_prices = prices

    def warmup_ok(self) -> Tuple[bool, str]:
        if len(self.mids) < WARMUP_SAMPLES:
            return False, f"warmup:{len(self.mids)}/{WARMUP_SAMPLES}"
        rng = max(self.mids) - min(self.mids)
        if rng < WARMUP_MIN_RANGE:
            return False, f"warmup_range:{rng:.1f}c<{WARMUP_MIN_RANGE}c"
        return True, "ok"

    def recent_std(self) -> float:
        if len(self.mids) < 5:
            return 0.0
        recent = list(self.mids)[-LOOKBACK:] if len(self.mids) >= LOOKBACK else list(self.mids)
        m = sum(recent) / len(recent)
        var = sum((x - m) ** 2 for x in recent) / len(recent)
        return math.sqrt(var) if var > 0 else 0.0

    def recent_mean(self) -> float:
        if len(self.mids) < 5:
            return 50.0
        recent = list(self.mids)[-LOOKBACK:] if len(self.mids) >= LOOKBACK else list(self.mids)
        return sum(recent) / len(recent)

    def avg_spread(self) -> float:
        if not self.spreads:
            return 999.0
        recent = list(self.spreads)[-LOOKBACK:] if len(self.spreads) >= LOOKBACK else list(self.spreads)
        return sum(recent) / len(recent)

# -------------------------
# LOGGING
# -------------------------

def _mk_log_base() -> str:
    os.makedirs("logs", exist_ok=True)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    return f"logs/spreadtest_SB_{ts}"

def _init_logs(base: str) -> Dict[str, str]:
    paths = {
        "snap": f"{base}_snapshots.csv",
        "trades": f"{base}_trades.csv",
        "pos": f"{base}_positions.csv",
        "events": f"{base}_events.csv",
        "summary": f"{base}_summary.json",
    }
    with open(paths["snap"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ts_utc","ticker",
            "yes_bid","yes_ask","no_bid","no_ask",
            "mid","spread",
            "t_std","t_mean","t_avg_spread",
            "active","open_ticker","open_side","open_entry_px","open_qty",
            "cooldown_until_utc","secs_to_close","clock_status",
        ])
    with open(paths["trades"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ts_utc","ticker","action","side","intended_px","fill_px","qty","fee_c","reason","order_id"
        ])
    with open(paths["pos"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ticker","side","qty","entry_px","exit_px","entry_time","exit_time","exit_type",
            "entry_fee_c","exit_fee_c","gross_pnl_c","net_pnl_c","hold_secs","detail"
        ])
    with open(paths["events"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["ts_utc","event","detail"])
    return paths

def log_event(paths: Dict[str,str], event: str, detail: str=""):
    with open(paths["events"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([utc_now().isoformat(), event, detail])

def log_snapshot(paths: Dict[str,str], ticker: str, prices: Dict[str,Any], stats: TickerStats,
                 active: bool, pos: Optional[OpenPosition], cooldown_until: Optional[dt.datetime],
                 secs_to_close: Optional[int], clock_status: str):
    yb = prices.get("best_yes_bid")
    ya = prices.get("imp_yes_ask")
    nb = prices.get("best_no_bid")
    na = prices.get("imp_no_ask")

    mid = ""
    spr = ""
    if yb is not None and ya is not None:
        mid = f"{(yb+ya)/2:.1f}"
        spr = f"{(ya-yb):.1f}"

    with open(paths["snap"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            utc_now().isoformat(),
            ticker,
            yb if yb is not None else "",
            ya if ya is not None else "",
            nb if nb is not None else "",
            na if na is not None else "",
            mid,
            spr,
            f"{stats.recent_std():.2f}",
            f"{stats.recent_mean():.2f}",
            f"{stats.avg_spread():.2f}",
            int(active),
            pos.ticker if pos else "",
            pos.side if pos else "",
            f"{pos.entry_price:.2f}" if pos else "",
            pos.qty if pos else "",
            cooldown_until.isoformat() if cooldown_until else "",
            secs_to_close if secs_to_close is not None else "",
            clock_status,
        ])

def log_trade(paths: Dict[str,str], ticker: str, action: str, side: str, intended_px: int,
              fill_px: Optional[float], qty: int, fee_c: float, reason: str, order_id: str):
    with open(paths["trades"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            utc_now().isoformat(), ticker, action, side, intended_px,
            f"{fill_px:.2f}" if fill_px is not None else "",
            qty, f"{fee_c:.2f}", reason, order_id
        ])

def log_closed(paths: Dict[str,str], c: ClosedPosition):
    with open(paths["pos"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            c.ticker, c.side, c.qty, f"{c.entry_price:.2f}", f"{c.exit_price:.2f}",
            c.entry_time, c.exit_time, c.exit_type,
            f"{c.entry_fee_c:.2f}", f"{c.exit_fee_c:.2f}",
            f"{c.gross_pnl_c:.2f}", f"{c.net_pnl_c:.2f}",
            c.hold_secs, c.detail
        ])

# -------------------------
# MARKET HELPERS
# -------------------------

def compute_mid_spread(prices: Dict[str,Any]) -> Tuple[Optional[float], Optional[float]]:
    yb = prices.get("best_yes_bid")
    ya = prices.get("imp_yes_ask")
    if yb is None or ya is None:
        return None, None
    return float((yb + ya)/2.0), float(ya - yb)

def current_book_ok(prices: Dict[str,Any]) -> Tuple[bool, str, Optional[float], Optional[float]]:
    """
    Hard guardrails using CURRENT snapshot to avoid broken books.
    """
    yb = prices.get("best_yes_bid")
    ya = prices.get("imp_yes_ask")
    nb = prices.get("best_no_bid")
    na = prices.get("imp_no_ask")

    if yb is None or ya is None or nb is None or na is None:
        return False, "missing_quotes", None, None

    mid = (yb + ya) / 2.0
    spr = (ya - yb)

    if spr > MAX_ENTRY_SPREAD_C:
        return False, f"cur_spread>{MAX_ENTRY_SPREAD_C:.0f} ({spr:.1f})", float(mid), float(spr)

    if yb < MIN_BID_C or nb < MIN_BID_C:
        return False, f"thin_bids yb={yb} nb={nb}", float(mid), float(spr)

    if mid < MID_MIN or mid > MID_MAX:
        return False, f"mid_outside[{MID_MIN:.0f},{MID_MAX:.0f}] ({mid:.1f})", float(mid), float(spr)

    return True, "ok", float(mid), float(spr)

def best_ticker_to_focus(stats_by: Dict[str,TickerStats], cooldowns: Dict[str, dt.datetime]) -> Tuple[Optional[str], str]:
    """
    Pick the best candidate based on:
      - warmup ok
      - min std
      - reasonable avg spread
      - NOT in cooldown
      - CURRENT book quality ok
    """
    best = None
    best_score = -1e18
    why_best = "no_candidates"

    now = utc_now()

    for t, s in stats_by.items():
        cd = cooldowns.get(t)
        if cd and cd > now:
            continue

        ok, _ = s.warmup_ok()
        if not ok:
            continue

        std = s.recent_std()
        avg_sp = s.avg_spread()

        if avg_sp > MAX_AVG_SPREAD_OK:
            continue
        if std < MIN_RECENT_STD_OK:
            continue

        prices = s.last_prices or {}
        cur_ok, _, _, cur_spr = current_book_ok(prices)
        if not cur_ok or cur_spr is None:
            continue

        # Score: favor movement and tighter spreads
        score = (std * 10.0) - (avg_sp * 2.0) - (cur_spr * 1.0)

        # tiny preference for mid near 50 (more two-sided)
        m = s.recent_mean()
        score -= abs(m - 50.0) * 0.25

        if score > best_score:
            best_score = score
            best = t
            why_best = f"score={best_score:.1f} std={std:.1f} avg_sp={avg_sp:.1f} cur_sp={cur_spr:.1f}"

    return best, why_best

# -------------------------
# EXECUTION HELPERS
# -------------------------

def maker_only_buy(private_key, ticker: str, side: str, implied_ask_px: int, qty: int,
                   reason: str, paths: Dict[str,str]) -> Tuple[int, Optional[float]]:
    """
    Maker-only entry:
      - place 1c better than implied ask (if possible)
      - wait a bit
      - if no fill, do NOTHING (no taker fallback)
    """
    maker_px = max(1, int(implied_ask_px) - MAKER_IMPROVE_CENTS)
    oid = place_limit_buy(private_key, ticker, side, maker_px, qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, side, max_wait_secs=MAKER_WAIT_SECS)
    if filled > 0:
        px = float(vwap or maker_px)
        fee = calc_taker_fee(int(px), filled)
        log_trade(paths, ticker, "entry_fill_maker", side, maker_px, px, filled, fee, reason, oid)
        return filled, vwap

    log_trade(paths, ticker, "entry_nofill_maker", side, maker_px, None, 0, 0.0, reason, oid)
    return 0, None

def exit_sell(private_key, pos: OpenPosition, prices: Dict[str,Any], exit_type: str,
              detail: str, paths: Dict[str,str]) -> Optional[ClosedPosition]:
    bid = prices.get("best_yes_bid") if pos.side == "yes" else prices.get("best_no_bid")
    if bid is None:
        return None

    px = max(1, int(bid))  # MR exit: we don't hyper-cross; no hard SL/TP
    oid = place_limit_sell(private_key, pos.ticker, pos.side, px, pos.qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, pos.side, max_wait_secs=MAKER_WAIT_SECS)
    if filled <= 0:
        log_trade(paths, pos.ticker, f"exit_{exit_type}_nofill", pos.side, px, None, 0, 0.0, detail, oid)
        return None

    exit_px = float(vwap or px)
    exit_fee = calc_taker_fee(int(exit_px), filled)

    gross = (exit_px - pos.entry_price) * filled
    net = gross - pos.entry_fee_c - exit_fee

    hold_secs = int((utc_now() - dt.datetime.fromisoformat(pos.entry_time.replace("Z","+00:00"))).total_seconds())

    closed = ClosedPosition(
        ticker=pos.ticker,
        side=pos.side,
        qty=filled,
        entry_price=pos.entry_price,
        exit_price=exit_px,
        entry_time=pos.entry_time,
        exit_time=utc_now().isoformat(),
        exit_type=exit_type,
        entry_fee_c=pos.entry_fee_c,
        exit_fee_c=exit_fee,
        gross_pnl_c=gross,
        net_pnl_c=net,
        hold_secs=hold_secs,
        detail=detail,
    )
    log_trade(paths, pos.ticker, f"exit_{exit_type}", pos.side, px, exit_px, filled, exit_fee, detail, oid)
    return closed

def try_lock(private_key, pos: OpenPosition, prices: Dict[str,Any], paths: Dict[str,str]) -> Tuple[bool, Optional[ClosedPosition]]:
    """
    Optional lock (riskless-ish close) if both sides are playable.
    This is the one place where we are allowed to "get out early" — but only
    if it's a true lock-in profit based on prices.
    """
    if pos.lock_attempts >= 2:
        return False, None

    if pos.side == "yes":
        opp_side = "no"
        opp_ask = prices.get("imp_no_ask")
    else:
        opp_side = "yes"
        opp_ask = prices.get("imp_yes_ask")

    if opp_ask is None:
        return False, None

    locked_profit = 100 - pos.entry_price - float(opp_ask)
    if locked_profit < 4.0:
        return False, None

    if not has_fill_liquidity_for_implied_buy(prices, opp_side, int(opp_ask), min_qty=pos.qty):
        return False, None

    pos.lock_attempts += 1

    oid = place_limit_buy(private_key, pos.ticker, opp_side, int(opp_ask), pos.qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, opp_side, max_wait_secs=MAKER_WAIT_SECS)
    if filled <= 0:
        log_trade(paths, pos.ticker, "lock_nofill", opp_side, int(opp_ask), None, 0, 0.0,
                  f"lock_profit={locked_profit:.1f}c", oid)
        return False, None

    lock_px = float(vwap or opp_ask)
    exit_fee = calc_taker_fee(int(lock_px), filled)

    gross = (100 - pos.entry_price - lock_px) * filled
    net = gross - pos.entry_fee_c - exit_fee

    hold_secs = int((utc_now() - dt.datetime.fromisoformat(pos.entry_time.replace("Z","+00:00"))).total_seconds())

    closed = ClosedPosition(
        ticker=pos.ticker,
        side=pos.side,
        qty=filled,
        entry_price=pos.entry_price,
        exit_price=lock_px,
        entry_time=pos.entry_time,
        exit_time=utc_now().isoformat(),
        exit_type="lock",
        entry_fee_c=pos.entry_fee_c,
        exit_fee_c=exit_fee,
        gross_pnl_c=gross,
        net_pnl_c=net,
        hold_secs=hold_secs,
        detail=f"lock_profit={gross/filled:.1f}c/ct",
    )
    log_trade(paths, pos.ticker, "lock_fill", opp_side, int(opp_ask), lock_px, filled, exit_fee,
              f"lock_profit={locked_profit:.1f}c", oid)
    return True, closed

# -------------------------
# ENTRY DECISION
# -------------------------

def net_edge_ok(mean_mid: float, cur_mid: float, cur_spread: float, qty: int) -> Tuple[bool, str]:
    """
    Require that the implied reversion distance is large enough to cover:
      - current spread
      - fees (entry+exit, rough estimate at current price)
      - buffer
    """
    edge = abs(cur_mid - mean_mid)
    if edge < MIN_EDGE_C:
        return False, f"edge<{MIN_EDGE_C:.0f} ({edge:.1f})"

    # Rough fee estimate: assume we pay taker fee on entry and exit at ~cur_mid price.
    fee_each = calc_taker_fee(int(round(cur_mid)), qty)
    cost = float(cur_spread) + float(fee_each) * 2.0 + EDGE_BUFFER_C

    if edge < cost:
        return False, f"edge<{cost:.1f} (edge={edge:.1f},spr={cur_spread:.1f},fees~{2*fee_each:.1f})"

    return True, f"edge_ok edge={edge:.1f} cost={cost:.1f}"

# -------------------------
# MAIN
# -------------------------

def main():
    if not SPREAD_TICKERS or "REPLACE_ME" in ",".join(SPREAD_TICKERS):
        raise RuntimeError("Update SPREAD_TICKERS with real Super Bowl spread tickers before running.")

    api_key = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
    key_path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()
    if not api_key or not key_path:
        raise RuntimeError("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")

    private_key = _load_private_key(key_path)
    clock = NFLGameClock()

    base = _mk_log_base()
    paths = _init_logs(base)
    log_event(paths, "start", f"tickers={len(SPREAD_TICKERS)} max_collateral=${MAX_COLLATERAL_DOLLARS:.2f}")

    stats_by = {t: TickerStats() for t in SPREAD_TICKERS}
    cooldowns: Dict[str, dt.datetime] = {}

    open_pos: Optional[OpenPosition] = None
    closed_positions: List[ClosedPosition] = []

    deadline = time.time() + (MAX_RUNTIME_MINS * 60)
    last_heartbeat = time.time()

    print_status(f"[SB spreads] Starting. Max collateral=${MAX_COLLATERAL_DOLLARS:.2f}, target_qty={ORDER_QTY}")
    print_status(f"[SB spreads] Watching {len(SPREAD_TICKERS)} tickers")
    print_status(f"[SB spreads] Maker-only. MAX_ENTRY_SPREAD={MAX_ENTRY_SPREAD_C:.0f}c, MID_RANGE=[{MID_MIN:.0f},{MID_MAX:.0f}], MIN_BID={MIN_BID_C:.0f}c")
    if PREFERRED_SIDE:
        print_status(f"[SB spreads] Preferred side in final mins: {PREFERRED_SIDE.upper()} (flatten opposite)")
    else:
        print_status("[SB spreads] No preferred side (neutral)")

    try:
        while True:
            if time.time() > deadline:
                log_event(paths, "stop", f"max_runtime_mins={MAX_RUNTIME_MINS}")
                print_status("Max runtime reached — stopping.")
                break

            # Kill-switch on total losses (run-level, not per-position stop)
            net_so_far = sum(c.net_pnl_c for c in closed_positions)
            if net_so_far <= -MAX_NET_LOSS_CENTS:
                log_event(paths, "kill_switch", f"net_so_far={net_so_far:.1f}c <= -{MAX_NET_LOSS_CENTS:.1f}c")
                print_status(f"[KILL] net_so_far={net_so_far:.1f}c hit loss limit; stopping.")
                break

            secs_to_close, clock_status = clock.get_secs_to_game_end()

            # Update all tickers
            for t in SPREAD_TICKERS:
                try:
                    ob = fetch_orderbook(t)
                    prices = derive_prices(ob)
                    mid, spr = compute_mid_spread(prices)
                    stats_by[t].update(mid, spr, prices)
                except Exception as e:
                    log_event(paths, "orderbook_error", f"{t}: {e}")
                    continue

            # Pick active ticker
            active_ticker, pick_reason = best_ticker_to_focus(stats_by, cooldowns)
            if active_ticker:
                log_event(paths, "active_ticker", f"{active_ticker} | {pick_reason}")

            # Snapshot all tickers
            now_dt = utc_now()
            for t, s in stats_by.items():
                cd = cooldowns.get(t)
                log_snapshot(
                    paths,
                    t,
                    s.last_prices or {},
                    s,
                    active=(t == active_ticker),
                    pos=open_pos,
                    cooldown_until=cd,
                    secs_to_close=secs_to_close,
                    clock_status=clock_status,
                )

            # Console heartbeat
            if time.time() - last_heartbeat >= HEARTBEAT_SECS:
                last_heartbeat = time.time()
                if active_ticker:
                    s = stats_by[active_ticker]
                    prices = s.last_prices or {}
                    ok, why, mid, spr = current_book_ok(prices)
                    print_status(f"[HB] active={active_ticker} | {pick_reason} | cur_ok={ok} {why} | secs_to_close={secs_to_close} status={clock_status}")
                else:
                    print_status(f"[HB] no active ticker (no candidates). secs_to_close={secs_to_close} status={clock_status}")

            # Manage open position
            if open_pos:
                t = open_pos.ticker
                prices = stats_by[t].last_prices or {}
                s = stats_by[t]

                # 1) Attempt riskless lock first
                _, closed = try_lock(private_key, open_pos, prices, paths)
                if closed:
                    closed_positions.append(closed)
                    log_closed(paths, closed)
                    print_status(f"[LOCK] {t} net={closed.net_pnl_c:.1f}c")
                    open_pos = None
                    time.sleep(POLL_SECS)
                    continue

                # 2) Late-game flatten:
                if secs_to_close is not None and secs_to_close <= FINAL_FLATTEN_SECS:
                    if PREFERRED_SIDE and open_pos.side != PREFERRED_SIDE:
                        closed = exit_sell(private_key, open_pos, prices, "flatten_final_wrong_side", f"secs_to_close={secs_to_close}", paths)
                        if closed:
                            closed_positions.append(closed)
                            log_closed(paths, closed)
                            print_status(f"[FLAT-FINAL] {t} net={closed.net_pnl_c:.1f}c")
                            if closed.net_pnl_c < 0:
                                cooldowns[t] = utc_now() + dt.timedelta(seconds=COOLDOWN_SECS)
                                log_event(paths, "cooldown", f"{t} until {cooldowns[t].isoformat()} (final_loss)")
                            open_pos = None
                            time.sleep(POLL_SECS)
                            continue
                    # If no preferred side, we let MR rule handle; no forced flatten.

                # 3) MR exit: price has reverted toward rolling mean
                mid, _ = compute_mid_spread(prices)
                std = s.recent_std()
                mean = s.recent_mean()
                if mid is not None and std > 0:
                    if open_pos.side == "yes":
                        # We bought YES when mid was below mean. Exit once mid >= mean - MR_TP_SIGMA*std
                        if mid >= mean - MR_TP_SIGMA * std:
                            closed = exit_sell(private_key, open_pos, prices, "mr_revert", f"mid={mid:.1f} mean={mean:.1f} std={std:.1f}", paths)
                            if closed:
                                closed_positions.append(closed)
                                log_closed(paths, closed)
                                print_status(f"[MR-EXIT] {t} net={closed.net_pnl_c:.1f}c")
                                if closed.net_pnl_c < 0:
                                    cooldowns[t] = utc_now() + dt.timedelta(seconds=COOLDOWN_SECS)
                                    log_event(paths, "cooldown", f"{t} until {cooldowns[t].isoformat()} (mr_loss)")
                                open_pos = None
                                time.sleep(POLL_SECS)
                                continue
                    else:
                        # We bought NO when mid was above mean. Exit once mid <= mean + MR_TP_SIGMA*std
                        if mid <= mean + MR_TP_SIGMA * std:
                            closed = exit_sell(private_key, open_pos, prices, "mr_revert", f"mid={mid:.1f} mean={mean:.1f} std={std:.1f}", paths)
                            if closed:
                                closed_positions.append(closed)
                                log_closed(paths, closed)
                                print_status(f"[MR-EXIT] {t} net={closed.net_pnl_c:.1f}c")
                                if closed.net_pnl_c < 0:
                                    cooldowns[t] = utc_now() + dt.timedelta(seconds=COOLDOWN_SECS)
                                    log_event(paths, "cooldown", f"{t} until {cooldowns[t].isoformat()} (mr_loss)")
                                open_pos = None
                                time.sleep(POLL_SECS)
                                continue

            # Entry (only if flat and we have an active ticker)
            if open_pos is None and active_ticker:
                s = stats_by[active_ticker]
                prices = s.last_prices or {}

                # Cooldown check
                cd = cooldowns.get(active_ticker)
                if cd and cd > utc_now():
                    log_event(paths, "skip_entry", f"{active_ticker} cooldown_until={cd.isoformat()}")
                else:
                    ok, why = s.warmup_ok()
                    if not ok:
                        log_event(paths, "skip_entry", f"{active_ticker} {why}")
                    else:
                        std = s.recent_std()
                        mean = s.recent_mean()
                        avg_sp = s.avg_spread()

                        # avg-spread and std filters (coarse)
                        if avg_sp > MAX_AVG_SPREAD_OK or std < MIN_RECENT_STD_OK:
                            log_event(paths, "skip_entry", f"{active_ticker} avg_sp={avg_sp:.1f} std={std:.1f}")
                        else:
                            # Hard current-book gates
                            cur_ok, cur_why, cur_mid, cur_spread = current_book_ok(prices)
                            if not cur_ok or cur_mid is None or cur_spread is None:
                                log_event(paths, "skip_entry", f"{active_ticker} {cur_why}")
                            else:
                                # Net-edge gate (MR: need enough "room" to revert to mean after fees)
                                # We'll compute qty after we know we want to trade.
                                edge_ok, edge_why = net_edge_ok(mean, cur_mid, cur_spread, ORDER_QTY)
                                if not edge_ok:
                                    log_event(paths, "skip_entry", f"{active_ticker} {edge_why}")
                                else:
                                    yes_ask = prices.get("imp_yes_ask")
                                    no_ask = prices.get("imp_no_ask")

                                    # Rough per-contract collateral ≈ price/100
                                    # We cap by MAX_COLLATERAL_DOLLARS and ORDER_QTY
                                    def compute_qty(ask_px: int) -> int:
                                        max_afford = int(MAX_COLLATERAL_DOLLARS * 100.0 / max(1, ask_px))
                                        return max(0, min(ORDER_QTY, max_afford))

                                    if cur_mid < mean and yes_ask is not None:
                                        base_qty = compute_qty(int(yes_ask))
                                        if base_qty <= 0:
                                            log_event(paths, "skip_entry", f"{active_ticker} no_collateral_for_YES ask={yes_ask}")
                                        elif has_fill_liquidity_for_implied_buy(prices, "yes", int(yes_ask), min_qty=base_qty):
                                            reason = f"MR_DOWN mid={cur_mid:.1f} mean={mean:.1f} std={std:.1f} {edge_why}"
                                            filled, vwap = maker_only_buy(private_key, active_ticker, "yes", int(yes_ask), base_qty, reason, paths)
                                            if filled > 0:
                                                px = float(vwap or (int(yes_ask) - MAKER_IMPROVE_CENTS))
                                                fee = calc_taker_fee(int(px), filled)
                                                open_pos = OpenPosition(
                                                    ticker=active_ticker, side="yes", entry_price=px, qty=filled,
                                                    entry_time=utc_now().isoformat(), entry_fee_c=fee, entry_reason=reason
                                                )
                                                print_status(f"[ENTRY] {active_ticker} YES {filled}x @ {px:.1f}c | {reason}")
                                                log_event(paths, "entry", f"{open_pos.ticker} {open_pos.side}@{open_pos.entry_price:.1f} {reason}")

                                    elif cur_mid > mean and no_ask is not None:
                                        base_qty = compute_qty(int(no_ask))
                                        if base_qty <= 0:
                                            log_event(paths, "skip_entry", f"{active_ticker} no_collateral_for_NO ask={no_ask}")
                                        elif has_fill_liquidity_for_implied_buy(prices, "no", int(no_ask), min_qty=base_qty):
                                            reason = f"MR_UP mid={cur_mid:.1f} mean={mean:.1f} std={std:.1f} {edge_why}"
                                            filled, vwap = maker_only_buy(private_key, active_ticker, "no", int(no_ask), base_qty, reason, paths)
                                            if filled > 0:# superbowl_spread_mr.py
#
# Super Bowl LIX - Spread mean reversion harness
# Seattle Seahawks vs New England Patriots
#
# Goals vs spread_beluic_test:
#   - Same "structural" idea (multi-ticker spread watcher)
#   - BUT:
#       * No explicit STOP LOSS
#       * No small TAKE_PROFIT cent target
#       * Exit logic is MR-style (revert toward rolling mean) + optional late flatten
#       * Higher default contract size (ORDER_QTY > 1)
#
# Uses NFLGameClock from superbowl_mr_strategy.py so we know roughly how
# close we are to the end of the game.

import os
import csv
import json
import time
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from dotenv import load_dotenv
load_dotenv()

from combo_vnext import (
    _load_private_key,
    utc_now,
    print_status,
    fetch_orderbook,
    derive_prices,
    place_limit_buy,
    place_limit_sell,
    wait_for_fill_or_timeout,
    has_fill_liquidity_for_implied_buy,
)

from production_strategies import calc_taker_fee
from superbowl_mr_strategy import NFLGameClock

# -------------------------
# SUPER BOWL SPREAD TICKERS
# -------------------------
#
# IMPORTANT: Replace these with the actual Kalshi tickers you want to trade.
# The idea is: cluster around SEA -4.5 (e.g. SEA -3.5, -4.5, -5.5),
# both Seahawks and Patriots sides, so mid has room to move.
#
# Example *placeholder* names — DO NOT USE AS-IS:
#   "KXSBSPREAD-26FEB08SEANE-SEA35",   # SEA -3.5
#   "KXSBSPREAD-26FEB08SEANE-SEA45",   # SEA -4.5 (current line)
#   "KXSBSPREAD-26FEB08SEANE-SEA55",   # SEA -5.5
#   "KXSBSPREAD-26FEB08SEANE-NE35",    # NE +3.5
#   "KXSBSPREAD-26FEB08SEANE-NE45",    # NE +4.5
#   "KXSBSPREAD-26FEB08SEANE-NE55",    # NE +5.5
#
# Paste the *real* tickers from your superbowl_markets.json output.

# -------------------------
# SUPER BOWL SPREAD TICKERS
# -------------------------
#
# Centered around SEA -4.5 with +/- 2 “points” (3, 5, 7)
# SEA = Seahawks side, NE = Patriots side
#
SPREAD_TICKERS: List[str] = [
    "KXNFLSPREAD-26FEB08SEANE-SEA3",
    "KXNFLSPREAD-26FEB08SEANE-SEA5",
    "KXNFLSPREAD-26FEB08SEANE-SEA7",
    "KXNFLSPREAD-26FEB08SEANE-NE3",
    "KXNFLSPREAD-26FEB08SEANE-NE5",
    "KXNFLSPREAD-26FEB08SEANE-NE7",
]


# -------------------------
# CONFIG (env-overridable)
# -------------------------

# Capital / sizing (per *position*; we only hold one at a time)
MAX_COLLATERAL_DOLLARS = float(os.getenv("SB_SPREAD_MAX_COLLATERAL_DOLLARS", "25.0"))

# Higher default size vs BEL/UIC (which was ORDER_QTY=1)
ORDER_QTY = int(os.getenv("SB_SPREAD_ORDER_QTY", "3"))

# Polling
POLL_SECS = float(os.getenv("SB_SPREAD_POLL_SECS", "3.0"))

# Warmup / stats
WARMUP_SAMPLES = int(os.getenv("SB_SPREAD_WARMUP_SAMPLES", "25"))
WARMUP_MIN_RANGE = float(os.getenv("SB_SPREAD_WARMUP_MIN_RANGE", "3.0"))
LOOKBACK = int(os.getenv("SB_SPREAD_LOOKBACK", "30"))

MIN_RECENT_STD_OK = float(os.getenv("SB_SPREAD_MIN_RECENT_STD_OK", "1.5"))
MAX_AVG_SPREAD_OK = float(os.getenv("SB_SPREAD_MAX_AVG_SPREAD_OK", "10.0"))

# HARD book-quality gates (CURRENT snapshot)
MAX_ENTRY_SPREAD_C = float(os.getenv("SB_SPREAD_MAX_ENTRY_SPREAD_C", "8.0"))
MIN_BID_C = float(os.getenv("SB_SPREAD_MIN_BID_C", "5.0"))
MID_MIN = float(os.getenv("SB_SPREAD_MID_MIN", "30.0"))
MID_MAX = float(os.getenv("SB_SPREAD_MID_MAX", "70.0"))

# Maker entry behavior (we stay maker-only for now to keep fees sane)
MAKER_IMPROVE_CENTS = int(os.getenv("SB_SPREAD_MAKER_IMPROVE_CENTS", "1"))
MAKER_WAIT_SECS = int(os.getenv("SB_SPREAD_MAKER_WAIT_SECS", "8"))

# Exit behavior (no hard SL/TP; MR exit + optional flatten)
MR_TP_SIGMA = float(os.getenv("SB_SPREAD_MR_TP_SIGMA", "1.0"))  # how close to mean before we call it "reverted"
FINAL_FLATTEN_SECS = int(os.getenv("SB_SPREAD_FINAL_FLATTEN_SECS", "300"))  # 5 min
PREFERRED_SIDE = os.getenv("SB_SPREAD_PREFERRED_SIDE", "").lower() or None  # "yes" / "no" or ""

# Net-edge gating (tradeability check)
EDGE_BUFFER_C = float(os.getenv("SB_SPREAD_EDGE_BUFFER_C", "3.0"))
MIN_EDGE_C = float(os.getenv("SB_SPREAD_MIN_EDGE_C", "8.0"))

# Cooldown after MR *loss* (we keep this, but exits are MR-based, not cent TP/SL)
COOLDOWN_SECS = int(os.getenv("SB_SPREAD_COOLDOWN_SECS", "180"))

# Auto-stop (kill-switch for this run, *not* per-position stop loss)
MAX_RUNTIME_MINS = int(os.getenv("SB_SPREAD_MAX_RUNTIME_MINS", "240"))
MAX_NET_LOSS_CENTS = float(os.getenv("SB_SPREAD_MAX_NET_LOSS_CENTS", "150.0"))

# Console heartbeat (optional)
HEARTBEAT_SECS = int(os.getenv("SB_SPREAD_HEARTBEAT_SECS", "60"))

# -------------------------
# DATA STRUCTURES
# -------------------------

@dataclass
class OpenPosition:
    ticker: str
    side: str          # "yes" or "no"
    entry_price: float
    qty: int
    entry_time: str
    entry_fee_c: float
    lock_attempts: int = 0
    entry_reason: str = ""

@dataclass
class ClosedPosition:
    ticker: str
    side: str
    qty: int
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    exit_type: str
    entry_fee_c: float
    exit_fee_c: float
    gross_pnl_c: float
    net_pnl_c: float
    hold_secs: int
    detail: str = ""

class TickerStats:
    """
    Rolling stats per ticker: mids, spreads and last prices.
    MR logic uses recent_mean/recent_std on mids.
    """
    def __init__(self, maxlen: int = 240):
        self.mids = deque(maxlen=maxlen)
        self.spreads = deque(maxlen=maxlen)
        self.last_prices: Optional[Dict[str, Any]] = None

    def update(self, mid: Optional[float], spread: Optional[float], prices: Dict[str, Any]):
        if mid is not None:
            self.mids.append(mid)
        if spread is not None:
            self.spreads.append(spread)
        self.last_prices = prices

    def warmup_ok(self) -> Tuple[bool, str]:
        if len(self.mids) < WARMUP_SAMPLES:
            return False, f"warmup:{len(self.mids)}/{WARMUP_SAMPLES}"
        rng = max(self.mids) - min(self.mids)
        if rng < WARMUP_MIN_RANGE:
            return False, f"warmup_range:{rng:.1f}c<{WARMUP_MIN_RANGE}c"
        return True, "ok"

    def recent_std(self) -> float:
        if len(self.mids) < 5:
            return 0.0
        recent = list(self.mids)[-LOOKBACK:] if len(self.mids) >= LOOKBACK else list(self.mids)
        m = sum(recent) / len(recent)
        var = sum((x - m) ** 2 for x in recent) / len(recent)
        return math.sqrt(var) if var > 0 else 0.0

    def recent_mean(self) -> float:
        if len(self.mids) < 5:
            return 50.0
        recent = list(self.mids)[-LOOKBACK:] if len(self.mids) >= LOOKBACK else list(self.mids)
        return sum(recent) / len(recent)

    def avg_spread(self) -> float:
        if not self.spreads:
            return 999.0
        recent = list(self.spreads)[-LOOKBACK:] if len(self.spreads) >= LOOKBACK else list(self.spreads)
        return sum(recent) / len(recent)

# -------------------------
# LOGGING
# -------------------------

def _mk_log_base() -> str:
    os.makedirs("logs", exist_ok=True)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    return f"logs/spreadtest_SB_{ts}"

def _init_logs(base: str) -> Dict[str, str]:
    paths = {
        "snap": f"{base}_snapshots.csv",
        "trades": f"{base}_trades.csv",
        "pos": f"{base}_positions.csv",
        "events": f"{base}_events.csv",
        "summary": f"{base}_summary.json",
    }
    with open(paths["snap"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ts_utc","ticker",
            "yes_bid","yes_ask","no_bid","no_ask",
            "mid","spread",
            "t_std","t_mean","t_avg_spread",
            "active","open_ticker","open_side","open_entry_px","open_qty",
            "cooldown_until_utc","secs_to_close","clock_status",
        ])
    with open(paths["trades"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ts_utc","ticker","action","side","intended_px","fill_px","qty","fee_c","reason","order_id"
        ])
    with open(paths["pos"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ticker","side","qty","entry_px","exit_px","entry_time","exit_time","exit_type",
            "entry_fee_c","exit_fee_c","gross_pnl_c","net_pnl_c","hold_secs","detail"
        ])
    with open(paths["events"], "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["ts_utc","event","detail"])
    return paths

def log_event(paths: Dict[str,str], event: str, detail: str=""):
    with open(paths["events"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([utc_now().isoformat(), event, detail])

def log_snapshot(paths: Dict[str,str], ticker: str, prices: Dict[str,Any], stats: TickerStats,
                 active: bool, pos: Optional[OpenPosition], cooldown_until: Optional[dt.datetime],
                 secs_to_close: Optional[int], clock_status: str):
    yb = prices.get("best_yes_bid")
    ya = prices.get("imp_yes_ask")
    nb = prices.get("best_no_bid")
    na = prices.get("imp_no_ask")

    mid = ""
    spr = ""
    if yb is not None and ya is not None:
        mid = f"{(yb+ya)/2:.1f}"
        spr = f"{(ya-yb):.1f}"

    with open(paths["snap"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            utc_now().isoformat(),
            ticker,
            yb if yb is not None else "",
            ya if ya is not None else "",
            nb if nb is not None else "",
            na if na is not None else "",
            mid,
            spr,
            f"{stats.recent_std():.2f}",
            f"{stats.recent_mean():.2f}",
            f"{stats.avg_spread():.2f}",
            int(active),
            pos.ticker if pos else "",
            pos.side if pos else "",
            f"{pos.entry_price:.2f}" if pos else "",
            pos.qty if pos else "",
            cooldown_until.isoformat() if cooldown_until else "",
            secs_to_close if secs_to_close is not None else "",
            clock_status,
        ])

def log_trade(paths: Dict[str,str], ticker: str, action: str, side: str, intended_px: int,
              fill_px: Optional[float], qty: int, fee_c: float, reason: str, order_id: str):
    with open(paths["trades"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            utc_now().isoformat(), ticker, action, side, intended_px,
            f"{fill_px:.2f}" if fill_px is not None else "",
            qty, f"{fee_c:.2f}", reason, order_id
        ])

def log_closed(paths: Dict[str,str], c: ClosedPosition):
    with open(paths["pos"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            c.ticker, c.side, c.qty, f"{c.entry_price:.2f}", f"{c.exit_price:.2f}",
            c.entry_time, c.exit_time, c.exit_type,
            f"{c.entry_fee_c:.2f}", f"{c.exit_fee_c:.2f}",
            f"{c.gross_pnl_c:.2f}", f"{c.net_pnl_c:.2f}",
            c.hold_secs, c.detail
        ])

# -------------------------
# MARKET HELPERS
# -------------------------

def compute_mid_spread(prices: Dict[str,Any]) -> Tuple[Optional[float], Optional[float]]:
    yb = prices.get("best_yes_bid")
    ya = prices.get("imp_yes_ask")
    if yb is None or ya is None:
        return None, None
    return float((yb + ya)/2.0), float(ya - yb)

def current_book_ok(prices: Dict[str,Any]) -> Tuple[bool, str, Optional[float], Optional[float]]:
    """
    Hard guardrails using CURRENT snapshot to avoid broken books.
    """
    yb = prices.get("best_yes_bid")
    ya = prices.get("imp_yes_ask")
    nb = prices.get("best_no_bid")
    na = prices.get("imp_no_ask")

    if yb is None or ya is None or nb is None or na is None:
        return False, "missing_quotes", None, None

    mid = (yb + ya) / 2.0
    spr = (ya - yb)

    if spr > MAX_ENTRY_SPREAD_C:
        return False, f"cur_spread>{MAX_ENTRY_SPREAD_C:.0f} ({spr:.1f})", float(mid), float(spr)

    if yb < MIN_BID_C or nb < MIN_BID_C:
        return False, f"thin_bids yb={yb} nb={nb}", float(mid), float(spr)

    if mid < MID_MIN or mid > MID_MAX:
        return False, f"mid_outside[{MID_MIN:.0f},{MID_MAX:.0f}] ({mid:.1f})", float(mid), float(spr)

    return True, "ok", float(mid), float(spr)

def best_ticker_to_focus(stats_by: Dict[str,TickerStats], cooldowns: Dict[str, dt.datetime]) -> Tuple[Optional[str], str]:
    """
    Pick the best candidate based on:
      - warmup ok
      - min std
      - reasonable avg spread
      - NOT in cooldown
      - CURRENT book quality ok
    """
    best = None
    best_score = -1e18
    why_best = "no_candidates"

    now = utc_now()

    for t, s in stats_by.items():
        cd = cooldowns.get(t)
        if cd and cd > now:
            continue

        ok, _ = s.warmup_ok()
        if not ok:
            continue

        std = s.recent_std()
        avg_sp = s.avg_spread()

        if avg_sp > MAX_AVG_SPREAD_OK:
            continue
        if std < MIN_RECENT_STD_OK:
            continue

        prices = s.last_prices or {}
        cur_ok, _, _, cur_spr = current_book_ok(prices)
        if not cur_ok or cur_spr is None:
            continue

        # Score: favor movement and tighter spreads
        score = (std * 10.0) - (avg_sp * 2.0) - (cur_spr * 1.0)

        # tiny preference for mid near 50 (more two-sided)
        m = s.recent_mean()
        score -= abs(m - 50.0) * 0.25

        if score > best_score:
            best_score = score
            best = t
            why_best = f"score={best_score:.1f} std={std:.1f} avg_sp={avg_sp:.1f} cur_sp={cur_spr:.1f}"

    return best, why_best

# -------------------------
# EXECUTION HELPERS
# -------------------------

def maker_only_buy(private_key, ticker: str, side: str, implied_ask_px: int, qty: int,
                   reason: str, paths: Dict[str,str]) -> Tuple[int, Optional[float]]:
    """
    Maker-only entry:
      - place 1c better than implied ask (if possible)
      - wait a bit
      - if no fill, do NOTHING (no taker fallback)
    """
    maker_px = max(1, int(implied_ask_px) - MAKER_IMPROVE_CENTS)
    oid = place_limit_buy(private_key, ticker, side, maker_px, qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, side, max_wait_secs=MAKER_WAIT_SECS)
    if filled > 0:
        px = float(vwap or maker_px)
        fee = calc_taker_fee(int(px), filled)
        log_trade(paths, ticker, "entry_fill_maker", side, maker_px, px, filled, fee, reason, oid)
        return filled, vwap

    log_trade(paths, ticker, "entry_nofill_maker", side, maker_px, None, 0, 0.0, reason, oid)
    return 0, None

def exit_sell(private_key, pos: OpenPosition, prices: Dict[str,Any], exit_type: str,
              detail: str, paths: Dict[str,str]) -> Optional[ClosedPosition]:
    bid = prices.get("best_yes_bid") if pos.side == "yes" else prices.get("best_no_bid")
    if bid is None:
        return None

    px = max(1, int(bid))  # MR exit: we don't hyper-cross; no hard SL/TP
    oid = place_limit_sell(private_key, pos.ticker, pos.side, px, pos.qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, pos.side, max_wait_secs=MAKER_WAIT_SECS)
    if filled <= 0:
        log_trade(paths, pos.ticker, f"exit_{exit_type}_nofill", pos.side, px, None, 0, 0.0, detail, oid)
        return None

    exit_px = float(vwap or px)
    exit_fee = calc_taker_fee(int(exit_px), filled)

    gross = (exit_px - pos.entry_price) * filled
    net = gross - pos.entry_fee_c - exit_fee

    hold_secs = int((utc_now() - dt.datetime.fromisoformat(pos.entry_time.replace("Z","+00:00"))).total_seconds())

    closed = ClosedPosition(
        ticker=pos.ticker,
        side=pos.side,
        qty=filled,
        entry_price=pos.entry_price,
        exit_price=exit_px,
        entry_time=pos.entry_time,
        exit_time=utc_now().isoformat(),
        exit_type=exit_type,
        entry_fee_c=pos.entry_fee_c,
        exit_fee_c=exit_fee,
        gross_pnl_c=gross,
        net_pnl_c=net,
        hold_secs=hold_secs,
        detail=detail,
    )
    log_trade(paths, pos.ticker, f"exit_{exit_type}", pos.side, px, exit_px, filled, exit_fee, detail, oid)
    return closed

def try_lock(private_key, pos: OpenPosition, prices: Dict[str,Any], paths: Dict[str,str]) -> Tuple[bool, Optional[ClosedPosition]]:
    """
    Optional lock (riskless-ish close) if both sides are playable.
    This is the one place where we are allowed to "get out early" — but only
    if it's a true lock-in profit based on prices.
    """
    if pos.lock_attempts >= 2:
        return False, None

    if pos.side == "yes":
        opp_side = "no"
        opp_ask = prices.get("imp_no_ask")
    else:
        opp_side = "yes"
        opp_ask = prices.get("imp_yes_ask")

    if opp_ask is None:
        return False, None

    locked_profit = 100 - pos.entry_price - float(opp_ask)
    if locked_profit < 4.0:
        return False, None

    if not has_fill_liquidity_for_implied_buy(prices, opp_side, int(opp_ask), min_qty=pos.qty):
        return False, None

    pos.lock_attempts += 1

    oid = place_limit_buy(private_key, pos.ticker, opp_side, int(opp_ask), pos.qty)
    filled, vwap = wait_for_fill_or_timeout(private_key, oid, opp_side, max_wait_secs=MAKER_WAIT_SECS)
    if filled <= 0:
        log_trade(paths, pos.ticker, "lock_nofill", opp_side, int(opp_ask), None, 0, 0.0,
                  f"lock_profit={locked_profit:.1f}c", oid)
        return False, None

    lock_px = float(vwap or opp_ask)
    exit_fee = calc_taker_fee(int(lock_px), filled)

    gross = (100 - pos.entry_price - lock_px) * filled
    net = gross - pos.entry_fee_c - exit_fee

    hold_secs = int((utc_now() - dt.datetime.fromisoformat(pos.entry_time.replace("Z","+00:00"))).total_seconds())

    closed = ClosedPosition(
        ticker=pos.ticker,
        side=pos.side,
        qty=filled,
        entry_price=pos.entry_price,
        exit_price=lock_px,
        entry_time=pos.entry_time,
        exit_time=utc_now().isoformat(),
        exit_type="lock",
        entry_fee_c=pos.entry_fee_c,
        exit_fee_c=exit_fee,
        gross_pnl_c=gross,
        net_pnl_c=net,
        hold_secs=hold_secs,
        detail=f"lock_profit={gross/filled:.1f}c/ct",
    )
    log_trade(paths, pos.ticker, "lock_fill", opp_side, int(opp_ask), lock_px, filled, exit_fee,
              f"lock_profit={locked_profit:.1f}c", oid)
    return True, closed

# -------------------------
# ENTRY DECISION
# -------------------------

def net_edge_ok(mean_mid: float, cur_mid: float, cur_spread: float, qty: int) -> Tuple[bool, str]:
    """
    Require that the implied reversion distance is large enough to cover:
      - current spread
      - fees (entry+exit, rough estimate at current price)
      - buffer
    """
    edge = abs(cur_mid - mean_mid)
    if edge < MIN_EDGE_C:
        return False, f"edge<{MIN_EDGE_C:.0f} ({edge:.1f})"

    # Rough fee estimate: assume we pay taker fee on entry and exit at ~cur_mid price.
    fee_each = calc_taker_fee(int(round(cur_mid)), qty)
    cost = float(cur_spread) + float(fee_each) * 2.0 + EDGE_BUFFER_C

    if edge < cost:
        return False, f"edge<{cost:.1f} (edge={edge:.1f},spr={cur_spread:.1f},fees~{2*fee_each:.1f})"

    return True, f"edge_ok edge={edge:.1f} cost={cost:.1f}"

# -------------------------
# MAIN
# -------------------------

def main():
    if not SPREAD_TICKERS or "REPLACE_ME" in ",".join(SPREAD_TICKERS):
        raise RuntimeError("Update SPREAD_TICKERS with real Super Bowl spread tickers before running.")

    api_key = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
    key_path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()
    if not api_key or not key_path:
        raise RuntimeError("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")

    private_key = _load_private_key(key_path)
    clock = NFLGameClock()

    base = _mk_log_base()
    paths = _init_logs(base)
    log_event(paths, "start", f"tickers={len(SPREAD_TICKERS)} max_collateral=${MAX_COLLATERAL_DOLLARS:.2f}")

    stats_by = {t: TickerStats() for t in SPREAD_TICKERS}
    cooldowns: Dict[str, dt.datetime] = {}

    open_pos: Optional[OpenPosition] = None
    closed_positions: List[ClosedPosition] = []

    deadline = time.time() + (MAX_RUNTIME_MINS * 60)
    last_heartbeat = time.time()

    print_status(f"[SB spreads] Starting. Max collateral=${MAX_COLLATERAL_DOLLARS:.2f}, target_qty={ORDER_QTY}")
    print_status(f"[SB spreads] Watching {len(SPREAD_TICKERS)} tickers")
    print_status(f"[SB spreads] Maker-only. MAX_ENTRY_SPREAD={MAX_ENTRY_SPREAD_C:.0f}c, MID_RANGE=[{MID_MIN:.0f},{MID_MAX:.0f}], MIN_BID={MIN_BID_C:.0f}c")
    if PREFERRED_SIDE:
        print_status(f"[SB spreads] Preferred side in final mins: {PREFERRED_SIDE.upper()} (flatten opposite)")
    else:
        print_status("[SB spreads] No preferred side (neutral)")

    try:
        while True:
            if time.time() > deadline:
                log_event(paths, "stop", f"max_runtime_mins={MAX_RUNTIME_MINS}")
                print_status("Max runtime reached — stopping.")
                break

            # Kill-switch on total losses (run-level, not per-position stop)
            net_so_far = sum(c.net_pnl_c for c in closed_positions)
            if net_so_far <= -MAX_NET_LOSS_CENTS:
                log_event(paths, "kill_switch", f"net_so_far={net_so_far:.1f}c <= -{MAX_NET_LOSS_CENTS:.1f}c")
                print_status(f"[KILL] net_so_far={net_so_far:.1f}c hit loss limit; stopping.")
                break

            secs_to_close, clock_status = clock.get_secs_to_game_end()

            # Update all tickers
            for t in SPREAD_TICKERS:
                try:
                    ob = fetch_orderbook(t)
                    prices = derive_prices(ob)
                    mid, spr = compute_mid_spread(prices)
                    stats_by[t].update(mid, spr, prices)
                except Exception as e:
                    log_event(paths, "orderbook_error", f"{t}: {e}")
                    continue

            # Pick active ticker
            active_ticker, pick_reason = best_ticker_to_focus(stats_by, cooldowns)
            if active_ticker:
                log_event(paths, "active_ticker", f"{active_ticker} | {pick_reason}")

            # Snapshot all tickers
            now_dt = utc_now()
            for t, s in stats_by.items():
                cd = cooldowns.get(t)
                log_snapshot(
                    paths,
                    t,
                    s.last_prices or {},
                    s,
                    active=(t == active_ticker),
                    pos=open_pos,
                    cooldown_until=cd,
                    secs_to_close=secs_to_close,
                    clock_status=clock_status,
                )

            # Console heartbeat
            if time.time() - last_heartbeat >= HEARTBEAT_SECS:
                last_heartbeat = time.time()
                if active_ticker:
                    s = stats_by[active_ticker]
                    prices = s.last_prices or {}
                    ok, why, mid, spr = current_book_ok(prices)
                    print_status(f"[HB] active={active_ticker} | {pick_reason} | cur_ok={ok} {why} | secs_to_close={secs_to_close} status={clock_status}")
                else:
                    print_status(f"[HB] no active ticker (no candidates). secs_to_close={secs_to_close} status={clock_status}")

            # Manage open position
            if open_pos:
                t = open_pos.ticker
                prices = stats_by[t].last_prices or {}
                s = stats_by[t]

                # 1) Attempt riskless lock first
                _, closed = try_lock(private_key, open_pos, prices, paths)
                if closed:
                    closed_positions.append(closed)
                    log_closed(paths, closed)
                    print_status(f"[LOCK] {t} net={closed.net_pnl_c:.1f}c")
                    open_pos = None
                    time.sleep(POLL_SECS)
                    continue

                # 2) Late-game flatten:
                if secs_to_close is not None and secs_to_close <= FINAL_FLATTEN_SECS:
                    if PREFERRED_SIDE and open_pos.side != PREFERRED_SIDE:
                        closed = exit_sell(private_key, open_pos, prices, "flatten_final_wrong_side", f"secs_to_close={secs_to_close}", paths)
                        if closed:
                            closed_positions.append(closed)
                            log_closed(paths, closed)
                            print_status(f"[FLAT-FINAL] {t} net={closed.net_pnl_c:.1f}c")
                            if closed.net_pnl_c < 0:
                                cooldowns[t] = utc_now() + dt.timedelta(seconds=COOLDOWN_SECS)
                                log_event(paths, "cooldown", f"{t} until {cooldowns[t].isoformat()} (final_loss)")
                            open_pos = None
                            time.sleep(POLL_SECS)
                            continue
                    # If no preferred side, we let MR rule handle; no forced flatten.

                # 3) MR exit: price has reverted toward rolling mean
                mid, _ = compute_mid_spread(prices)
                std = s.recent_std()
                mean = s.recent_mean()
                if mid is not None and std > 0:
                    if open_pos.side == "yes":
                        # We bought YES when mid was below mean. Exit once mid >= mean - MR_TP_SIGMA*std
                        if mid >= mean - MR_TP_SIGMA * std:
                            closed = exit_sell(private_key, open_pos, prices, "mr_revert", f"mid={mid:.1f} mean={mean:.1f} std={std:.1f}", paths)
                            if closed:
                                closed_positions.append(closed)
                                log_closed(paths, closed)
                                print_status(f"[MR-EXIT] {t} net={closed.net_pnl_c:.1f}c")
                                if closed.net_pnl_c < 0:
                                    cooldowns[t] = utc_now() + dt.timedelta(seconds=COOLDOWN_SECS)
                                    log_event(paths, "cooldown", f"{t} until {cooldowns[t].isoformat()} (mr_loss)")
                                open_pos = None
                                time.sleep(POLL_SECS)
                                continue
                    else:
                        # We bought NO when mid was above mean. Exit once mid <= mean + MR_TP_SIGMA*std
                        if mid <= mean + MR_TP_SIGMA * std:
                            closed = exit_sell(private_key, open_pos, prices, "mr_revert", f"mid={mid:.1f} mean={mean:.1f} std={std:.1f}", paths)
                            if closed:
                                closed_positions.append(closed)
                                log_closed(paths, closed)
                                print_status(f"[MR-EXIT] {t} net={closed.net_pnl_c:.1f}c")
                                if closed.net_pnl_c < 0:
                                    cooldowns[t] = utc_now() + dt.timedelta(seconds=COOLDOWN_SECS)
                                    log_event(paths, "cooldown", f"{t} until {cooldowns[t].isoformat()} (mr_loss)")
                                open_pos = None
                                time.sleep(POLL_SECS)
                                continue

            # Entry (only if flat and we have an active ticker)
            if open_pos is None and active_ticker:
                s = stats_by[active_ticker]
                prices = s.last_prices or {}

                # Cooldown check
                cd = cooldowns.get(active_ticker)
                if cd and cd > utc_now():
                    log_event(paths, "skip_entry", f"{active_ticker} cooldown_until={cd.isoformat()}")
                else:
                    ok, why = s.warmup_ok()
                    if not ok:
                        log_event(paths, "skip_entry", f"{active_ticker} {why}")
                    else:
                        std = s.recent_std()
                        mean = s.recent_mean()
                        avg_sp = s.avg_spread()

                        # avg-spread and std filters (coarse)
                        if avg_sp > MAX_AVG_SPREAD_OK or std < MIN_RECENT_STD_OK:
                            log_event(paths, "skip_entry", f"{active_ticker} avg_sp={avg_sp:.1f} std={std:.1f}")
                        else:
                            # Hard current-book gates
                            cur_ok, cur_why, cur_mid, cur_spread = current_book_ok(prices)
                            if not cur_ok or cur_mid is None or cur_spread is None:
                                log_event(paths, "skip_entry", f"{active_ticker} {cur_why}")
                            else:
                                # Net-edge gate (MR: need enough "room" to revert to mean after fees)
                                # We'll compute qty after we know we want to trade.
                                edge_ok, edge_why = net_edge_ok(mean, cur_mid, cur_spread, ORDER_QTY)
                                if not edge_ok:
                                    log_event(paths, "skip_entry", f"{active_ticker} {edge_why}")
                                else:
                                    yes_ask = prices.get("imp_yes_ask")
                                    no_ask = prices.get("imp_no_ask")

                                    # Rough per-contract collateral ≈ price/100
                                    # We cap by MAX_COLLATERAL_DOLLARS and ORDER_QTY
                                    def compute_qty(ask_px: int) -> int:
                                        max_afford = int(MAX_COLLATERAL_DOLLARS * 100.0 / max(1, ask_px))
                                        return max(0, min(ORDER_QTY, max_afford))

                                    if cur_mid < mean and yes_ask is not None:
                                        base_qty = compute_qty(int(yes_ask))
                                        if base_qty <= 0:
                                            log_event(paths, "skip_entry", f"{active_ticker} no_collateral_for_YES ask={yes_ask}")
                                        elif has_fill_liquidity_for_implied_buy(prices, "yes", int(yes_ask), min_qty=base_qty):
                                            reason = f"MR_DOWN mid={cur_mid:.1f} mean={mean:.1f} std={std:.1f} {edge_why}"
                                            filled, vwap = maker_only_buy(private_key, active_ticker, "yes", int(yes_ask), base_qty, reason, paths)
                                            if filled > 0:
                                                px = float(vwap or (int(yes_ask) - MAKER_IMPROVE_CENTS))
                                                fee = calc_taker_fee(int(px), filled)
                                                open_pos = OpenPosition(
                                                    ticker=active_ticker, side="yes", entry_price=px, qty=filled,
                                                    entry_time=utc_now().isoformat(), entry_fee_c=fee, entry_reason=reason
                                                )
                                                print_status(f"[ENTRY] {active_ticker} YES {filled}x @ {px:.1f}c | {reason}")
                                                log_event(paths, "entry", f"{open_pos.ticker} {open_pos.side}@{open_pos.entry_price:.1f} {reason}")

                                    elif cur_mid > mean and no_ask is not None:
                                        base_qty = compute_qty(int(no_ask))
                                        if base_qty <= 0:
                                            log_event(paths, "skip_entry", f"{active_ticker} no_collateral_for_NO ask={no_ask}")
                                        elif has_fill_liquidity_for_implied_buy(prices, "no", int(no_ask), min_qty=base_qty):
                                            reason = f"MR_UP mid={cur_mid:.1f} mean={mean:.1f} std={std:.1f} {edge_why}"
                                            filled, vwap = maker_only_buy(private_key, active_ticker, "no", int(no_ask), base_qty, reason, paths)
                                            if filled > 0:
                                                px = float(vwap or (int(no_ask) - MAKER_IMPROVE_CENTS))
                                                fee = calc_taker_fee(int(px), filled)
                                                open_pos = OpenPosition(
                                                    ticker=active_ticker, side="no", entry_price=px, qty=filled,
                                                    entry_time=utc_now().isofee, entry_reason=reason
                                                )
                                                print_status(f"[ENTRY] {active_ticker} NO {filled}x @ {px:.1f}c | {reason}")
                                                log_event(paths, "entry", f"{open_pos.ticker} {open_pos.side}@{open_pos.entry_price:.1f} {reason}")
                                    else:
                                        log_event(paths, "skip_entry", f"{active_ticker} no_direction cur_mid={cur_mid:.1f} mean={mean:.1f}")

            time.sleep(POLL_SECS)

    except KeyboardInterrupt:
        print_status("\n\n⚠ Interrupted by user")

    # -------------------------
    # SUMMARY
    # -------------------------
    net = sum(c.net_pnl_c for c in closed_positions)
    fees = sum(c.entry_fee_c + c.exit_fee_c for c in closed_positions)
    wins = sum(1 for c in closed_positions if c.net_pnl_c > 0)
    losses = sum(1 for c in closed_positions if c.net_pnl_c <= 0)

    summary = {
        "tickers": SPREAD_TICKERS,
        "max_collateral_dollars": MAX_COLLATERAL_DOLLARS,
        "order_qty_target": ORDER_QTY,
        "trades": len(closed_positions),
        "wins": wins,
        "losses": losses,
        "net_pnl_c": net,
        "net_pnl_dollars": net/100.0,
        "fees_c": fees,
        "closed_positions": [c.__dict__ for c in closed_positions],
        "params": {
            "POLL_SECS": POLL_SECS,
            "WARMUP_SAMPLES": WARMUP_SAMPLES,
            "WARMUP_MIN_RANGE": WARMUP_MIN_RANGE,
            "LOOKBACK": LOOKBACK,
            "MIN_RECENT_STD_OK": MIN_RECENT_STD_OK,
            "MAX_AVG_SPREAD_OK": MAX_AVG_SPREAD_OK,
            "MAX_ENTRY_SPREAD_C": MAX_ENTRY_SPREAD_C,
            "MIN_BID_C": MIN_BID_C,
            "MID_RANGE": [MID_MIN, MID_MAX],
            "MR_TP_SIGMA": MR_TP_SIGMA,
            "EDGE_BUFFER_C": EDGE_BUFFER_C,
            "MIN_EDGE_C": MIN_EDGE_C,
            "COOLDOWN_SECS": COOLDOWN_SECS,
            "MAX_NET_LOSS_CENTS": MAX_NET_LOSS_CENTS,
            "FINAL_FLATTEN_SECS": FINAL_FLATTEN_SECS,
            "PREFERRED_SIDE": PREFERRED_SIDE,
        }
    }
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_status("="*80)
    print_status(f"[SUMMARY] trades={len(closed_positions)} wins={wins} losses={losses} net={net:.1f}c (${net/100:.2f}) fees={fees:.1f}c")
    print_status(f"[SUMMARY] summary_json={paths['summary']}")
    print_status("="*80)

if __name__ == "__main__":
    main()

