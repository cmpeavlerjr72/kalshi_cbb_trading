# production_strategies.py
# Updated Feb 23, 2026 — strategy refactoring
#
# Strategies:
#   - MeanReversionStrategy: refined with mean-shift exit, multi-TF trend, tiered trailing stop
#   - PairedMeanReversionStrategy: outcome-independent (buy YES + NO when cheap)
#   - FinalMinutesStrategy: late-game trades when ESPN WP > 90%
#   - ExposureTracker: shared capital guard across strategies
#
# Removed: ModelEdge, SpreadCapture, PregameAnchored (all unprofitable)

import os
import sys
import time
import math
import csv
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import deque
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from combo_vnext import (
    _load_private_key,
    _get,
    get_markets_in_series,
    parse_iso,
    utc_now,
    print_status,
    fetch_orderbook,
    derive_prices,
    place_limit_buy,
    place_limit_sell,
    wait_for_fill_or_timeout,
    has_fill_liquidity_for_implied_buy,
    cancel_order,
    fetch_fills_for_order,
    SERIES_TICKER,
    MIN_LIQUIDITY_CONTRACTS,
)
from fetch_kalshi_ledger import fetch_positions

# =============================================================================
# FEE CALCULATIONS (P5)
# =============================================================================

def calc_taker_fee(price_cents: int, qty: int = 1) -> float:
    """Kalshi taker fee in cents: 0.07 × P × (1−P), max 2¢ @ 50¢"""
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    p = price_cents / 100.0
    fee_per = min(0.07 * p * (1 - p), 0.02)
    return fee_per * qty * 100



# =============================================================================
# MARKET QUALITY FILTER (P3)
# =============================================================================

class MarketQualityMonitor:
    """
    P3: Tracks market quality metrics over a warmup window.
    Allows strategies to skip dead/illiquid markets.
    """

    def __init__(self, warmup_samples: int = 40, warmup_secs: int = 180,
                 std_min: float = 3.0, range_min: float = 5.0, spread_max: float = 6.0):
        self.warmup_samples = warmup_samples
        self.warmup_secs = warmup_secs
        self.std_min = std_min
        self.range_min = range_min
        self.spread_max = spread_max
        self.midpoints: deque = deque(maxlen=200)
        self.spreads: deque = deque(maxlen=200)
        self.first_sample_time: Optional[float] = None
        self._quality_decided = False
        self._quality_ok = False
        self._quality_reason = ""

    def update(self, mid: Optional[float], spread: Optional[int]):
        if self.first_sample_time is None:
            self.first_sample_time = time.time()
        if mid is not None:
            self.midpoints.append(mid)
        if spread is not None:
            self.spreads.append(spread)

    def is_warmup_complete(self) -> bool:
        if self.first_sample_time is None:
            return False
        elapsed = time.time() - self.first_sample_time
        return len(self.midpoints) >= self.warmup_samples and elapsed >= self.warmup_secs

    def is_tradeable(self) -> Tuple[bool, str]:
        """
        Returns (tradeable, reason).
        Only decides once after warmup, then caches.
        Re-evaluates periodically for ongoing dead-market detection.
        """
        if not self.is_warmup_complete():
            return False, f"warmup:{len(self.midpoints)}/{self.warmup_samples}"

        prices = list(self.midpoints)
        if len(prices) < 10:
            return False, "insufficient_data"

        # Metrics over recent window
        recent = prices[-60:] if len(prices) >= 60 else prices
        price_range = max(recent) - min(recent)
        mean = sum(recent) / len(recent)
        var = sum((p - mean) ** 2 for p in recent) / len(recent)
        std = math.sqrt(var) if var > 0 else 0

        # Spread check
        recent_spreads = list(self.spreads)[-30:] if len(self.spreads) >= 30 else list(self.spreads)
        avg_spread = sum(recent_spreads) / len(recent_spreads) if recent_spreads else 99

        # P3 criteria: skip if std/range/spread outside thresholds
        if std < self.std_min:
            return False, f"dead_market:std={std:.1f}c<{self.std_min}"
        if price_range < self.range_min:
            return False, f"dead_market:range={price_range:.1f}c<{self.range_min}"
        if avg_spread > self.spread_max:
            return False, f"illiquid:avg_spread={avg_spread:.1f}c>{self.spread_max}"

        return True, f"ok:std={std:.1f},range={price_range:.1f},spread={avg_spread:.1f}"

    def recent_volatility(self) -> float:
        """Return std of last 30 midpoints"""
        if len(self.midpoints) < 5:
            return 0.0
        recent = list(self.midpoints)[-30:]
        mean = sum(recent) / len(recent)
        var = sum((p - mean) ** 2 for p in recent) / len(recent)
        return math.sqrt(var) if var > 0 else 0.0

    def has_recent_movement(self, lookback: int = 30, min_move: float = 3.0) -> bool:
        """P2: Dead market detection — any >min_move¢ swing in last N samples?"""
        if len(self.midpoints) < lookback:
            return True  # benefit of doubt during warmup
        recent = list(self.midpoints)[-lookback:]
        return (max(recent) - min(recent)) >= min_move


# =============================================================================
# EXPOSURE TRACKER — shared capital guard across strategies within one game
# =============================================================================

class ExposureTracker:
    """
    Thread-safe shared capital tracker.
    Multiple strategies call try_reserve() before sizing entries;
    the tracker ensures total exposure across all strategies stays
    within max_exposure_dollars for the game.
    """

    def __init__(self, max_exposure_dollars: float):
        self.max_exposure = max_exposure_dollars
        self._used = 0.0
        self._lock = threading.Lock()

    def try_reserve(self, amount: float) -> bool:
        with self._lock:
            if self._used + amount <= self.max_exposure:
                self._used += amount
                return True
            return False

    def release(self, amount: float):
        with self._lock:
            self._used -= amount
            self._used = max(0.0, self._used)

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.max_exposure - self._used)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    id: str
    strategy: str
    side: str          # "yes" or "no"
    entry_price: float
    qty: int
    entry_time: dt.datetime
    entry_reason: str
    entry_fee: float = 0.0
    lock_attempts: int = 0  # P5: track lock attempts
        # NEW: MR profit-defense tracking (per-contract cents)
    max_fav_pnl_cents: float = 0.0
    max_fav_ts: Optional[str] = None


@dataclass
class ClosedPosition:
    id: str
    strategy: str
    side: str
    entry_price: float
    exit_price: float
    qty: int
    entry_time: str
    exit_time: str
    exit_type: str      # "lock", "stop", "take_profit", "timeout"
    entry_fee: float
    exit_fee: float
    gross_pnl: float
    net_pnl: float
    hold_secs: int
    lock_price: Optional[float] = None


# =============================================================================
# BASE STRATEGY
# =============================================================================

class BaseStrategy(ABC):

    def __init__(self, name: str, max_capital: float, params: Dict[str, Any]):
        self.name = name
        self.max_capital = max_capital
        self.params = params

        self.positions: List[Position] = []
        self.closed: List[ClosedPosition] = []
        self.capital_used = 0.0
        self.position_counter = 0

        # Throttling
        self.last_entry_time: Optional[float] = None
        self.min_entry_gap_secs = params.get("min_entry_gap_secs", 45)

        # P1: Per-side re-entry cooldown tracking
        self._side_last_stop: Dict[str, float] = {}  # side -> timestamp of last stop
        self._side_cooldown_secs = params.get("side_cooldown_secs", 120)

        self.total_fees = 0.0

    def _next_position_id(self) -> str:
        self.position_counter += 1
        return f"{self.name}_{self.position_counter}"

    def can_enter(self, secs_to_close: int = 9999) -> Tuple[bool, str]:
        # Capital limit
        if self.capital_used >= self.max_capital:
            return False, "max_capital"

        # Position limit
        if len(self.positions) >= self.params.get("max_positions", 2):
            return False, "max_positions"

        # P1: Final 5-minute lockout (configurable)
        lockout_secs = self.params.get("stop_trading_before_close_secs", 300)
        if secs_to_close < lockout_secs:
            return False, f"near_close:{int(secs_to_close)}s"

        # General throttle
        if self.last_entry_time:
            elapsed = time.time() - self.last_entry_time
            if elapsed < self.min_entry_gap_secs:
                return False, f"throttle:{elapsed:.0f}s"

        return True, "ok"

    def update_params(self, overrides: dict):
        """Hot-reload: merge new param values and rebuild dependent state."""
        self.params.update(overrides)

        # Rebuild cached scalars from params
        if "min_entry_gap_secs" in overrides:
            self.min_entry_gap_secs = overrides["min_entry_gap_secs"]
        if "side_cooldown_secs" in overrides:
            self._side_cooldown_secs = overrides["side_cooldown_secs"]

        # Rebuild deques with correct maxlen (preserving existing data)
        if "lookback" in overrides and hasattr(self, "prices"):
            old = list(self.prices)
            self.prices = deque(old, maxlen=self.params["lookback"])
        if "long_lookback" in overrides and hasattr(self, "prices_long"):
            old = list(self.prices_long)
            self.prices_long = deque(old, maxlen=self.params["long_lookback"])

    def _side_on_cooldown(self, side: str) -> Tuple[bool, str]:
        """P1: Check if a side is on post-stop cooldown"""
        last_stop = self._side_last_stop.get(side)
        if last_stop is None:
            return False, "ok"
        elapsed = time.time() - last_stop
        if elapsed < self._side_cooldown_secs:
            return True, f"side_cooldown:{side}:{elapsed:.0f}s/{self._side_cooldown_secs}s"
        return False, "ok"
    
    def entry_wait_secs(self, context: Dict[str, Any]) -> int:
        """How long the GameRunner should wait for an entry fill before cancel."""
        return 15

    def on_entry_result(self, filled: bool, context: Dict[str, Any]) -> None:
        """Called by GameRunner after an entry attempt, even if no fill."""
        return

    @abstractmethod
    def evaluate_entry(
        self,
        prices: Dict[str, Any],
        secs_to_close: int,
        context: Dict[str, Any],
    ) -> Tuple[bool, str, int, int, str]:
        """Returns: (should_enter, side, price, qty, reason)"""
        pass

    @abstractmethod
    def evaluate_exit(
        self,
        position: Position,
        prices: Dict[str, Any],
        secs_to_close: int,
        context: Dict[str, Any],
    ) -> Tuple[bool, str, int, str]:
        """Returns: (should_exit, exit_type, exit_price, reason)"""
        pass

    def record_entry(self, side: str, price: float, qty: int, reason: str) -> Position:
        fee = calc_taker_fee(int(price), qty)
        self.total_fees += fee

        pos = Position(
            id=self._next_position_id(),
            strategy=self.name,
            side=side,
            entry_price=price,
            qty=qty,
            entry_time=utc_now(),
            entry_reason=reason,
            entry_fee=fee,
        )

        self.positions.append(pos)
        self.last_entry_time = time.time()

        # Cash required = price you pay, for BOTH sides
        self.capital_used += price * qty / 100

        return pos

    def record_exit(
        self,
        position: Position,
        exit_type: str,
        exit_price: float,
        lock_price: Optional[float] = None,
    ) -> ClosedPosition:
        exit_fee = calc_taker_fee(int(exit_price), position.qty)
        self.total_fees += exit_fee

        if exit_type == "lock" and lock_price is not None:
            gross_pnl = (100 - position.entry_price - lock_price) * position.qty
        else:
            gross_pnl = (exit_price - position.entry_price) * position.qty

        net_pnl = gross_pnl - position.entry_fee - exit_fee

        now = utc_now()
        hold_secs = int((now - position.entry_time).total_seconds())

        closed = ClosedPosition(
            id=position.id,
            strategy=self.name,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            qty=position.qty,
            entry_time=position.entry_time.isoformat(),
            exit_time=now.isoformat(),
            exit_type=exit_type,
            entry_fee=position.entry_fee,
            exit_fee=exit_fee,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            hold_secs=hold_secs,
            lock_price=lock_price,
        )

        self.closed.append(closed)
        self.positions.remove(position)

        # Free capital — matches the price we paid on entry
        self.capital_used -= position.entry_price * position.qty / 100
        self.capital_used = max(0, self.capital_used)

        # P1: Record stop-out for cooldown
        if exit_type == "stop":
            self._side_last_stop[position.side] = time.time()

        return closed

    def get_stats(self) -> Dict[str, Any]:
        if not self.closed:
            return {
                "strategy": self.name,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "locks": 0,
                "stops": 0,
                "lock_rate": 0.0,
                "gross_pnl": 0,
                "fees": self.total_fees,
                "net_pnl": -self.total_fees,
            }

        gross = sum(c.gross_pnl for c in self.closed)
        net = sum(c.net_pnl for c in self.closed)
        wins = [c for c in self.closed if c.net_pnl > 0]
        losses = [c for c in self.closed if c.net_pnl <= 0]
        locks = [c for c in self.closed if c.exit_type == "lock"]
        stops = [c for c in self.closed if c.exit_type == "stop"]

        return {
            "strategy": self.name,
            "trades": len(self.closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.closed),
            "locks": len(locks),
            "stops": len(stops),
            "lock_rate": len(locks) / len(self.closed) if self.closed else 0,
            "gross_pnl": gross,
            "fees": self.total_fees,
            "net_pnl": net,
            "avg_hold_secs": sum(c.hold_secs for c in self.closed) / len(self.closed),
            "open_positions": len(self.positions),
        }


# =============================================================================
# STRATEGY: MEAN REVERSION (P2 adaptive threshold + dead market detection)
# =============================================================================

class MeanReversionStrategy(BaseStrategy):
    """
    MR with:
      - stop loss OFF
      - dynamic sigma thresholds (existing)
      - late-game directional behavior:
          * within last N secs, ONLY enter preferred side
          * within last N secs, flatten any wrong-side open inventory
      - size-aware entries: qty scales to remaining allocated capital
      - 2a: dynamic PnL floor (mean-shift exit when thesis collapses)
      - 2b: multi-timeframe trend validation
      - 2c: tiered trailing stop (tighter at higher peaks)
      - 2d: shared exposure tracker integration
    """

    def __init__(self, max_capital: float, preferred_side: Optional[str] = None,
                 exposure_tracker: Optional["ExposureTracker"] = None):
        params = {
            "lookback": 120,
            "high_vol_std_mult": 1.5,
            "low_vol_std_mult": 2.5,
            "low_vol_cutoff": 5.0,

            # Revert exit: proportional to volatility (0.5σ, min 2c)
            "revert_sigma_frac": 0.5,
            "revert_min_cents": 2.0,

            # stop loss intentionally unused / OFF
            "stop_loss": 0,

            # allow more room; you can tune
            "max_positions": 4,

            "min_entry_gap_secs": 8,
            "side_cooldown_secs": 20,

            # IMPORTANT: allow entries into late game; directional filter controls safety
            "stop_trading_before_close_secs": 0,

            # warmup / dead-market checks
            "warmup_samples": 120,
            "warmup_min_range": 5.0,
            "dead_lookback": 30,
            "dead_min_move": 3.0,

            # NEW: direction window (last 5 minutes)
            "directional_close_secs": 300,
            "force_exit_wrong_side": True,

            # --- SIZING ---
            "size_base_frac": 0.35,
            "size_max_frac": 0.80,
            "size_sigma_floor": 1.5,
            "size_sigma_cap": 3.0,
            "size_min_qty": 3,
            "max_order_qty": 200,

            # PROFIT DEFENSE (giveback cap) — tiered (2c)
            "profit_defense_activate_cents": 10.0,
            "profit_defense_giveback_frac": 0.50,       # 50% giveback for 10-15c peaks
            "profit_defense_high_threshold": 15.0,       # above this → tighter giveback
            "profit_defense_giveback_frac_high": 0.30,   # 30% giveback for >15c peaks
            "profit_defense_min_keep_cents": 1.0,

            # 2a: mean-shift exit (broken thesis)
            "mean_shift_exit_cents": 3.0,

            # 2b: multi-timeframe trend filter
            "long_lookback": 250,
            "trend_tolerance_cents": 4.0,
        }
        super().__init__("mean_reversion", max_capital, params)

        self.prices: deque = deque(maxlen=params["lookback"])
        self.prices_long: deque = deque(maxlen=params["long_lookback"])  # 2b

        # 2a: track rolling mean at time of each position’s entry
        self._entry_means: Dict[str, float] = {}  # position_id -> mean at entry

        # 2d: shared exposure tracker
        self.exposure_tracker = exposure_tracker

        if preferred_side is not None:
            preferred_side = preferred_side.lower().strip()
            if preferred_side not in ("yes", "no"):
                preferred_side = None
        self.preferred_side = preferred_side

    def _calc_stats(self) -> Tuple[float, float]:
        if len(self.prices) < 5:
            return 50.0, 10.0
        prices = list(self.prices)
        mean = sum(prices) / len(prices)
        var = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = math.sqrt(var) if var > 0 else 5.0
        return mean, std

    def _calc_long_stats(self) -> Tuple[float, float]:
        """2b: Stats over the longer lookback window."""
        if len(self.prices_long) < 20:
            return 50.0, 10.0
        prices = list(self.prices_long)
        mean = sum(prices) / len(prices)
        var = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = math.sqrt(var) if var > 0 else 5.0
        return mean, std

    def _check_warmup(self) -> Tuple[bool, str]:
        n = len(self.prices)
        if n < self.params["warmup_samples"]:
            return False, f"warmup:{n}/{self.params['warmup_samples']}"

        recent = list(self.prices)
        price_range = max(recent) - min(recent)
        if price_range < self.params["warmup_min_range"]:
            return False, f"warmup_range:{price_range:.1f}c<{self.params['warmup_min_range']}c"

        return True, "ok"

    def _is_dead_market(self) -> bool:
        lookback = self.params["dead_lookback"]
        if len(self.prices) < lookback:
            return False
        recent = list(self.prices)[-lookback:]
        return (max(recent) - min(recent)) < self.params["dead_min_move"]

    def _collateral_per_contract(self, side: str, price_cents: int) -> float:
        """
        Approx collateral in dollars per contract:
        Kalshi cash required to BUY:
          YES buy at p: you pay p cents
          NO  buy at p: you pay p cents
        """
        return price_cents / 100.0

    def _size_qty(self, side: str, price_cents: int, sigma_mult: float = 1.5) -> int:
        """
        σ-scaled tranche sizing:
          - Compute remaining capital in dollars
          - Pick a fraction of that capital based on how strong the signal is (σ)
          - Weaker signal (1.5σ) → 35% of remaining. Stronger (3σ+) → 80%.
          - This naturally scales DOWN as capital depletes across entries,
            but still pushes high counts early.
          - Minimum floor of size_min_qty so every trade is meaningful.
        """
        remaining = max(0.0, float(self.max_capital) - float(self.capital_used))
        per = self._collateral_per_contract(side, price_cents)
        if per <= 0:
            return 0

        # How many contracts could we possibly buy with ALL remaining capital?
        max_possible = int(remaining / per)

        # σ-scaled fraction: linear interpolation between base_frac and max_frac
        sigma_floor = float(self.params.get("size_sigma_floor", 1.5))
        sigma_cap = float(self.params.get("size_sigma_cap", 3.0))
        base_frac = float(self.params.get("size_base_frac", 0.35))
        max_frac = float(self.params.get("size_max_frac", 0.80))

        # Clamp sigma_mult into [floor, cap] then interpolate
        t = max(0.0, min(1.0, (sigma_mult - sigma_floor) / max(0.01, sigma_cap - sigma_floor)))
        frac = base_frac + t * (max_frac - base_frac)

        qty = int(max_possible * frac)

        # Apply floor and cap
        min_qty = int(self.params.get("size_min_qty", 3))
        max_qty = int(self.params.get("max_order_qty", 200))

        qty = max(min(min_qty, max_possible), qty)  # floor, but don't exceed what's affordable
        qty = min(qty, max_qty)
        qty = min(qty, max_possible)  # final safety: never exceed capital

        return max(0, qty)

    def _in_directional_window(self, secs_to_close: int) -> bool:
        directional_secs = int(self.params.get("directional_close_secs", 300))
        return (
            self.preferred_side in ("yes", "no")
            and secs_to_close <= directional_secs
        )

    def feed_price(self, mid: float):
        """Append mid to rolling windows (called every tick, regardless of MQ state)."""
        self.prices.append(mid)
        self.prices_long.append(mid)

    def evaluate_entry(self, prices, secs_to_close, context) -> Tuple[bool, str, int, int, str]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")

        if yes_bid is None or yes_ask is None:
            return False, "", 0, 0, "no_prices"

        mid = (yes_bid + yes_ask) / 2
        # prices are now fed externally via feed_price()

        warmup_ok, warmup_reason = self._check_warmup()
        if not warmup_ok:
            return False, "", 0, 0, warmup_reason

        if self._is_dead_market():
            return False, "", 0, 0, "dead_market"

        mean, std = self._calc_stats()
        long_mean, _ = self._calc_long_stats()  # 2b

        std_mult = self.params["low_vol_std_mult"] if std < self.params["low_vol_cutoff"] else self.params["high_vol_std_mult"]
        threshold = std_mult * std

        in_dir = self._in_directional_window(int(secs_to_close))
        trend_tol = float(self.params.get("trend_tolerance_cents", 4.0))

        # Price dropped → buy YES
        if mid < mean - threshold:
            # 2b: reject if short mean is above long mean by >trend_tol (fighting downtrend)
            if len(self.prices_long) >= 20 and (mean - long_mean) > trend_tol:
                return False, "", 0, 0, f"trend_block_yes:short={mean:.1f}>long={long_mean:.1f}+{trend_tol}"

            if in_dir and self.preferred_side != "yes":
                return False, "", 0, 0, f"dir_block:want_yes_pref_{self.preferred_side}"
            cooled, _ = self._side_on_cooldown("yes")
            if cooled:
                return False, "", 0, 0, "cooldown_yes"
            edge = mean - mid
            qty = self._size_qty("yes", int(yes_ask), sigma_mult=edge / std)
            if qty <= 0:
                return False, "", 0, 0, "no_capital"

            # 2d: check shared exposure
            if self.exposure_tracker:
                cost = int(yes_ask) * qty / 100.0
                if not self.exposure_tracker.try_reserve(cost):
                    return False, "", 0, 0, "exposure_limit"

            return True, "yes", int(yes_ask), qty, f"below:{edge:.0f}c({edge/std:.1f}σ),mult={std_mult},qty={qty}"

        # Price spiked → buy NO
        if mid > mean + threshold and no_ask is not None:
            # 2b: reject if short mean is below long mean by >trend_tol (fighting uptrend)
            if len(self.prices_long) >= 20 and (long_mean - mean) > trend_tol:
                return False, "", 0, 0, f"trend_block_no:short={mean:.1f}<long={long_mean:.1f}-{trend_tol}"

            if in_dir and self.preferred_side != "no":
                return False, "", 0, 0, f"dir_block:want_no_pref_{self.preferred_side}"
            cooled, _ = self._side_on_cooldown("no")
            if cooled:
                return False, "", 0, 0, "cooldown_no"
            edge = mid - mean
            qty = self._size_qty("no", int(no_ask), sigma_mult=edge / std)
            if qty <= 0:
                return False, "", 0, 0, "no_capital"

            # 2d: check shared exposure
            if self.exposure_tracker:
                cost = int(no_ask) * qty / 100.0
                if not self.exposure_tracker.try_reserve(cost):
                    return False, "", 0, 0, "exposure_limit"

            return True, "no", int(no_ask), qty, f"above:{edge:.0f}c({edge/std:.1f}σ),mult={std_mult},qty={qty}"

        return False, "", 0, 0, "in_range"

    def record_entry(self, side: str, price: float, qty: int, reason: str) -> Position:
        """Override to track entry mean for 2a mean-shift exit."""
        pos = super().record_entry(side, price, qty, reason)
        mean, _ = self._calc_stats()
        self._entry_means[pos.id] = mean
        return pos

    def _get_giveback_frac(self, peak: float) -> float:
        """2c: Tiered trailing stop — tighter giveback at higher peaks."""
        high_thresh = float(self.params.get("profit_defense_high_threshold", 15.0))
        if peak >= high_thresh:
            return float(self.params.get("profit_defense_giveback_frac_high", 0.30))
        return float(self.params.get("profit_defense_giveback_frac", 0.50))

    def _check_mean_shift_exit(self, position: Position) -> Tuple[bool, str]:
        """2a: Exit if the rolling mean has shifted toward entry price (thesis collapsed)."""
        entry_mean = self._entry_means.get(position.id)
        if entry_mean is None:
            return False, ""
        current_mean, _ = self._calc_stats()
        shift_limit = float(self.params.get("mean_shift_exit_cents", 3.0))

        if position.side == "yes":
            # We bought YES because mean was high and price dipped.
            # If mean has dropped >=3c toward our entry, thesis is broken.
            shift = entry_mean - current_mean
            if shift >= shift_limit:
                return True, f"mean_shift:entry_mean={entry_mean:.1f}→{current_mean:.1f} shift={shift:.1f}c"
        else:
            # We bought NO because mean was low and price spiked.
            # If mean has risen >=3c toward our entry, thesis is broken.
            shift = current_mean - entry_mean
            if shift >= shift_limit:
                return True, f"mean_shift:entry_mean={entry_mean:.1f}→{current_mean:.1f} shift={shift:.1f}c"

        return False, ""

    def evaluate_exit(self, position, prices, secs_to_close, context) -> Tuple[bool, str, int, str]:
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        yes_ask = prices.get("imp_yes_ask")

        if yes_bid is None:
            return False, "", 0, "no_bid"

        # Late-game flatten wrong-side inventory
        if (
            self._in_directional_window(int(secs_to_close))
            and bool(self.params.get("force_exit_wrong_side", True))
            and self.preferred_side in ("yes", "no")
            and position.side != self.preferred_side
        ):
            if position.side == "yes" and yes_bid is not None:
                return True, "timeout", max(1, int(yes_bid) - 1), f"dir_flatten:{int(secs_to_close)}s"
            if position.side == "no" and no_bid is not None:
                return True, "timeout", max(1, int(no_bid) - 1), f"dir_flatten:{int(secs_to_close)}s"

        mid = (yes_bid + (yes_ask or yes_bid)) / 2
        mean, std = self._calc_stats()

        revert_sigma_frac = float(self.params.get("revert_sigma_frac", 0.5))
        revert_min_cents = float(self.params.get("revert_min_cents", 2.0))
        revert_dist = max(revert_min_cents, revert_sigma_frac * std)

        if position.side == "yes":
            pnl = float(yes_bid) - float(position.entry_price)

            # Track peak favorable pnl
            if pnl > position.max_fav_pnl_cents:
                position.max_fav_pnl_cents = pnl
                position.max_fav_ts = utc_now().isoformat()

            # 2c: Tiered profit-defense giveback exit
            activate = float(self.params.get("profit_defense_activate_cents", 10.0))
            giveback_frac = self._get_giveback_frac(position.max_fav_pnl_cents)
            min_keep = float(self.params.get("profit_defense_min_keep_cents", 1.0))

            if position.max_fav_pnl_cents >= activate:
                floor = position.max_fav_pnl_cents * (1.0 - giveback_frac)
                if pnl <= max(min_keep, floor):
                    return True, "profit_defense", max(1, int(yes_bid) - 1), \
                        f"profit_defense:pnl={pnl:.0f}c peak={position.max_fav_pnl_cents:.0f}c floor={floor:.0f}c gb={giveback_frac:.0%}"

            # 2a: mean-shift exit (broken thesis)
            ms_exit, ms_reason = self._check_mean_shift_exit(position)
            if ms_exit:
                return True, "mean_shift", max(1, int(yes_bid) - 1), ms_reason

            if mid >= mean - revert_dist:
                return True, "revert", max(1, int(yes_bid) - 1), f"reverted:{pnl:.0f}c(dist={revert_dist:.1f})"

        else:
            if no_bid is not None:
                pnl = float(no_bid) - float(position.entry_price)

                if pnl > position.max_fav_pnl_cents:
                    position.max_fav_pnl_cents = pnl
                    position.max_fav_ts = utc_now().isoformat()

                # 2c: Tiered profit-defense giveback exit
                activate = float(self.params.get("profit_defense_activate_cents", 10.0))
                giveback_frac = self._get_giveback_frac(position.max_fav_pnl_cents)
                min_keep = float(self.params.get("profit_defense_min_keep_cents", 1.0))

                if position.max_fav_pnl_cents >= activate:
                    floor = position.max_fav_pnl_cents * (1.0 - giveback_frac)
                    if pnl <= max(min_keep, floor):
                        return True, "profit_defense", max(1, int(no_bid) - 1), \
                            f"profit_defense:pnl={pnl:.0f}c peak={position.max_fav_pnl_cents:.0f}c floor={floor:.0f}c gb={giveback_frac:.0%}"

                # 2a: mean-shift exit (broken thesis)
                ms_exit, ms_reason = self._check_mean_shift_exit(position)
                if ms_exit:
                    return True, "mean_shift", max(1, int(no_bid) - 1), ms_reason

                if mid <= mean + revert_dist:
                    return True, "revert", max(1, int(no_bid) - 1), f"reverted:{pnl:.0f}c(dist={revert_dist:.1f})"

        return False, "", 0, "hold"

# =============================================================================
# STRATEGY: PAIRED MEAN REVERSION (outcome-independent)
# =============================================================================

class PairedMeanReversionStrategy(BaseStrategy):
    """
    Outcome-independent strategy: buy YES when cheap AND buy NO when cheap.
    If avg_yes_cost + avg_no_cost < 100c, guaranteed profit regardless of winner.

    Uses the same MR signal infrastructure (sigma thresholds, warmup, dead market).
    Paired positions are held to settlement. Unpaired excess uses standard MR exits.
    """

    def __init__(self, max_capital: float, preferred_side: Optional[str] = None,
                 exposure_tracker: Optional["ExposureTracker"] = None):
        params = {
            "lookback": 60,
            "high_vol_std_mult": 1.5,
            "low_vol_std_mult": 2.5,
            "low_vol_cutoff": 5.0,

            "max_positions": 8,  # 4 YES + 4 NO max
            "max_yes_positions": 4,
            "max_no_positions": 4,

            "min_entry_gap_secs": 8,
            "side_cooldown_secs": 20,
            "stop_trading_before_close_secs": 0,

            # warmup / dead-market
            "warmup_samples": 60,
            "warmup_min_range": 5.0,
            "dead_lookback": 30,
            "dead_min_move": 3.0,

            # sizing (same as MR)
            "size_base_frac": 0.35,
            "size_max_frac": 0.80,
            "size_sigma_floor": 1.5,
            "size_sigma_cap": 3.0,
            "size_min_qty": 3,
            "max_order_qty": 200,

            # pairing params
            "pair_target_profit_cents": 8,  # target 8c+ guaranteed profit per pair
            "pair_eagerness_boost": 0.3,     # reduce σ threshold by this for the missing side

            # exit params for unpaired excess
            "reversion_exit_pnl_floor": -3,
            "directional_close_secs": 300,
            "force_exit_wrong_side": True,

            # profit defense for unpaired
            "profit_defense_activate_cents": 10.0,
            "profit_defense_giveback_frac": 0.50,
            "profit_defense_high_threshold": 15.0,
            "profit_defense_giveback_frac_high": 0.30,
            "profit_defense_min_keep_cents": 1.0,

            # long trend filter
            "long_lookback": 250,
            "trend_tolerance_cents": 4.0,
        }
        super().__init__("paired_mr", max_capital, params)

        self.prices: deque = deque(maxlen=params["lookback"])
        self.prices_long: deque = deque(maxlen=params["long_lookback"])

        self.exposure_tracker = exposure_tracker

        if preferred_side is not None:
            preferred_side = preferred_side.lower().strip()
            if preferred_side not in ("yes", "no"):
                preferred_side = None
        self.preferred_side = preferred_side

        # Track cumulative YES/NO fills for pairing math
        self._yes_qty = 0
        self._yes_cost_sum = 0.0  # sum of (entry_price * qty) for YES
        self._no_qty = 0
        self._no_cost_sum = 0.0

    # --- Stats helpers (same as MR) ---

    def _calc_stats(self) -> Tuple[float, float]:
        if len(self.prices) < 5:
            return 50.0, 10.0
        prices = list(self.prices)
        mean = sum(prices) / len(prices)
        var = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = math.sqrt(var) if var > 0 else 5.0
        return mean, std

    def _calc_long_stats(self) -> Tuple[float, float]:
        if len(self.prices_long) < 20:
            return 50.0, 10.0
        prices = list(self.prices_long)
        mean = sum(prices) / len(prices)
        var = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = math.sqrt(var) if var > 0 else 5.0
        return mean, std

    def _check_warmup(self) -> Tuple[bool, str]:
        n = len(self.prices)
        if n < self.params["warmup_samples"]:
            return False, f"warmup:{n}/{self.params['warmup_samples']}"
        recent = list(self.prices)
        price_range = max(recent) - min(recent)
        if price_range < self.params["warmup_min_range"]:
            return False, f"warmup_range:{price_range:.1f}c<{self.params['warmup_min_range']}c"
        return True, "ok"

    def _is_dead_market(self) -> bool:
        lookback = self.params["dead_lookback"]
        if len(self.prices) < lookback:
            return False
        recent = list(self.prices)[-lookback:]
        return (max(recent) - min(recent)) < self.params["dead_min_move"]

    def _collateral_per_contract(self, side: str, price_cents: int) -> float:
        return price_cents / 100.0

    def _size_qty(self, side: str, price_cents: int, sigma_mult: float = 1.5) -> int:
        remaining = max(0.0, float(self.max_capital) - float(self.capital_used))
        per = self._collateral_per_contract(side, price_cents)
        if per <= 0:
            return 0
        max_possible = int(remaining / per)

        sigma_floor = float(self.params.get("size_sigma_floor", 1.5))
        sigma_cap = float(self.params.get("size_sigma_cap", 3.0))
        base_frac = float(self.params.get("size_base_frac", 0.35))
        max_frac = float(self.params.get("size_max_frac", 0.80))

        t = max(0.0, min(1.0, (sigma_mult - sigma_floor) / max(0.01, sigma_cap - sigma_floor)))
        frac = base_frac + t * (max_frac - base_frac)

        qty = int(max_possible * frac)
        min_qty = int(self.params.get("size_min_qty", 3))
        max_qty = int(self.params.get("max_order_qty", 200))

        qty = max(min(min_qty, max_possible), qty)
        qty = min(qty, max_qty)
        qty = min(qty, max_possible)
        return max(0, qty)

    # --- Pairing math ---

    @property
    def _avg_yes_cost(self) -> float:
        return (self._yes_cost_sum / self._yes_qty) if self._yes_qty > 0 else 0.0

    @property
    def _avg_no_cost(self) -> float:
        return (self._no_cost_sum / self._no_qty) if self._no_qty > 0 else 0.0

    @property
    def _paired_qty(self) -> int:
        return min(self._yes_qty, self._no_qty)

    @property
    def _guaranteed_profit_per_pair(self) -> float:
        if self._paired_qty == 0:
            return 0.0
        return 100.0 - self._avg_yes_cost - self._avg_no_cost

    def _count_side_positions(self, side: str) -> int:
        return sum(1 for p in self.positions if p.side == side)

    def _side_qty(self, side: str) -> int:
        return sum(p.qty for p in self.positions if p.side == side)

    def record_entry(self, side: str, price: float, qty: int, reason: str) -> Position:
        pos = super().record_entry(side, price, qty, reason)
        if side == "yes":
            self._yes_qty += qty
            self._yes_cost_sum += price * qty
        else:
            self._no_qty += qty
            self._no_cost_sum += price * qty
        return pos

    # --- Entry ---

    def _in_directional_window(self, secs_to_close: int) -> bool:
        directional_secs = int(self.params.get("directional_close_secs", 300))
        return (
            self.preferred_side in ("yes", "no")
            and secs_to_close <= directional_secs
        )

    def feed_price(self, mid: float):
        """Append mid to rolling windows (called every tick, regardless of MQ state)."""
        self.prices.append(mid)
        self.prices_long.append(mid)

    def evaluate_entry(self, prices, secs_to_close, context) -> Tuple[bool, str, int, int, str]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")

        if yes_bid is None or yes_ask is None:
            return False, "", 0, 0, "no_prices"

        mid = (yes_bid + yes_ask) / 2
        # prices are now fed externally via feed_price()

        warmup_ok, warmup_reason = self._check_warmup()
        if not warmup_ok:
            return False, "", 0, 0, warmup_reason

        if self._is_dead_market():
            return False, "", 0, 0, "dead_market"

        mean, std = self._calc_stats()
        long_mean, _ = self._calc_long_stats()

        std_mult = self.params["low_vol_std_mult"] if std < self.params["low_vol_cutoff"] else self.params["high_vol_std_mult"]
        threshold = std_mult * std

        # Eagerness boost: reduce threshold for the missing side
        boost = float(self.params.get("pair_eagerness_boost", 0.3))
        yes_has = self._yes_qty > 0
        no_has = self._no_qty > 0

        yes_threshold = threshold - (boost * std if no_has and not yes_has else 0)
        no_threshold = threshold - (boost * std if yes_has and not no_has else 0)

        trend_tol = float(self.params.get("trend_tolerance_cents", 4.0))
        target_profit = float(self.params.get("pair_target_profit_cents", 8))
        max_yes_pos = int(self.params.get("max_yes_positions", 4))
        max_no_pos = int(self.params.get("max_no_positions", 4))

        in_dir = self._in_directional_window(int(secs_to_close))

        # Price dropped → buy YES (cheap YES)
        if mid < mean - yes_threshold and self._count_side_positions("yes") < max_yes_pos:
            # Check pairing feasibility: would buying YES at ask still allow profit?
            hypothetical_yes_cost = float(yes_ask)
            if no_has:
                combined = hypothetical_yes_cost + self._avg_no_cost
                if combined > (100 - target_profit):
                    return False, "", 0, 0, f"pair_too_expensive:yes={hypothetical_yes_cost:.0f}+no_avg={self._avg_no_cost:.0f}={combined:.0f}c"

            # Trend filter
            if len(self.prices_long) >= 20 and (mean - long_mean) > trend_tol:
                return False, "", 0, 0, f"trend_block_yes:short={mean:.1f}>long={long_mean:.1f}+{trend_tol}"

            if in_dir and self.preferred_side != "yes":
                return False, "", 0, 0, f"dir_block:want_yes_pref_{self.preferred_side}"

            cooled, _ = self._side_on_cooldown("yes")
            if cooled:
                return False, "", 0, 0, "cooldown_yes"

            edge = mean - mid
            qty = self._size_qty("yes", int(yes_ask), sigma_mult=edge / std)
            if qty <= 0:
                return False, "", 0, 0, "no_capital"

            if self.exposure_tracker:
                cost = int(yes_ask) * qty / 100.0
                if not self.exposure_tracker.try_reserve(cost):
                    return False, "", 0, 0, "exposure_limit"

            paired_info = f"Y:{self._yes_qty}+{qty}/N:{self._no_qty}"
            return True, "yes", int(yes_ask), qty, \
                f"pair_yes:{edge:.0f}c({edge/std:.1f}σ) {paired_info} guar={self._guaranteed_profit_per_pair:.1f}c"

        # Price spiked → buy NO (cheap NO)
        if mid > mean + no_threshold and no_ask is not None and self._count_side_positions("no") < max_no_pos:
            hypothetical_no_cost = float(no_ask)
            if yes_has:
                combined = self._avg_yes_cost + hypothetical_no_cost
                if combined > (100 - target_profit):
                    return False, "", 0, 0, f"pair_too_expensive:yes_avg={self._avg_yes_cost:.0f}+no={hypothetical_no_cost:.0f}={combined:.0f}c"

            if len(self.prices_long) >= 20 and (long_mean - mean) > trend_tol:
                return False, "", 0, 0, f"trend_block_no:short={mean:.1f}<long={long_mean:.1f}-{trend_tol}"

            if in_dir and self.preferred_side != "no":
                return False, "", 0, 0, f"dir_block:want_no_pref_{self.preferred_side}"

            cooled, _ = self._side_on_cooldown("no")
            if cooled:
                return False, "", 0, 0, "cooldown_no"

            edge = mid - mean
            qty = self._size_qty("no", int(no_ask), sigma_mult=edge / std)
            if qty <= 0:
                return False, "", 0, 0, "no_capital"

            if self.exposure_tracker:
                cost = int(no_ask) * qty / 100.0
                if not self.exposure_tracker.try_reserve(cost):
                    return False, "", 0, 0, "exposure_limit"

            paired_info = f"Y:{self._yes_qty}/N:{self._no_qty}+{qty}"
            return True, "no", int(no_ask), qty, \
                f"pair_no:{edge:.0f}c({edge/std:.1f}σ) {paired_info} guar={self._guaranteed_profit_per_pair:.1f}c"

        return False, "", 0, 0, "in_range"

    # --- Exit ---

    def evaluate_exit(self, position, prices, secs_to_close, context) -> Tuple[bool, str, int, str]:
        """
        Paired positions: hold to settlement (guaranteed profit).
        Unpaired excess: use standard MR exit logic.
        """
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        yes_ask = prices.get("imp_yes_ask")

        if yes_bid is None:
            return False, "", 0, "no_bid"

        paired = self._paired_qty

        # Determine if this position is part of a paired set
        # Paired positions = min(yes_qty, no_qty) on each side → hold to settlement
        yes_positions = sorted([p for p in self.positions if p.side == "yes"], key=lambda p: p.entry_time)
        no_positions = sorted([p for p in self.positions if p.side == "no"], key=lambda p: p.entry_time)

        # Count how many contracts are paired on each side
        paired_yes_remaining = paired
        is_paired = False
        if position.side == "yes":
            for p in yes_positions:
                if paired_yes_remaining <= 0:
                    break
                if p.id == position.id:
                    is_paired = True
                    break
                paired_yes_remaining -= p.qty
        else:
            paired_no_remaining = paired
            for p in no_positions:
                if paired_no_remaining <= 0:
                    break
                if p.id == position.id:
                    is_paired = True
                    break
                paired_no_remaining -= p.qty

        # Paired → hold to settlement
        if is_paired:
            return False, "", 0, f"paired:hold_to_settle guar={self._guaranteed_profit_per_pair:.1f}c"

        # Unpaired excess → standard MR exit logic
        # Late-game flatten
        if (
            self._in_directional_window(int(secs_to_close))
            and bool(self.params.get("force_exit_wrong_side", True))
            and self.preferred_side in ("yes", "no")
            and position.side != self.preferred_side
        ):
            if position.side == "yes" and yes_bid is not None:
                return True, "timeout", max(1, int(yes_bid) - 1), f"unpaired_dir_flatten:{int(secs_to_close)}s"
            if position.side == "no" and no_bid is not None:
                return True, "timeout", max(1, int(no_bid) - 1), f"unpaired_dir_flatten:{int(secs_to_close)}s"

        mid = (yes_bid + (yes_ask or yes_bid)) / 2
        mean, std = self._calc_stats()
        pnl_floor = float(self.params.get("reversion_exit_pnl_floor", -3))

        if position.side == "yes":
            pnl = float(yes_bid) - float(position.entry_price)

            if pnl > position.max_fav_pnl_cents:
                position.max_fav_pnl_cents = pnl
                position.max_fav_ts = utc_now().isoformat()

            activate = float(self.params.get("profit_defense_activate_cents", 10.0))
            high_thresh = float(self.params.get("profit_defense_high_threshold", 15.0))
            giveback_frac = float(self.params.get("profit_defense_giveback_frac_high", 0.30)) \
                if position.max_fav_pnl_cents >= high_thresh \
                else float(self.params.get("profit_defense_giveback_frac", 0.50))
            min_keep = float(self.params.get("profit_defense_min_keep_cents", 1.0))

            if position.max_fav_pnl_cents >= activate:
                floor = position.max_fav_pnl_cents * (1.0 - giveback_frac)
                if pnl <= max(min_keep, floor):
                    return True, "take_profit", max(1, int(yes_bid) - 1), \
                        f"unpaired_defense:pnl={pnl:.0f}c peak={position.max_fav_pnl_cents:.0f}c"

            if pnl < pnl_floor:
                return False, "", 0, f"hold_pnl_floor:{pnl:.0f}c"

            if mid >= mean - 2:
                return True, "take_profit", max(1, int(yes_bid) - 1), f"unpaired_reverted:{pnl:.0f}c"

        else:
            if no_bid is not None:
                pnl = float(no_bid) - float(position.entry_price)

                if pnl > position.max_fav_pnl_cents:
                    position.max_fav_pnl_cents = pnl
                    position.max_fav_ts = utc_now().isoformat()

                activate = float(self.params.get("profit_defense_activate_cents", 10.0))
                high_thresh = float(self.params.get("profit_defense_high_threshold", 15.0))
                giveback_frac = float(self.params.get("profit_defense_giveback_frac_high", 0.30)) \
                    if position.max_fav_pnl_cents >= high_thresh \
                    else float(self.params.get("profit_defense_giveback_frac", 0.50))
                min_keep = float(self.params.get("profit_defense_min_keep_cents", 1.0))

                if position.max_fav_pnl_cents >= activate:
                    floor = position.max_fav_pnl_cents * (1.0 - giveback_frac)
                    if pnl <= max(min_keep, floor):
                        return True, "take_profit", max(1, int(no_bid) - 1), \
                            f"unpaired_defense:pnl={pnl:.0f}c peak={position.max_fav_pnl_cents:.0f}c"

                if pnl < pnl_floor:
                    return False, "", 0, f"hold_pnl_floor:{pnl:.0f}c"

                if mid <= mean + 2:
                    return True, "take_profit", max(1, int(no_bid) - 1), f"unpaired_reverted:{pnl:.0f}c"

        return False, "", 0, "unpaired_hold"


# =============================================================================
# STRATEGY: FINAL MINUTES (near-guaranteed late-game trades)
# =============================================================================

class FinalMinutesStrategy(BaseStrategy):
    """
    Only trades in the last 5 minutes when ESPN win probability is decisive (>90% or <10%).
    Buys the winning side below fair value, exploiting thin books / slow price updates.
    Holds to settlement (no exit unless WP dramatically reverses).
    """

    bypass_market_quality = True  # late-game markets often look "dead" at 95/5

    def __init__(self, max_capital: float):
        params = {
            "max_positions": 4,
            "min_entry_gap_secs": 15,
            "side_cooldown_secs": 0,
            "stop_trading_before_close_secs": 0,

            # activation
            "active_window_secs": 300,     # last 5 minutes
            "min_wp_threshold": 0.90,      # ESPN WP must be > this (or < 1-this)
            "discount_cents": 10,          # buy at fair - this many cents

            # exit
            "wp_reversal_threshold": 0.70, # exit if WP reverses below this

            # sizing
            "entry_frac": 0.80,           # 80% of remaining allocation per entry
            "min_qty": 5,
        }
        super().__init__("final_minutes", max_capital, params)

    def evaluate_entry(self, prices, secs_to_close, context) -> Tuple[bool, str, int, int, str]:
        # Only active in last N minutes
        window = int(self.params.get("active_window_secs", 300))
        if secs_to_close > window:
            return False, "", 0, 0, f"too_early:{int(secs_to_close)}s>{window}s"

        espn_wp = context.get("espn_live_win_pct")
        if espn_wp is None:
            return False, "", 0, 0, "no_espn_wp"

        espn_wp = float(espn_wp)
        if espn_wp > 1.0:
            espn_wp /= 100.0

        wp_thresh = float(self.params.get("min_wp_threshold", 0.90))
        discount = float(self.params.get("discount_cents", 10))

        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")

        # High WP → buy YES below fair
        if espn_wp >= wp_thresh and yes_ask is not None:
            fair_yes = espn_wp * 100.0
            if float(yes_ask) <= fair_yes - discount:
                qty = self._calc_qty(int(yes_ask))
                if qty <= 0:
                    return False, "", 0, 0, "no_capital"
                return True, "yes", int(yes_ask), qty, \
                    f"final_yes:wp={espn_wp:.2f} fair={fair_yes:.0f} ask={yes_ask} disc={fair_yes - float(yes_ask):.0f}c"

        # Low WP → buy NO below fair
        if espn_wp <= (1.0 - wp_thresh) and no_ask is not None:
            fair_no = (1.0 - espn_wp) * 100.0
            if float(no_ask) <= fair_no - discount:
                qty = self._calc_qty(int(no_ask))
                if qty <= 0:
                    return False, "", 0, 0, "no_capital"
                return True, "no", int(no_ask), qty, \
                    f"final_no:wp={espn_wp:.2f} fair_no={fair_no:.0f} ask={no_ask} disc={fair_no - float(no_ask):.0f}c"

        return False, "", 0, 0, f"no_signal:wp={espn_wp:.2f}"

    def _calc_qty(self, price_cents: int) -> int:
        remaining = max(0.0, float(self.max_capital) - float(self.capital_used))
        per = price_cents / 100.0
        if per <= 0:
            return 0
        max_possible = int(remaining / per)
        frac = float(self.params.get("entry_frac", 0.80))
        qty = max(int(max_possible * frac), int(self.params.get("min_qty", 5)))
        qty = min(qty, max_possible)
        return max(0, qty)

    def evaluate_exit(self, position, prices, secs_to_close, context) -> Tuple[bool, str, int, str]:
        """Hold to settlement unless WP dramatically reverses."""
        espn_wp = context.get("espn_live_win_pct")
        if espn_wp is None:
            return False, "", 0, "hold_no_wp"

        espn_wp = float(espn_wp)
        if espn_wp > 1.0:
            espn_wp /= 100.0

        reversal = float(self.params.get("wp_reversal_threshold", 0.70))

        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")

        # If we hold YES but WP dropped below reversal threshold
        if position.side == "yes" and espn_wp < reversal and yes_bid is not None:
            return True, "stop", max(1, int(yes_bid) - 1), \
                f"wp_reversal:wp={espn_wp:.2f}<{reversal}"

        # If we hold NO but WP rose above (1 - reversal) threshold
        if position.side == "no" and espn_wp > (1.0 - reversal) and no_bid is not None:
            return True, "stop", max(1, int(no_bid) - 1), \
                f"wp_reversal:wp={espn_wp:.2f}>{1.0 - reversal:.2f}"

        return False, "", 0, "hold_to_settle"


# =============================================================================
# GAME RUNNER (P3 market quality + P4 settlement-aware reporting)
# =============================================================================

POLL_INTERVAL_SECS = 3.0
STOP_TRADING_BEFORE_CLOSE_SECS = 300


class GameRunner:

    def __init__(
        self,
        game_label: str,
        ticker: str,
        market: Dict[str, Any],
        strategies: List[BaseStrategy],
        private_key,
        espn_clock=None,
        log_dir: Optional[str] = None,
        maker_entries: bool = False,
        maker_exits: bool = False,
        min_entry_price: int = 0,
        mq_params: Optional[Dict[str, float]] = None,
        base_strategy_overrides: Optional[Dict[str, Dict]] = None,
    ):
        self.label = game_label
        self.ticker = ticker
        self.market = market
        self.strategies = strategies
        self.private_key = private_key
        self.close_time = parse_iso(market["close_time"])
        self.espn_clock = espn_clock
        self.maker_entries = maker_entries
        self.maker_exits = maker_exits
        self.min_entry_price = min_entry_price
        self._base_strategy_overrides = base_strategy_overrides

        # P3: Market quality monitor
        mq_kw = mq_params or {}
        self.quality = MarketQualityMonitor(warmup_samples=40, warmup_secs=180, **mq_kw)
        self._market_killed = False
        self._market_kill_reason = ""

        # Logging
        if log_dir is None:
            raise ValueError("GameRunner requires log_dir")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_path = self.log_dir / "snapshots.csv"
        self.trade_path = self.log_dir / "trades.csv"
        self.position_path = self.log_dir / "positions.csv"
        self.event_path = self.log_dir / "events.csv"
        self.orderbook_path = self.log_dir / "orderbook.jsonl"

        self._init_logs()
        self.snapshots = 0

        # Live config reload
        self._config_path = Path("strategy_config.json")
        self._last_config_mtime = 0.0
        self._config_check_interval = 10  # check every 10 loop iterations (~30s)

    def _init_logs(self):
        is_restart = False
        for path, headers in [
            (self.snapshot_path, [
                "timestamp", "ticker", "secs_to_close", "clock_source",
                "yes_bid", "yes_ask", "no_bid", "no_ask", "mid", "spread",
                "mq_std", "mq_tradeable", "espn_live_win_pct", "game_progress", "secs_to_tip",
            ]),
            (self.trade_path, [
                "timestamp", "ticker", "strategy", "action", "side",
                "intended_price", "fill_price", "qty", "fee_cents",
                "reason", "order_id",
            ]),
            (self.position_path, [
                "position_id", "strategy", "side",
                "entry_price", "entry_time", "entry_fee",
                "exit_price", "exit_time", "exit_type", "exit_fee",
                "lock_price", "gross_pnl", "net_pnl", "hold_secs",
            ]),
            (self.event_path, [
                "timestamp", "strategy", "event", "detail",
            ]),
        ]:
            if path.exists() and path.stat().st_size > 0:
                is_restart = True
            else:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(headers)

        if is_restart:
            self._log_event("system", "runner_restart", f"Appending to existing logs in {self.log_dir}")

    def _log_orderbook(self, ob: Dict[str, Any], secs_to_close: float, clock_source: str, depth: int = 25):
        """
        Store top-of-book + depth for later replay/backtests.
        JSONL keeps it stream-friendly for R2 + dashboards.
        """
        try:
            book = (ob.get("orderbook") or {})
            yes_levels = (book.get("yes") or [])[:depth]
            no_levels  = (book.get("no") or [])[:depth]

            rec = {
                "ts_utc": utc_now().isoformat(),
                "ticker": self.ticker,
                "secs_to_close": int(secs_to_close),
                "clock_source": clock_source,
                "depth": depth,
                "yes": yes_levels,  # [[price, qty], ...] already in your format
                "no": no_levels,
            }
            with open(self.orderbook_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass


    def _log_snapshot(self, prices, secs_to_close, clock_source, mq_std, mq_ok, espn_wp, game_progress, secs_to_tip):

        yes_bid = prices.get("best_yes_bid", "")
        yes_ask = prices.get("imp_yes_ask", "")
        no_bid = prices.get("best_no_bid", "")
        no_ask = prices.get("imp_no_ask", "")
        mid = ""
        spread = ""
        if yes_bid and yes_ask:
            mid = f"{(yes_bid + yes_ask) / 2:.1f}"
            spread = yes_ask - yes_bid
        with open(self.snapshot_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                utc_now().isoformat(), self.ticker, int(secs_to_close), clock_source,
                yes_bid, yes_ask, no_bid, no_ask, mid, spread,
                f"{mq_std:.1f}", mq_ok,
                espn_wp if espn_wp is not None else "",
                game_progress if game_progress is not None else "",
                secs_to_tip if secs_to_tip is not None else "",
            ])


    def _log_trade(self, strategy, action, side, intended, fill, qty, fee, reason, oid=""):
        with open(self.trade_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                utc_now().isoformat(), self.ticker, strategy, action, side,
                intended, f"{fill:.1f}" if fill else "", qty, f"{fee:.2f}",
                reason, oid,
            ])

    def _log_position(self, closed: ClosedPosition):
        with open(self.position_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                closed.id, closed.strategy, closed.side,
                closed.entry_price, closed.entry_time, f"{closed.entry_fee:.2f}",
                closed.exit_price, closed.exit_time, closed.exit_type, f"{closed.exit_fee:.2f}",
                closed.lock_price or "", f"{closed.gross_pnl:.1f}", f"{closed.net_pnl:.1f}",
                closed.hold_secs,
            ])

    def _log_event(self, strategy, event, detail=""):
        with open(self.event_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([utc_now().isoformat(), strategy, event, detail])

    # --- Execution helpers ---

    def _execute_entry(self, strategy: BaseStrategy, side: str, price: int, qty: int, reason: str,
                    context: Dict[str, Any]):

        mode = "maker" if self.maker_entries else "taker"
        print_status(f"[{self.label}][{strategy.name}] ENTRY [{mode}] {side.upper()} {qty}x @ {price}c — {reason}")
        self._log_event(strategy.name, f"entry_attempt_{mode}", f"{side} {qty}x@{price}c {reason}")

        order_id = None
        try:
            ob = fetch_orderbook(self.ticker)
            px = derive_prices(ob)

            if self.maker_entries:
                # Maker mode: place order 1c below implied ask so it rests on the book
                maker_price = price - 1
                fresh_ask = px.get("imp_yes_ask") if side == "yes" else px.get("imp_no_ask")
                if fresh_ask is None or maker_price >= int(fresh_ask):
                    print_status(f"[{self.label}][{strategy.name}] SKIP maker — spread=0 or no ask (price={maker_price}c ask={fresh_ask})")
                    self._log_event(strategy.name, "entry_skip_maker_spread0", f"{side}@{maker_price}c ask={fresh_ask}")
                    return None
                price = maker_price
            else:
                # Taker mode: liquidity check before sending order
                if not has_fill_liquidity_for_implied_buy(px, side, price, min_qty=max(MIN_LIQUIDITY_CONTRACTS, qty)):
                    print_status(f"[{self.label}][{strategy.name}] SKIP — no liquidity for {side} @{price}c")
                    self._log_event(strategy.name, "entry_skip_liquidity", f"{side}@{price}c")
                    return None

            order_id = place_limit_buy(self.private_key, self.ticker, side, price, qty)
            wait_secs = 45 if self.maker_entries else int(strategy.entry_wait_secs(context) or 15)
            filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, side, max_wait_secs=wait_secs)

            if filled > 0:
                actual = vwap if vwap else float(price)
                pos = strategy.record_entry(side, actual, filled, reason)

                if self.maker_entries:
                    # Maker orders have 0 fee — correct the taker fee recorded by record_entry
                    strategy.total_fees -= pos.entry_fee
                    pos.entry_fee = 0.0
                    fee = 0.0
                else:
                    fee = calc_taker_fee(int(actual), filled)

                print_status(f"[{self.label}][{strategy.name}] FILLED [{mode}] {filled}x @ {actual:.1f}c (fee={fee:.1f}c)")
                strategy.on_entry_result(True, context)
                self._log_trade(strategy.name, f"entry_fill_{mode}", side, price, actual, filled, fee, reason, order_id)
                return pos
            else:
                print_status(f"[{self.label}][{strategy.name}] NO FILL [{mode}]")
                strategy.on_entry_result(False, context)
                self._log_trade(strategy.name, f"entry_nofill_{mode}", side, price, 0, 0, 0, reason, order_id)
                return None

        except Exception as e:
            oid_info = f" order_id={order_id}" if order_id else ""
            print_status(f"[{self.label}][{strategy.name}] ENTRY ERROR: {e}{oid_info}")
            self._log_event(strategy.name, "entry_error", f"{e}{oid_info}")
            return None

    def _execute_exit(self, strategy: BaseStrategy, position: Position,
                      exit_type: str, price: int, reason: str):
        print_status(f"[{self.label}][{strategy.name}] EXIT {position.side.upper()} @ {price}c — {reason}")
        self._log_event(strategy.name, f"exit_{exit_type}_attempt", f"{position.side}@{price}c {reason}")

        lock_price = None
        use_maker = (self.maker_exits and exit_type not in ("timeout", "lock"))
        mode = "maker" if use_maker else "taker"
        order_id = None

        try:
            if exit_type == "lock":
                lock_side = "no" if position.side == "yes" else "yes"
                order_id = place_limit_buy(self.private_key, self.ticker, lock_side, price, position.qty)
                filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, lock_side, max_wait_secs=15)
                if filled > 0:
                    lock_price = vwap if vwap else float(price)
                    actual_exit = lock_price
                else:
                    print_status(f"[{self.label}][{strategy.name}] LOCK NO FILL")
                    self._log_event(strategy.name, "lock_nofill", f"{lock_side}@{price}c")
                    return None

            elif use_maker:
                # Maker exit: place sell at bid+1 (resting on the book)
                ob = fetch_orderbook(self.ticker)
                px = derive_prices(ob)
                bid_key = "best_yes_bid" if position.side == "yes" else "best_no_bid"
                ask_key = "imp_yes_ask" if position.side == "yes" else "imp_no_ask"
                fresh_bid = px.get(bid_key)
                fresh_ask = px.get(ask_key)

                if fresh_bid is None:
                    print_status(f"[{self.label}][{strategy.name}] MAKER EXIT SKIP — no bid")
                    self._log_event(strategy.name, "exit_skip_maker_nobid", f"{position.side}")
                    # Fall through to taker
                    use_maker = False
                else:
                    maker_price = int(fresh_bid) + 1
                    if fresh_ask is not None and maker_price >= int(fresh_ask):
                        print_status(f"[{self.label}][{strategy.name}] MAKER EXIT SKIP — spread=0 (bid+1={maker_price} >= ask={fresh_ask})")
                        self._log_event(strategy.name, "exit_skip_maker_spread0", f"{position.side} bid+1={maker_price} ask={fresh_ask}")
                        use_maker = False
                    else:
                        order_id = place_limit_sell(self.private_key, self.ticker, position.side, maker_price, position.qty)
                        filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, position.side, max_wait_secs=45)
                        if filled > 0:
                            actual_exit = vwap if vwap else float(maker_price)
                            fee = 0.0
                            print_status(f"[{self.label}][{strategy.name}] FILLED [maker] exit @ {actual_exit:.1f}c (fee=0)")
                        else:
                            # Cancel maker, fall back to taker
                            print_status(f"[{self.label}][{strategy.name}] MAKER EXIT NO FILL — falling back to taker")
                            self._log_event(strategy.name, "exit_maker_nofill_fallback", f"{position.side}@{maker_price}c")
                            try:
                                cancel_order(self.private_key, order_id)
                            except Exception:
                                pass
                            use_maker = False

                # Taker fallback if maker failed
                if not use_maker and exit_type != "lock":
                    mode = "taker"
                    ob = fetch_orderbook(self.ticker)
                    px = derive_prices(ob)
                    bid_key = "best_yes_bid" if position.side == "yes" else "best_no_bid"
                    fresh_bid = px.get(bid_key)
                    if fresh_bid is None:
                        print_status(f"[{self.label}][{strategy.name}] EXIT NO BID — cannot sell")
                        self._log_event(strategy.name, "exit_nofill", f"{position.side} no_bid")
                        return None
                    taker_price = max(1, int(fresh_bid) - 1)
                    order_id = place_limit_sell(self.private_key, self.ticker, position.side, taker_price, position.qty)
                    filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, position.side, max_wait_secs=15)
                    if filled > 0:
                        actual_exit = vwap if vwap else float(taker_price)
                        fee = calc_taker_fee(int(actual_exit), position.qty)
                    else:
                        print_status(f"[{self.label}][{strategy.name}] EXIT NO FILL [taker fallback]")
                        self._log_event(strategy.name, "exit_nofill", f"{position.side}@{taker_price}c")
                        return None

            else:
                # Standard taker exit
                order_id = place_limit_sell(self.private_key, self.ticker, position.side, price, position.qty)
                filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, position.side, max_wait_secs=15)
                if filled > 0:
                    actual_exit = vwap if vwap else float(price)
                    fee = calc_taker_fee(int(actual_exit), position.qty)
                else:
                    print_status(f"[{self.label}][{strategy.name}] EXIT NO FILL")
                    self._log_event(strategy.name, "exit_nofill", f"{position.side}@{price}c")
                    return None

            closed = strategy.record_exit(position, exit_type, actual_exit, lock_price)
            if exit_type == "lock":
                fee = calc_taker_fee(int(actual_exit), position.qty)
            # Maker exits: correct the taker fee recorded by record_exit
            if mode == "maker" and fee == 0.0:
                strategy.total_fees -= closed.exit_fee
                closed.exit_fee = 0.0
                closed.net_pnl = closed.gross_pnl - closed.entry_fee - closed.exit_fee
            print_status(f"[{self.label}][{strategy.name}] CLOSED [{mode}]: net={closed.net_pnl:.1f}c")
            self._log_trade(strategy.name, f"exit_{exit_type}_{mode}", position.side, price,
                           actual_exit, position.qty, fee, reason, order_id)
            self._log_position(closed)
            return closed

        except Exception as e:
            oid_info = f" order_id={order_id}" if order_id else ""
            print_status(f"[{self.label}][{strategy.name}] EXIT ERROR: {e}{oid_info}")
            self._log_event(strategy.name, "exit_error", f"{e}{oid_info}")
            return None

    # --- P4: Settlement-aware P&L ---

    def _settlement_pnl_warning(self, prices: Dict[str, Any], secs_to_close: int):
        """P4: Warn about open positions that will expire badly"""
        if secs_to_close > 300:
            return

        for strategy in self.strategies:
            for pos in strategy.positions:
                # Estimate settlement value
                mid = None
                yes_bid = prices.get("best_yes_bid")
                yes_ask = prices.get("imp_yes_ask")
                if yes_bid and yes_ask:
                    mid = (yes_bid + yes_ask) / 2

                if mid is None:
                    continue

                # If YES side and market says <30¢, this position likely expires worthless
                if pos.side == "yes" and mid < 30:
                    potential_loss = pos.entry_price * pos.qty
                    print_status(
                        f"[{self.label}][{strategy.name}] ⚠ SETTLEMENT RISK: "
                        f"{pos.side.upper()} @{pos.entry_price:.0f}c, market={mid:.0f}c, "
                        f"potential loss={potential_loss:.0f}c if YES loses"
                    )
                    self._log_event(strategy.name, "settlement_warning",
                                   f"{pos.side}@{pos.entry_price:.0f}c mid={mid:.0f}c")

                if pos.side == "no" and mid > 70:
                    potential_loss = pos.entry_price * pos.qty
                    print_status(
                        f"[{self.label}][{strategy.name}] ⚠ SETTLEMENT RISK: "
                        f"{pos.side.upper()} @{pos.entry_price:.0f}c, market={mid:.0f}c, "
                        f"potential loss={potential_loss:.0f}c if NO loses"
                    )
                    self._log_event(strategy.name, "settlement_warning",
                                   f"{pos.side}@{pos.entry_price:.0f}c mid={mid:.0f}c")

    def _calc_unrealized_pnl(self, prices: Dict[str, Any]) -> float:
        """
        Mark-to-market UNREALIZED net PnL (in cents), using best bids as exit.
        Includes estimated taker fee for the exit.
        """
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")

        unreal = 0.0
        for s in self.strategies:
            for pos in s.positions:
                if pos.side == "yes" and yes_bid is not None:
                    gross = (float(yes_bid) - pos.entry_price) * pos.qty
                    fee_est = calc_taker_fee(int(yes_bid), pos.qty)
                    unreal += (gross - fee_est)
                elif pos.side == "no" and no_bid is not None:
                    gross = (float(no_bid) - pos.entry_price) * pos.qty
                    fee_est = calc_taker_fee(int(no_bid), pos.qty)
                    unreal += (gross - fee_est)
        return unreal

    def _calc_realized_pnl(self) -> float:
        """Realized net PnL (in cents) from closed positions."""
        realized = 0.0
        for s in self.strategies:
            realized += sum(c.net_pnl for c in s.closed)
        return realized
    
    def _open_inventory_by_side(self) -> str:
        yes_qty = 0
        no_qty = 0
        for s in self.strategies:
            for pos in s.positions:
                if pos.side == "yes":
                    yes_qty += pos.qty
                elif pos.side == "no":
                    no_qty += pos.qty
        return f"inv YES:{yes_qty} NO:{no_qty}"


    # --- Main loop ---

    def _get_secs_to_close(self) -> Tuple[float, str, Optional[float], Optional[float], Optional[int], bool]:
        """
        Returns:
        (secs_to_close, clock_source, espn_live_win_pct, game_progress, secs_to_tip)
        """
        now = utc_now()
        kalshi_secs = (self.close_time - now).total_seconds()

        espn_wp = None
        game_progress = None
        secs_to_tip = None

        if self.espn_clock:
            try:
                ctx = self.espn_clock.get_live_context()
                state = (ctx.get("state") or "").lower()
                secs_to_end = ctx.get("secs_to_game_end")
                espn_is_live = (state == "in" and secs_to_end is not None)

                espn_wp = ctx.get("espn_live_win_pct")
                game_progress = ctx.get("game_progress")
                secs_to_tip = ctx.get("secs_to_tip")

                game_secs = ctx.get("secs_to_game_end")
                why = ctx.get("why", "espn")

                if game_secs is not None:
                    return min(kalshi_secs, float(game_secs)), f"espn:{why}", espn_wp, game_progress, secs_to_tip, espn_is_live

                # pregame: we still return kalshi_secs (placeholder), but expose secs_to_tip
                return kalshi_secs, f"espn:{why}", espn_wp, game_progress, secs_to_tip, espn_is_live

            except Exception:
                pass

        return kalshi_secs, "kalshi", espn_wp, game_progress, secs_to_tip, False

    def run(self) -> Dict[str, Any]:
        print_status(f"[{self.label}] Starting — Strategies: {[s.name for s in self.strategies]}")
        print_status(f"[{self.label}] Close: {self.close_time}")

        while True:
            secs_to_close, clock_source, espn_wp, game_progress, secs_to_tip, espn_is_live = self._get_secs_to_close()

            if secs_to_close <= 0:
                print_status(f"[{self.label}] Market closed")
                break

            try:
                ob = fetch_orderbook(self.ticker)
                prices = derive_prices(ob)
                self._log_orderbook(ob, secs_to_close, clock_source, depth=int(os.getenv("OB_DEPTH") or "10"))
            except Exception as e:
                print_status(f"[{self.label}] Orderbook error: {e}")
                time.sleep(POLL_INTERVAL_SECS)
                continue

            self.snapshots += 1

            # Live config reload check
            if self.snapshots % self._config_check_interval == 0:
                self._check_config_reload()

            # Periodic position reconciliation (~5 min at 3s polling)
            if self.snapshots % 100 == 0 and self.snapshots > 0:
                self._reconcile_positions()

            # Compute midpoint and spread for quality monitor
            yes_bid = prices.get("best_yes_bid")
            yes_ask = prices.get("imp_yes_ask")
            mid = None
            spread = None
            if yes_bid is not None and yes_ask is not None:
                mid = (yes_bid + yes_ask) / 2
                spread = yes_ask - yes_bid

            # P3: Update quality monitor
            self.quality.update(mid, spread)
            mq_std = self.quality.recent_volatility()

            # P3: Check if market is tradeable
            if not self._market_killed:
                mq_ok, mq_reason = self.quality.is_tradeable()
            else:
                mq_ok = False
                mq_reason = self._market_kill_reason

            self._log_snapshot(prices, secs_to_close, clock_source, mq_std, mq_ok, espn_wp, game_progress, secs_to_tip)


            # Status every 30 ticks
            if self.snapshots % 30 == 0:
                open_count = sum(len(s.positions) for s in self.strategies)
                realized = self._calc_realized_pnl()
                unreal = self._calc_unrealized_pnl(prices)
                total = realized + unreal
                inv = self._open_inventory_by_side()
                pref = ""
                for s in self.strategies:
                    if hasattr(s, "preferred_side") and s.preferred_side:
                        pref = s.preferred_side.upper()
                        break

                print_status(
                    f"[{self.label}] Y:{yes_bid}/{yes_ask} | "
                    f"secs:{int(secs_to_close)} src:{clock_source} | "
                    f"open:{open_count} | mq:{mq_reason} | "
                    f"{inv} pref:{pref} | "
                    f"PnL R:{realized:.1f}c U:{unreal:.1f}c T:{total:.1f}c "
                    f"(${total/100:.2f})"
                )



            # P4: Settlement warnings near close
            self._settlement_pnl_warning(prices, int(secs_to_close))

            context = {
                "espn_is_live": espn_is_live,
                "secs_to_close": secs_to_close,
                "market_quality_ok": mq_ok,
                "espn_live_win_pct": espn_wp,     # 0..1 or None
                "game_progress": game_progress,  # 0..1 or None
                "secs_to_tip": secs_to_tip,      # int seconds or None
            }


            # Feed prices to all strategies regardless of MQ state
            # so warmup progresses even during MQ-blocked periods
            if mid is not None:
                for strategy in self.strategies:
                    if hasattr(strategy, 'feed_price'):
                        strategy.feed_price(mid)

            for strategy in self.strategies:
                # --- Exits first (always allowed, even in dead markets) ---
                for pos in list(strategy.positions):
                    should_exit, exit_type, exit_price, reason = strategy.evaluate_exit(
                        pos, prices, int(secs_to_close), context
                    )
                    if should_exit:
                        self._execute_exit(strategy, pos, exit_type, exit_price, reason)

                # --- Entries (only in tradeable markets, unless strategy bypasses) ---
                if not mq_ok and not getattr(strategy, 'bypass_market_quality', False):
                    continue  # P3: skip entries in dead/illiquid markets

                can_enter, _ = strategy.can_enter(secs_to_close=int(secs_to_close))
                if can_enter:
                    should_enter, side, price, qty, reason = strategy.evaluate_entry(
                        prices, int(secs_to_close), context
                    )
                    if should_enter:
                        if self.min_entry_price > 0 and price < self.min_entry_price:
                            self._log_event(strategy.name, "entry_skip_min_price",
                                           f"{side}@{price}c<min={self.min_entry_price}c")
                        else:
                            self._execute_entry(strategy, side, price, qty, reason, context)


            time.sleep(POLL_INTERVAL_SECS)

        # --- Final summary ---
        print_status(f"[{self.label}] === SUMMARY ===")
        result = {"strategies": {}}

        for strategy in self.strategies:
            stats = strategy.get_stats()
            result["strategies"][strategy.name] = stats
            print_status(
                f"[{self.label}][{strategy.name}] "
                f"Trades:{stats['trades']} "
                f"W:{stats['wins']}/L:{stats['losses']} | "
                f"Locks:{stats['locks']} Stops:{stats['stops']} | "
                f"Net:{stats['net_pnl']:.1f}c (${stats['net_pnl']/100:.2f}) | "
                f"Fees:{stats['fees']:.1f}c"
            )

        # Save summary JSON
        ts = utc_now().strftime("%Y%m%d_%H%M%S")
        summary_path = f"logs/summary_{self.label}_{ts}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print_status(f"[{self.label}] Summary: {summary_path}")

        # Verify fills against Kalshi
        self._verify_fills()

        return result

    def _check_config_reload(self):
        """Hot-reload strategy params from strategy_config.json if modified."""
        try:
            if not self._config_path.exists():
                return
            mtime = self._config_path.stat().st_mtime
            if mtime > self._last_config_mtime:
                is_first_load = (self._last_config_mtime == 0.0)
                new_cfg = json.loads(self._config_path.read_text(encoding="utf-8"))
                for strategy in self.strategies:
                    # Re-apply sport-specific base overrides first
                    if self._base_strategy_overrides:
                        base = self._base_strategy_overrides.get(strategy.name, {})
                        if base:
                            strategy.update_params(base)
                    # Then apply live config on top
                    overrides = new_cfg.get(strategy.name, {})
                    if overrides:
                        strategy.update_params(overrides)
                self._last_config_mtime = mtime
                if not is_first_load:
                    self._log_event("system", "config_reload", f"params updated from {self._config_path}")
                    print_status(f"[{self.label}] Config reloaded from {self._config_path}")
        except Exception as e:
            self._log_event("system", "config_reload_error", str(e))

    def _verify_fills(self):
        """Compare our trades.csv against Kalshi's fill records."""
        try:
            if not self.trade_path.exists():
                return

            order_ids = set()
            with open(self.trade_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    oid = row.get("order_id", "").strip()
                    if oid:
                        order_ids.add(oid)

            if not order_ids:
                self._log_event("system", "verify", "no orders to verify")
                return

            mismatches = 0
            for oid in order_ids:
                try:
                    fills = fetch_fills_for_order(self.private_key, oid)
                    if not fills:
                        self._log_event("verify", "fill_mismatch", f"order_id={oid} no fills from Kalshi")
                        mismatches += 1
                        continue

                    kalshi_qty = sum(int(f.get("count", 0)) for f in fills)
                    # Compare with our logged qty for this order
                    logged_qty = 0
                    with open(self.trade_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get("order_id", "").strip() == oid:
                                try:
                                    logged_qty = int(row.get("qty", 0))
                                except (ValueError, TypeError):
                                    pass
                                break

                    if kalshi_qty != logged_qty:
                        self._log_event("verify", "fill_mismatch",
                                       f"order_id={oid} logged_qty={logged_qty} kalshi_qty={kalshi_qty}")
                        mismatches += 1

                except Exception as e:
                    self._log_event("verify", "fill_check_error", f"order_id={oid} {e}")

            if mismatches == 0:
                self._log_event("system", "verify", f"fills_ok: {len(order_ids)} orders verified")
            else:
                self._log_event("system", "verify", f"{mismatches} mismatches out of {len(order_ids)} orders")

            print_status(f"[{self.label}] Verification: {len(order_ids)} orders, {mismatches} mismatches")

        except Exception as e:
            self._log_event("system", "verify_error", str(e))

    def _reconcile_positions(self):
        """Compare in-memory positions against Kalshi API positions for this ticker."""
        try:
            kalshi_positions = fetch_positions(self.private_key)
            # Filter to this ticker
            kalshi_for_ticker = [p for p in kalshi_positions if p.get("ticker") == self.ticker]

            # Sum Kalshi qty by side
            kalshi_yes = sum(int(p.get("count", 0)) for p in kalshi_for_ticker if p.get("side") == "yes")
            kalshi_no = sum(int(p.get("count", 0)) for p in kalshi_for_ticker if p.get("side") == "no")

            # Sum in-memory strategy positions by side
            bot_yes = 0
            bot_no = 0
            for s in self.strategies:
                for pos in s.positions:
                    if pos.side == "yes":
                        bot_yes += pos.qty
                    elif pos.side == "no":
                        bot_no += pos.qty

            if kalshi_yes == bot_yes and kalshi_no == bot_no:
                self._log_event("system", "reconcile_ok",
                               f"YES bot={bot_yes} kalshi={kalshi_yes} | NO bot={bot_no} kalshi={kalshi_no}")
            else:
                self._log_event("system", "reconcile_MISMATCH",
                               f"YES bot={bot_yes} kalshi={kalshi_yes} | NO bot={bot_no} kalshi={kalshi_no}")
                print_status(
                    f"\033[1;31m[{self.label}] *** POSITION MISMATCH *** "
                    f"YES bot={bot_yes} kalshi={kalshi_yes} | "
                    f"NO bot={bot_no} kalshi={kalshi_no}\033[0m"
                )
        except Exception as e:
            self._log_event("system", "reconcile_error", str(e))