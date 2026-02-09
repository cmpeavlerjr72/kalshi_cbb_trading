# production_strategies.py
# Updated Feb 6, 2026 — fixes from Feb 5 post-mortem
#
# Changes from last night:
#   P1: Model Edge circuit breakers (divergence, final-min lockout, re-entry cooldown, market-decided)
#   P2: Mean Reversion adaptive threshold (dynamic σ, vol warmup, dead market detection)
#   P3: Market quality pre-filter (skip illiquid/dead markets)
#   P4: Settlement-aware P&L reporting
#   P5: Fee management (fee-aware edge, lock attempt cap)

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
    SERIES_TICKER,
    MIN_LIQUIDITY_CONTRACTS,
)

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


def fee_aware_min_edge(price_cents: int, stop_loss_cents: int = 15,
                       lock_prob: float = 0.55) -> float:
    """
    P5: Minimum edge needed to be +EV after round-trip fees.
    Conservative lock_prob=0.55 (last night was ~64% win but still lost money).
    """
    entry_fee = calc_taker_fee(price_cents, 1)
    exit_fee = calc_taker_fee(price_cents, 1)  # approximate
    total_fees = entry_fee + exit_fee

    # EV = P(lock) × (lock_profit - fees) - P(stop) × (stop + fees)
    # For breakeven: lock_profit_needed = P(stop)/P(lock) × (stop + fees) + fees
    p_stop = 1.0 - lock_prob
    needed = (p_stop / lock_prob) * (stop_loss_cents + total_fees) + total_fees
    return needed


# =============================================================================
# MARKET QUALITY FILTER (P3)
# =============================================================================

class MarketQualityMonitor:
    """
    P3: Tracks market quality metrics over a warmup window.
    Allows strategies to skip dead/illiquid markets.
    """

    def __init__(self, warmup_samples: int = 40, warmup_secs: int = 180):
        self.warmup_samples = warmup_samples
        self.warmup_secs = warmup_secs
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

        # P3 criteria: skip if std <3¢, range <5¢, or spread >6¢ consistently
        if std < 3.0:
            return False, f"dead_market:std={std:.1f}c"
        if price_range < 5.0:
            return False, f"dead_market:range={price_range:.1f}c"
        if avg_spread > 6.0:
            return False, f"illiquid:avg_spread={avg_spread:.1f}c"

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
# STRATEGY 1: MODEL EDGE (with P1 circuit breakers + P5 fee awareness)
# =============================================================================

class ModelEdgeStrategy(BaseStrategy):
    """
    Enter when market differs from model fair price.
    
    Feb 6 fixes:
      - P1a: Divergence filter (block when |market - model| > divergence_cap)
      - P1b: Final 5min lockout (inherited from BaseStrategy)
      - P1c: Re-entry cooldown after stop (inherited, 120s)
      - P1d: Market decided filter (skip if market >88¢ or <12¢)
      - P5:  Fee-aware minimum edge
    """

    def __init__(self, max_capital: float, model_fair_cents: int):
        params = {
            "min_entry_edge": 10,
            "min_lock_profit": 10,
            "stop_loss": 18,
            "take_profit": 12,
            "max_positions": 2,
            "min_entry_gap_secs": 45,
            "side_cooldown_secs": 120,       # P1c
            "stop_trading_before_close_secs": 300,  # P1b
            "divergence_cap_cents": 20,      # P1a
            "market_decided_hi": 88,         # P1d
            "market_decided_lo": 12,         # P1d
            "max_lock_attempts": 2,          # P5
        }
        super().__init__("model_edge", max_capital, params)
        self.model_fair = model_fair_cents
        self.price_history: deque = deque(maxlen=60)

    def _get_dynamic_fair(
        self,
        current_mid: float,
        secs_to_close: int,
        espn_live_win_pct: Optional[float],
        game_progress: Optional[float],
    ) -> float:
        """
        New behavior:
        - If ESPN win% is available: blend pregame model -> ESPN with aggressive decay (3.5)
        - Else: fallback to legacy blend pregame model -> market midpoint (decay 2.0)
        """
        self.price_history.append(current_mid)

        total_game = 2400
        elapsed = max(0, total_game - secs_to_close)
        progress = min(1.0, elapsed / total_game)

        # Prefer ESPN-supplied progress if available
        if isinstance(game_progress, (int, float)):
            progress = float(max(0.0, min(1.0, game_progress)))

        # ESPN blend path
        if isinstance(espn_live_win_pct, (int, float)):
            espn_pct = float(espn_live_win_pct)
            # allow 0..1 input; if 0..100 mistakenly arrives, normalize
            if espn_pct > 1.0:
                espn_pct /= 100.0
            espn_cents = espn_pct * 100.0

            model_weight = math.exp(-3.5 * progress)
            return model_weight * float(self.model_fair) + (1 - model_weight) * espn_cents

        # Fallback: market blend
        model_weight = math.exp(-2.0 * progress)
        market_mid = sum(self.price_history) / len(self.price_history)
        return model_weight * float(self.model_fair) + (1 - model_weight) * market_mid

    def evaluate_entry(self, prices, secs_to_close, context) -> Tuple[bool, str, int, int, str]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")

        if not all([yes_bid, yes_ask]):
            return False, "", 0, 0, "no_prices"

        mid = (yes_bid + yes_ask) / 2

        # --- P1d: Market decided filter ---
        hi = self.params["market_decided_hi"]
        lo = self.params["market_decided_lo"]
        if mid > hi or mid < lo:
            return False, "", 0, 0, f"market_decided:{mid:.0f}c"

        espn_wp = context.get("espn_live_win_pct")
        game_progress = context.get("game_progress")
        fair = self._get_dynamic_fair(mid, secs_to_close, espn_wp, game_progress)


        # --- P1a: Divergence filter ---
        divergence = abs(mid - fair)

        cap = self.params["divergence_cap_cents"]
        if divergence > cap:
            return False, "", 0, 0, f"divergence:{divergence:.0f}c>{cap}c"

        # --- P5: Fee-aware minimum edge ---
        base_min_edge = self.params["min_entry_edge"]
        fee_min_edge = fee_aware_min_edge(
            int(mid), stop_loss_cents=self.params["stop_loss"]
        )
        effective_min_edge = max(base_min_edge, int(math.ceil(fee_min_edge)))

        # Check YES side
        yes_edge = fair - yes_ask
        # P1c: side cooldown
        yes_cooled, _ = self._side_on_cooldown("yes")

        if yes_edge >= effective_min_edge and not yes_cooled:
            return True, "yes", int(yes_ask), 1, \
                f"edge:{yes_edge:.1f}c,fair:{fair:.1f}c,minE:{effective_min_edge}"

        # Check NO side
        if no_ask:
            no_fair = 100 - fair
            no_edge = no_fair - no_ask
            no_cooled, _ = self._side_on_cooldown("no")

            if no_edge >= effective_min_edge and not no_cooled:
                return True, "no", int(no_ask), 1, \
                    f"edge:{no_edge:.1f}c,fair:{fair:.1f}c,minE:{effective_min_edge}"

        return False, "", 0, 0, "no_edge"

    def evaluate_exit(self, position, prices, secs_to_close, context) -> Tuple[bool, str, int, str]:
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        no_ask = prices.get("imp_no_ask")
        yes_ask = prices.get("imp_yes_ask")

        if position.side == "yes":
            # Try to lock (P5: respect max lock attempts)
            if no_ask and position.lock_attempts < self.params["max_lock_attempts"]:
                profit = 100 - position.entry_price - no_ask
                if profit >= self.params["min_lock_profit"]:
                    position.lock_attempts += 1
                    return True, "lock", int(no_ask), f"lock:{profit:.0f}c"

            if yes_bid:
                pnl = yes_bid - position.entry_price
                # if pnl <= -self.params["stop_loss"]:
                #     return True, "stop", int(yes_bid) - 1, f"stop:{pnl:.0f}c"
                if pnl >= self.params["take_profit"]:
                    return True, "take_profit", int(yes_bid) - 1, f"tp:{pnl:.0f}c"
        else:
            if yes_ask and position.lock_attempts < self.params["max_lock_attempts"]:
                profit = 100 - position.entry_price - yes_ask
                if profit >= self.params["min_lock_profit"]:
                    position.lock_attempts += 1
                    return True, "lock", int(yes_ask), f"lock:{profit:.0f}c"

            if no_bid:
                pnl = no_bid - position.entry_price
                # if pnl <= -self.params["stop_loss"]:
                #     return True, "stop", int(no_bid) - 1, f"stop:{pnl:.0f}c"
                if pnl >= self.params["take_profit"]:
                    return True, "take_profit", int(no_bid) - 1, f"tp:{pnl:.0f}c"

        return False, "", 0, "hold"


# =============================================================================
# STRATEGY 2: MEAN REVERSION (P2 adaptive threshold + dead market detection)
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
    """

    def __init__(self, max_capital: float, preferred_side: Optional[str] = None):
        params = {
            "lookback": 60,
            "high_vol_std_mult": 1.5,
            "low_vol_std_mult": 2.5,
            "low_vol_cutoff": 5.0,
            "reversion_exit_pnl_floor": -3,

            # stop loss intentionally unused / OFF
            "stop_loss": 0,

            # allow more room; you can tune
            "max_positions": 4,

            "min_entry_gap_secs": 8,
            "side_cooldown_secs": 20,

            # IMPORTANT: allow entries into late game; directional filter controls safety
            "stop_trading_before_close_secs": 0,

            # warmup / dead-market checks (unchanged)
            "warmup_samples": 60,
            "warmup_min_range": 5.0,
            "dead_lookback": 30,
            "dead_min_move": 3.0,

            # NEW: direction window (last 5 minutes)
            "directional_close_secs": 300,
            "force_exit_wrong_side": True,

            # --- SIZING ---
            # Each entry uses a fraction of remaining capital, scaled by signal strength.
            # At 1.5σ (minimum trigger in high-vol) → base_frac of remaining capital.
            # At 3σ+ → up to max_frac. Linear interpolation between.
            "size_base_frac": 0.35,       # fraction of remaining capital at minimum σ trigger
            "size_max_frac": 0.80,        # fraction at 3σ+
            "size_sigma_floor": 1.5,      # σ level that maps to base_frac
            "size_sigma_cap": 3.0,        # σ level that maps to max_frac
            "size_min_qty": 3,            # never go below this (keeps every trade meaningful)
            "max_order_qty": 200,         # hard cap per single order
        }
        super().__init__("mean_reversion", max_capital, params)

        self.prices: deque = deque(maxlen=params["lookback"])

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

    def evaluate_entry(self, prices, secs_to_close, context) -> Tuple[bool, str, int, int, str]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")

        if yes_bid is None or yes_ask is None:
            return False, "", 0, 0, "no_prices"

        mid = (yes_bid + yes_ask) / 2
        self.prices.append(mid)

        warmup_ok, warmup_reason = self._check_warmup()
        if not warmup_ok:
            return False, "", 0, 0, warmup_reason

        if self._is_dead_market():
            return False, "", 0, 0, "dead_market"

        mean, std = self._calc_stats()

        std_mult = self.params["low_vol_std_mult"] if std < self.params["low_vol_cutoff"] else self.params["high_vol_std_mult"]
        threshold = std_mult * std

        in_dir = self._in_directional_window(int(secs_to_close))

        # Price dropped → buy YES
        if mid < mean - threshold:
            if in_dir and self.preferred_side != "yes":
                return False, "", 0, 0, f"dir_block:want_yes_pref_{self.preferred_side}"
            cooled, _ = self._side_on_cooldown("yes")
            if cooled:
                return False, "", 0, 0, "cooldown_yes"
            edge = mean - mid
            qty = self._size_qty("yes", int(yes_ask), sigma_mult=edge / std)
            if qty <= 0:
                return False, "", 0, 0, "no_capital"
            return True, "yes", int(yes_ask), qty, f"below:{edge:.0f}c({edge/std:.1f}σ),mult={std_mult},qty={qty}"

        # Price spiked → buy NO
        if mid > mean + threshold and no_ask is not None:
            if in_dir and self.preferred_side != "no":
                return False, "", 0, 0, f"dir_block:want_no_pref_{self.preferred_side}"
            cooled, _ = self._side_on_cooldown("no")
            if cooled:
                return False, "", 0, 0, "cooldown_no"
            edge = mid - mean
            qty = self._size_qty("no", int(no_ask), sigma_mult=edge / std)
            if qty <= 0:
                return False, "", 0, 0, "no_capital"
            return True, "no", int(no_ask), qty, f"above:{edge:.0f}c({edge/std:.1f}σ),mult={std_mult},qty={qty}"

        return False, "", 0, 0, "in_range"

    def evaluate_exit(self, position, prices, secs_to_close, context) -> Tuple[bool, str, int, str]:
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        yes_ask = prices.get("imp_yes_ask")

        if yes_bid is None:
            return False, "", 0, "no_bid"

        # NEW: late-game flatten wrong-side inventory
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

        # Existing: reversion exit (stop loss OFF) — now PATIENT via pnl floor gate
        pnl_floor = float(self.params.get("reversion_exit_pnl_floor", -3))

        if position.side == "yes":
            pnl = float(yes_bid) - float(position.entry_price)

            # If we're still very underwater, do NOT "revert exit" just because mean drifted.
            if pnl < pnl_floor:
                return False, "", 0, f"hold_pnl_floor:{pnl:.0f}c<{pnl_floor:.0f}c"

            if mid >= mean - 2:
                return True, "take_profit", max(1, int(yes_bid) - 1), f"reverted:{pnl:.0f}c"

        else:
            if no_bid is not None:
                pnl = float(no_bid) - float(position.entry_price)

                if pnl < pnl_floor:
                    return False, "", 0, f"hold_pnl_floor:{pnl:.0f}c<{pnl_floor:.0f}c"

                if mid <= mean + 2:
                    return True, "take_profit", max(1, int(no_bid) - 1), f"reverted:{pnl:.0f}c"

        return False, "", 0, "hold"

# =============================================================================
# STRATEGY 3: PREGAME ANCHORED + LIVE MANAGEMENT (partner test)
# =============================================================================

class PregameAnchoredStrategy(BaseStrategy):
    """
    Partner test strategy:
      - Place ONE pregame limit entry within the last ~5 minutes before tip,
        improving the market ask by a cushion (5-6c).
      - If filled, manage live with:
          1) lock ASAP when profitable,
          2) take profit at +10-15c,
          3) exit with 10 minutes remaining if still open,
          4) cut at -15 to -20c.

    Fair (pregame) is a blend of:
      - market midpoint (default 70%)
      - two models (default 30%), internally split between our model and partner.
    """

    def __init__(
        self,
        max_capital: float,
        model_fair_cents: int,
        partner_fair_cents: int,
        cushion_cents: int = 6,
        market_weight: float = 0.70,
        model_share: float = 0.50,
    ):
        params = {
            "min_entry_edge": 8,              # pregame: require a real mismatch
            "min_lock_profit": 10,
            "stop_loss": 18,
            "take_profit": 15,
            "exit_with_secs_remaining": 600,  # 10 minutes
            "pregame_window_secs": 300,       # 5 minutes
            "max_positions": 1,
            "min_entry_gap_secs": 999999,     # we handle re-tries ourselves
            "side_cooldown_secs": 0,
            "stop_trading_before_close_secs": 0,  # allow entry pregame even though "close" is far
            "max_lock_attempts": 2,
        }
        super().__init__("pregame_anchored", max_capital, params)
        self.model_fair = int(model_fair_cents)
        self.partner_fair = int(partner_fair_cents)
        self.cushion_cents = int(cushion_cents)
        self.market_weight = float(market_weight)
        self.model_share = float(model_share)

        # State: only attempt ONE pregame order; if it doesn't fill, we skip the game.
        self._pregame_attempted = False
        self._pregame_failed = False

    # --- BaseStrategy hooks ---

    def entry_wait_secs(self, context: Dict[str, Any]) -> int:
        """Wait up to tip (<=5m window) so the limit rests and can fill."""
        secs_to_tip = context.get("secs_to_tip")
        if isinstance(secs_to_tip, (int, float)):
            window = int(self.params.get("pregame_window_secs", 300))
            if 0 < secs_to_tip <= window:
                return max(15, int(secs_to_tip))
        return 15

    def on_entry_result(self, filled: bool, context: Dict[str, Any]) -> None:
        # If our single pregame attempt doesn't fill, do not keep chasing.
        if self._pregame_attempted and not filled:
            self._pregame_failed = True

    # --- Internal helpers ---

    def _pregame_fair_yes(self, market_mid: float) -> float:
        models_weight = 1.0 - self.market_weight
        model_mix = self.model_share * float(self.model_fair) + (1.0 - self.model_share) * float(self.partner_fair)
        return self.market_weight * float(market_mid) + models_weight * model_mix

    def _improved_limit(self, bid: Optional[int], ask: Optional[int]) -> Optional[int]:
        if ask is None:
            return None
        # target = ask - cushion, but ensure we improve the bid by at least 1 tick if bid exists
        target = int(round(ask - self.cushion_cents))
        if bid is not None:
            target = max(target, int(bid) + 1)
        target = min(target, int(ask) - 1)
        if target < 1 or target > 99:
            return None
        return target

    # --- Strategy interface ---

    def evaluate_entry(self, prices, secs_to_close, context) -> Tuple[bool, str, int, int, str]:
        if self._pregame_failed:
            return False, "", 0, 0, "pregame_failed"
        if self.positions:
            return False, "", 0, 0, "already_in"
        if self._pregame_attempted:
            return False, "", 0, 0, "pregame_attempted"

        secs_to_tip = context.get("secs_to_tip")
        window = int(self.params.get("pregame_window_secs", 300))
        if not isinstance(secs_to_tip, (int, float)):
            return False, "", 0, 0, "no_secs_to_tip"
        if not (0 < secs_to_tip <= window):
            return False, "", 0, 0, f"not_in_window:{int(secs_to_tip)}s"

        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_bid = prices.get("best_no_bid")
        no_ask = prices.get("imp_no_ask")

        if yes_bid is None or yes_ask is None:
            return False, "", 0, 0, "no_yes_prices"

        mid = (yes_bid + yes_ask) / 2
        fair_yes = self._pregame_fair_yes(mid)

        min_edge = int(self.params.get("min_entry_edge", 8))

        # Decide side based on fair vs market
        side = ""
        if fair_yes >= mid + min_edge:
            side = "yes"
            px = self._improved_limit(yes_bid, yes_ask)
        elif fair_yes <= mid - min_edge:
            side = "no"
            px = self._improved_limit(no_bid, no_ask)
        else:
            return False, "", 0, 0, f"no_edge:fair={fair_yes:.1f},mid={mid:.1f}"

        if not side or px is None:
            return False, "", 0, 0, "no_price_after_improve"

        self._pregame_attempted = True
        return True, side, int(px), 1, f"pregame:fair={fair_yes:.1f} mid={mid:.1f} tip={int(secs_to_tip)}s"

    def evaluate_exit(self, position, prices, secs_to_close, context) -> Tuple[bool, str, int, str]:
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        no_ask = prices.get("imp_no_ask")
        yes_ask = prices.get("imp_yes_ask")

        # 3) time-based exit with 10 minutes remaining
        exit_with = int(self.params.get("exit_with_secs_remaining", 600))
        if secs_to_close <= exit_with:
            if position.side == "yes" and yes_bid is not None:
                return True, "timeout", int(yes_bid) - 1, f"time_exit:{int(secs_to_close)}s"
            if position.side == "no" and no_bid is not None:
                return True, "timeout", int(no_bid) - 1, f"time_exit:{int(secs_to_close)}s"

        # 1) lock ASAP if profitable
        if position.side == "yes":
            if no_ask is not None and position.lock_attempts < self.params["max_lock_attempts"]:
                profit = 100 - position.entry_price - no_ask
                if profit >= self.params["min_lock_profit"]:
                    position.lock_attempts += 1
                    return True, "lock", int(no_ask), f"lock:{profit:.0f}c"

            # 2/4) take profit / stop
            if yes_bid is not None:
                pnl = yes_bid - position.entry_price
                if pnl >= self.params["take_profit"]:
                    return True, "take_profit", int(yes_bid) - 1, f"tp:{pnl:.0f}c"
                # if pnl <= -self.params["stop_loss"]:
                #     return True, "stop", int(yes_bid) - 1, f"stop:{pnl:.0f}c"

        else:
            if yes_ask is not None and position.lock_attempts < self.params["max_lock_attempts"]:
                profit = 100 - position.entry_price - yes_ask
                if profit >= self.params["min_lock_profit"]:
                    position.lock_attempts += 1
                    return True, "lock", int(yes_ask), f"lock:{profit:.0f}c"

            if no_bid is not None:
                pnl = no_bid - position.entry_price
                if pnl >= self.params["take_profit"]:
                    return True, "take_profit", int(no_bid) - 1, f"tp:{pnl:.0f}c"
                # if pnl <= -self.params["stop_loss"]:
                #     return True, "stop", int(no_bid) - 1, f"stop:{pnl:.0f}c"

        return False, "", 0, "hold"



# =============================================================================
# STRATEGY 3: SPREAD CAPTURE
# =============================================================================

class SpreadCaptureStrategy(BaseStrategy):
    """Profit from wide spreads. Largely unchanged — wasn't the problem."""

    def __init__(self, max_capital: float):
        params = {
            "min_spread": 6,
            "improve_cents": 1,
            "min_lock_profit": 8,
            "stop_loss": 12,
            "max_positions": 2,
            "min_entry_gap_secs": 60,
            "side_cooldown_secs": 90,
            "stop_trading_before_close_secs": 300,
            "max_lock_attempts": 2,
        }
        super().__init__("spread_capture", max_capital, params)

    def evaluate_entry(self, prices, secs_to_close, context) -> Tuple[bool, str, int, int, str]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_bid = prices.get("best_no_bid")

        if not all([yes_bid, yes_ask, no_bid]):
            return False, "", 0, 0, "no_prices"

        spread = yes_ask - yes_bid
        if spread < self.params["min_spread"]:
            return False, "", 0, 0, f"spread:{spread}c"

        our_bid = yes_bid + self.params["improve_cents"]
        potential_lock_cost = our_bid + (100 - no_bid)
        potential_profit = 100 - potential_lock_cost

        if potential_profit >= self.params["min_lock_profit"]:
            return True, "yes", our_bid, 1, f"spread:{spread}c,pot:{potential_profit:.0f}c"

        return False, "", 0, 0, "no_opp"

    def evaluate_exit(self, position, prices, secs_to_close, context) -> Tuple[bool, str, int, str]:
        yes_bid = prices.get("best_yes_bid")
        no_ask = prices.get("imp_no_ask")
        yes_ask = prices.get("imp_yes_ask")
        no_bid = prices.get("best_no_bid")

        if position.side == "yes":
            if no_ask and position.lock_attempts < self.params["max_lock_attempts"]:
                profit = 100 - position.entry_price - no_ask
                if profit >= self.params["min_lock_profit"]:
                    position.lock_attempts += 1
                    return True, "lock", int(no_ask), f"lock:{profit:.0f}c"
            if yes_bid:
                pnl = yes_bid - position.entry_price
                # if pnl <= -self.params["stop_loss"]:
                #     return True, "stop", int(yes_bid) - 1, f"stop:{pnl:.0f}c"
        else:
            if yes_ask and position.lock_attempts < self.params["max_lock_attempts"]:
                profit = 100 - position.entry_price - yes_ask
                if profit >= self.params["min_lock_profit"]:
                    position.lock_attempts += 1
                    return True, "lock", int(yes_ask), f"lock:{profit:.0f}c"
            if no_bid:
                pnl = no_bid - position.entry_price
                # if pnl <= -self.params["stop_loss"]:
                #     return True, "stop", int(no_bid) - 1, f"stop:{pnl:.0f}c"

        return False, "", 0, "waiting"


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
    ):
        self.label = game_label
        self.ticker = ticker
        self.market = market
        self.strategies = strategies
        self.private_key = private_key
        self.close_time = parse_iso(market["close_time"])
        self.espn_clock = espn_clock

        # P3: Market quality monitor
        self.quality = MarketQualityMonitor(warmup_samples=40, warmup_secs=180)
        self._market_killed = False
        self._market_kill_reason = ""

        # Logging
        os.makedirs("logs", exist_ok=True)
        ts = utc_now().strftime("%Y%m%d_%H%M%S")
        base = f"logs/multistrat_{game_label}_{ts}"
        self.snapshot_path = f"{base}_snapshots.csv"
        self.trade_path = f"{base}_trades.csv"
        self.position_path = f"{base}_positions.csv"
        self.event_path = f"{base}_events.csv"
        self._init_logs()
        self.snapshots = 0

    def _init_logs(self):
        with open(self.snapshot_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "ticker", "secs_to_close", "clock_source",
                "yes_bid", "yes_ask", "no_bid", "no_ask", "mid", "spread",
                "mq_std", "mq_tradeable","espn_live_win_pct", "game_progress", "secs_to_tip",
            ])
        with open(self.trade_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "ticker", "strategy", "action", "side",
                "intended_price", "fill_price", "qty", "fee_cents",
                "reason", "order_id",
            ])
        with open(self.position_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "position_id", "strategy", "side",
                "entry_price", "entry_time", "entry_fee",
                "exit_price", "exit_time", "exit_type", "exit_fee",
                "lock_price", "gross_pnl", "net_pnl", "hold_secs",
            ])
        with open(self.event_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "strategy", "event", "detail",
            ])

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

        print_status(f"[{self.label}][{strategy.name}] ENTRY {side.upper()} {qty}x @ {price}c — {reason}")
        self._log_event(strategy.name, "entry_attempt", f"{side} {qty}x@{price}c {reason}")

        try:
            # Liquidity check before sending order
            ob = fetch_orderbook(self.ticker)
            px = derive_prices(ob)
            if not has_fill_liquidity_for_implied_buy(px, side, price, min_qty=max(MIN_LIQUIDITY_CONTRACTS, qty)):
                print_status(f"[{self.label}][{strategy.name}] SKIP — no liquidity for {side} @{price}c")
                self._log_event(strategy.name, "entry_skip_liquidity", f"{side}@{price}c")
                return None


            order_id = place_limit_buy(self.private_key, self.ticker, side, price, qty)
            wait_secs = int(strategy.entry_wait_secs(context) or 15)
            filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, side, max_wait_secs=wait_secs)


            if filled > 0:
                actual = vwap if vwap else float(price)
                pos = strategy.record_entry(side, actual, filled, reason)
                fee = calc_taker_fee(int(actual), filled)
                print_status(f"[{self.label}][{strategy.name}] FILLED {filled}x @ {actual:.1f}c")
                strategy.on_entry_result(True, context)
                self._log_trade(strategy.name, "entry_fill", side, price, actual, filled, fee, reason, order_id)
                return pos
            else:
                print_status(f"[{self.label}][{strategy.name}] NO FILL")
                strategy.on_entry_result(False, context)
                self._log_trade(strategy.name, "entry_nofill", side, price, 0, 0, 0, reason, order_id)
                return None

        except Exception as e:
            print_status(f"[{self.label}][{strategy.name}] ENTRY ERROR: {e}")
            self._log_event(strategy.name, "entry_error", str(e))
            return None

    def _execute_exit(self, strategy: BaseStrategy, position: Position,
                      exit_type: str, price: int, reason: str):
        print_status(f"[{self.label}][{strategy.name}] EXIT {position.side.upper()} @ {price}c — {reason}")
        self._log_event(strategy.name, f"exit_{exit_type}_attempt", f"{position.side}@{price}c {reason}")

        lock_price = None
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
            else:
                order_id = place_limit_sell(self.private_key, self.ticker, position.side, price, position.qty)
                filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, position.side, max_wait_secs=15)
                if filled > 0:
                    actual_exit = vwap if vwap else float(price)
                else:
                    print_status(f"[{self.label}][{strategy.name}] EXIT NO FILL")
                    self._log_event(strategy.name, "exit_nofill", f"{position.side}@{price}c")
                    return None

            closed = strategy.record_exit(position, exit_type, actual_exit, lock_price)
            fee = calc_taker_fee(int(actual_exit), position.qty)
            print_status(f"[{self.label}][{strategy.name}] CLOSED: net={closed.net_pnl:.1f}c")
            self._log_trade(strategy.name, f"exit_{exit_type}", position.side, price,
                           actual_exit, position.qty, fee, reason, order_id)
            self._log_position(closed)
            return closed

        except Exception as e:
            print_status(f"[{self.label}][{strategy.name}] EXIT ERROR: {e}")
            self._log_event(strategy.name, "exit_error", str(e))
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

    def _get_secs_to_close(self) -> Tuple[float, str, Optional[float], Optional[float], Optional[int]]:
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
                espn_wp = ctx.get("espn_live_win_pct")
                game_progress = ctx.get("game_progress")
                secs_to_tip = ctx.get("secs_to_tip")

                game_secs = ctx.get("secs_to_game_end")
                why = ctx.get("why", "espn")

                if game_secs is not None:
                    return min(kalshi_secs, float(game_secs)), f"espn:{why}", espn_wp, game_progress, secs_to_tip

                # pregame: we still return kalshi_secs (placeholder), but expose secs_to_tip
                return kalshi_secs, f"espn:{why}", espn_wp, game_progress, secs_to_tip

            except Exception:
                pass

        return kalshi_secs, "kalshi", espn_wp, game_progress, secs_to_tip

    def run(self) -> Dict[str, Any]:
        print_status(f"[{self.label}] Starting — Strategies: {[s.name for s in self.strategies]}")
        print_status(f"[{self.label}] Close: {self.close_time}")

        while True:
            secs_to_close, clock_source, espn_wp, game_progress, secs_to_tip = self._get_secs_to_close()

            if secs_to_close <= 0:
                print_status(f"[{self.label}] Market closed")
                break

            try:
                ob = fetch_orderbook(self.ticker)
                prices = derive_prices(ob)
            except Exception as e:
                print_status(f"[{self.label}] Orderbook error: {e}")
                time.sleep(POLL_INTERVAL_SECS)
                continue

            self.snapshots += 1

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
                "secs_to_close": secs_to_close,
                "market_quality_ok": mq_ok,
                "espn_live_win_pct": espn_wp,     # 0..1 or None
                "game_progress": game_progress,  # 0..1 or None
                "secs_to_tip": secs_to_tip,      # int seconds or None
            }


            for strategy in self.strategies:
                # --- Exits first (always allowed, even in dead markets) ---
                for pos in list(strategy.positions):
                    should_exit, exit_type, exit_price, reason = strategy.evaluate_exit(
                        pos, prices, int(secs_to_close), context
                    )
                    if should_exit:
                        self._execute_exit(strategy, pos, exit_type, exit_price, reason)

                # --- Entries (only in tradeable markets) ---
                if not mq_ok:
                    continue  # P3: skip entries in dead/illiquid markets

                can_enter, _ = strategy.can_enter(secs_to_close=int(secs_to_close))
                if can_enter:
                    should_enter, side, price, qty, reason = strategy.evaluate_entry(
                        prices, int(secs_to_close), context
                    )
                    if should_enter:
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

        return result