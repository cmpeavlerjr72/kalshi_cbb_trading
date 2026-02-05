# tonight_multistrategy.py
#
# TRUE MULTI-STRATEGY TESTING for Feb 5, 2026
#
# Runs 3 different strategies in parallel on each game:
#   1. Model Edge - Your current approach (improved)
#   2. Mean Reversion - Pure price-based, no model
#   3. Spread Capture - Market making / liquidity provision
#
# Each strategy gets its own allocation and logs separately
# Tomorrow we compare which performed best
#
# Run with: python tonight_multistrategy.py

import os
import sys
import time
import json
import math
import threading
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import datetime as dt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Import from your existing combo_vnext
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
    SERIES_TICKER,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Tonight's games
GAMES = [
    {
        "label": "Tarleton_vs_CalBaptist",
        "team_name": "TARL",
        "model_p_win": 0.49,
        "tip_time": "10:00 PM ET",
        "segment": "Home Dog",  # Best historical segment
    },
    {
        "label": "Fairfield_vs_SacredHeart",
        "team_name": "FAIR", 
        "model_p_win": 0.58,
        "tip_time": "7:00 PM ET",
        "segment": "Away Fav",
    },
]

# Strategy allocations (per game)
STRATEGY_ALLOCATIONS = {
    "model_edge": 2.00,      # $2 - Your existing approach
    "mean_reversion": 2.00,  # $2 - Pure price reversion
    "spread_capture": 2.00,  # $2 - Spread harvesting
}

# Global settings
POLL_INTERVAL_SECS = 3.0
STOP_TRADING_BEFORE_CLOSE_SECS = 300

# =============================================================================
# FEE CALCULATIONS
# =============================================================================

def calc_taker_fee(price_cents: int, qty: int = 1) -> float:
    """Kalshi taker fee in cents"""
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    p = price_cents / 100.0
    fee_per = min(0.07 * p * (1 - p), 0.02)
    return fee_per * qty * 100

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    """Tracks an open position"""
    id: str
    strategy: str
    side: str  # "yes" or "no"
    entry_price: float
    qty: int
    entry_time: dt.datetime
    entry_reason: str
    entry_fee: float = 0.0
    
@dataclass
class ClosedPosition:
    """Record of a completed trade"""
    id: str
    strategy: str
    side: str
    entry_price: float
    exit_price: float
    qty: int
    entry_time: str
    exit_time: str
    exit_type: str  # "lock", "stop", "take_profit", "timeout"
    entry_fee: float
    exit_fee: float
    gross_pnl: float
    net_pnl: float
    hold_secs: int
    lock_price: Optional[float] = None

# =============================================================================
# BASE STRATEGY CLASS
# =============================================================================

class BaseStrategy(ABC):
    """Abstract base for all strategies"""
    
    def __init__(self, name: str, max_capital: float, params: Dict[str, Any]):
        self.name = name
        self.max_capital = max_capital
        self.params = params
        
        # State
        self.positions: List[Position] = []
        self.closed: List[ClosedPosition] = []
        self.capital_used = 0.0
        self.position_counter = 0
        
        # Throttling
        self.last_entry_time: Optional[float] = None
        self.min_entry_gap_secs = params.get("min_entry_gap_secs", 45)
        
        # Tracking
        self.total_fees = 0.0
    
    def _next_position_id(self) -> str:
        self.position_counter += 1
        return f"{self.name}_{self.position_counter}"
    
    def can_enter(self) -> Tuple[bool, str]:
        """Check if we can enter a new position"""
        if self.capital_used >= self.max_capital:
            return False, "max_capital"
        
        if len(self.positions) >= self.params.get("max_positions", 2):
            return False, "max_positions"
        
        if self.last_entry_time:
            elapsed = time.time() - self.last_entry_time
            if elapsed < self.min_entry_gap_secs:
                return False, f"throttle:{elapsed:.0f}s"
        
        return True, "ok"
    
    @abstractmethod
    def evaluate_entry(
        self, 
        prices: Dict[str, Any],
        secs_to_close: int,
        context: Dict[str, Any]
    ) -> Tuple[bool, str, int, int, str]:
        """Returns: (should_enter, side, price, qty, reason)"""
        pass
    
    @abstractmethod
    def evaluate_exit(
        self,
        position: Position,
        prices: Dict[str, Any],
        secs_to_close: int,
        context: Dict[str, Any]
    ) -> Tuple[bool, str, int, str]:
        """Returns: (should_exit, exit_type, exit_price, reason)"""
        pass
    
    def record_entry(self, side: str, price: float, qty: int, reason: str) -> Position:
        """Record a new position"""
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
        
        # Update capital used
        if side == "yes":
            self.capital_used += (100 - price) * qty / 100
        else:
            self.capital_used += price * qty / 100
        
        return pos
    
    def record_exit(
        self, 
        position: Position, 
        exit_type: str, 
        exit_price: float,
        lock_price: Optional[float] = None
    ) -> ClosedPosition:
        """Record closing a position"""
        exit_fee = calc_taker_fee(int(exit_price), position.qty)
        self.total_fees += exit_fee
        
        # Calculate P&L
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
        
        # Free up capital
        if position.side == "yes":
            self.capital_used -= (100 - position.entry_price) * position.qty / 100
        else:
            self.capital_used -= position.entry_price * position.qty / 100
        
        return closed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.closed:
            return {
                "strategy": self.name,
                "trades": 0,
                "gross_pnl": 0,
                "fees": self.total_fees,
                "net_pnl": -self.total_fees,
            }
        
        gross = sum(c.gross_pnl for c in self.closed)
        net = sum(c.net_pnl for c in self.closed)
        wins = [c for c in self.closed if c.net_pnl > 0]
        locks = [c for c in self.closed if c.exit_type == "lock"]
        stops = [c for c in self.closed if c.exit_type == "stop"]
        
        return {
            "strategy": self.name,
            "trades": len(self.closed),
            "wins": len(wins),
            "win_rate": len(wins) / len(self.closed) if self.closed else 0,
            "locks": len(locks),
            "stops": len(stops),
            "gross_pnl": gross,
            "fees": self.total_fees,
            "net_pnl": net,
            "avg_hold_secs": sum(c.hold_secs for c in self.closed) / len(self.closed),
            "open_positions": len(self.positions),
        }

# =============================================================================
# STRATEGY 1: MODEL EDGE (Your Current Approach, Improved)
# =============================================================================

class ModelEdgeStrategy(BaseStrategy):
    """
    Enter when market price differs significantly from model fair price.
    Try to lock pairs for guaranteed profit.
    """
    
    def __init__(self, max_capital: float, model_fair_cents: int):
        params = {
            "min_entry_edge": 10,      # Fee-aware threshold
            "min_lock_profit": 12,     # Min locked profit per contract
            "stop_loss": 20,           # Stop loss in cents
            "take_profit": 12,         # Take profit in cents
            "max_positions": 2,
            "min_entry_gap_secs": 45,
        }
        super().__init__("model_edge", max_capital, params)
        self.model_fair = model_fair_cents
        self.price_history: deque = deque(maxlen=30)
    
    def _get_dynamic_fair(self, current_mid: float, secs_to_close: int) -> float:
        """Blend model with market as game progresses"""
        self.price_history.append(current_mid)
        
        total_game = 2400
        elapsed = max(0, total_game - secs_to_close)
        progress = min(1.0, elapsed / total_game)
        
        model_weight = math.exp(-2 * progress)
        market_mid = sum(self.price_history) / len(self.price_history)
        
        return model_weight * self.model_fair + (1 - model_weight) * market_mid
    
    def evaluate_entry(self, prices, secs_to_close, context) -> Tuple[bool, str, int, int, str]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")
        
        if not all([yes_bid, yes_ask]):
            return False, "", 0, 0, "no_prices"
        
        mid = (yes_bid + yes_ask) / 2
        fair = self._get_dynamic_fair(mid, secs_to_close)
        
        # Check YES side
        yes_edge = fair - yes_ask
        if yes_edge >= self.params["min_entry_edge"]:
            return True, "yes", yes_ask, 1, f"edge:{yes_edge:.1f}c"
        
        # Check NO side
        if no_ask:
            no_fair = 100 - fair
            no_edge = no_fair - no_ask
            if no_edge >= self.params["min_entry_edge"]:
                return True, "no", no_ask, 1, f"edge:{no_edge:.1f}c"
        
        return False, "", 0, 0, "no_edge"
    
    def evaluate_exit(self, position, prices, secs_to_close, context) -> Tuple[bool, str, int, str]:
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        no_ask = prices.get("imp_no_ask")
        yes_ask = prices.get("imp_yes_ask")
        
        if position.side == "yes":
            # Try to lock
            if no_ask:
                profit = 100 - position.entry_price - no_ask
                if profit >= self.params["min_lock_profit"]:
                    return True, "lock", no_ask, f"lock:{profit:.0f}c"
            
            if yes_bid:
                pnl = yes_bid - position.entry_price
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", yes_bid - 1, f"stop:{pnl:.0f}c"
                if pnl >= self.params["take_profit"]:
                    return True, "take_profit", yes_bid - 1, f"tp:{pnl:.0f}c"
        else:
            if yes_ask:
                profit = 100 - position.entry_price - yes_ask
                if profit >= self.params["min_lock_profit"]:
                    return True, "lock", yes_ask, f"lock:{profit:.0f}c"
            
            if no_bid:
                pnl = no_bid - position.entry_price
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", no_bid - 1, f"stop:{pnl:.0f}c"
                if pnl >= self.params["take_profit"]:
                    return True, "take_profit", no_bid - 1, f"tp:{pnl:.0f}c"
        
        return False, "", 0, "hold"

# =============================================================================
# STRATEGY 2: MEAN REVERSION (No Model Needed)
# =============================================================================

class MeanReversionStrategy(BaseStrategy):
    """
    Pure price-based: buy when price drops sharply below moving average,
    exit when it reverts. No directional model needed.
    """
    
    def __init__(self, max_capital: float):
        params = {
            "lookback": 20,
            "entry_std_mult": 1.5,
            "stop_loss": 15,
            "max_positions": 2,
            "min_entry_gap_secs": 30,
        }
        super().__init__("mean_reversion", max_capital, params)
        self.prices: deque = deque(maxlen=params["lookback"])
    
    def _calc_stats(self) -> Tuple[float, float]:
        if len(self.prices) < 5:
            return 50.0, 10.0
        
        prices = list(self.prices)
        mean = sum(prices) / len(prices)
        var = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = math.sqrt(var) if var > 0 else 5.0
        
        return mean, std
    
    def evaluate_entry(self, prices, secs_to_close, context) -> Tuple[bool, str, int, int, str]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")
        
        if not yes_bid or not yes_ask:
            return False, "", 0, 0, "no_prices"
        
        mid = (yes_bid + yes_ask) / 2
        self.prices.append(mid)
        
        if len(self.prices) < 10:
            return False, "", 0, 0, f"warming:{len(self.prices)}/10"
        
        mean, std = self._calc_stats()
        threshold = self.params["entry_std_mult"] * std
        
        # Price dropped - buy YES
        if mid < mean - threshold:
            edge = mean - mid
            return True, "yes", yes_ask, 1, f"below:{edge:.0f}c({edge/std:.1f}σ)"
        
        # Price spiked - buy NO
        if mid > mean + threshold and no_ask:
            edge = mid - mean
            return True, "no", no_ask, 1, f"above:{edge:.0f}c({edge/std:.1f}σ)"
        
        return False, "", 0, 0, "in_range"
    
    def evaluate_exit(self, position, prices, secs_to_close, context) -> Tuple[bool, str, int, str]:
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        yes_ask = prices.get("imp_yes_ask")
        
        if not yes_bid:
            return False, "", 0, "no_bid"
        
        mid = (yes_bid + (yes_ask or yes_bid)) / 2
        mean, std = self._calc_stats()
        
        if position.side == "yes":
            pnl = yes_bid - position.entry_price
            
            # Reverted to mean
            if mid >= mean - 2:
                return True, "take_profit", yes_bid - 1, f"reverted:{pnl:.0f}c"
            
            if pnl <= -self.params["stop_loss"]:
                return True, "stop", yes_bid - 1, f"stop:{pnl:.0f}c"
        
        else:
            if no_bid:
                pnl = no_bid - position.entry_price
                
                if mid <= mean + 2:
                    return True, "take_profit", no_bid - 1, f"reverted:{pnl:.0f}c"
                
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", no_bid - 1, f"stop:{pnl:.0f}c"
        
        return False, "", 0, "hold"

# =============================================================================
# STRATEGY 3: SPREAD CAPTURE (Market Making Lite)
# =============================================================================

class SpreadCaptureStrategy(BaseStrategy):
    """
    Profit from wide bid-ask spreads. Place improving limit orders
    and try to lock quickly when filled.
    """
    
    def __init__(self, max_capital: float):
        params = {
            "min_spread": 6,
            "improve_cents": 1,
            "min_lock_profit": 8,
            "stop_loss": 12,
            "max_positions": 2,
            "min_entry_gap_secs": 60,
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
            if no_ask:
                profit = 100 - position.entry_price - no_ask
                if profit >= self.params["min_lock_profit"]:
                    return True, "lock", no_ask, f"lock:{profit:.0f}c"
            
            if yes_bid:
                pnl = yes_bid - position.entry_price
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", yes_bid - 1, f"stop:{pnl:.0f}c"
        
        else:
            if yes_ask:
                profit = 100 - position.entry_price - yes_ask
                if profit >= self.params["min_lock_profit"]:
                    return True, "lock", yes_ask, f"lock:{profit:.0f}c"
            
            if no_bid:
                pnl = no_bid - position.entry_price
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", no_bid - 1, f"stop:{pnl:.0f}c"
        
        return False, "", 0, "waiting"

# =============================================================================
# GAME RUNNER
# =============================================================================

class GameRunner:
    """Runs multiple strategies on one game"""
    
    def __init__(
        self,
        game_label: str,
        ticker: str,
        market: Dict[str, Any],
        strategies: List[BaseStrategy],
        private_key,
    ):
        self.label = game_label
        self.ticker = ticker
        self.market = market
        self.strategies = strategies
        self.private_key = private_key
        self.close_time = parse_iso(market["close_time"])
        
        # Logging
        os.makedirs("logs", exist_ok=True)
        ts = utc_now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"logs/multistrat_{game_label}_{ts}.csv"
        self._init_log()
        
        self.snapshots = 0
    
    def _init_log(self):
        with open(self.log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "ticker", "secs_to_close",
                "yes_bid", "yes_ask", "no_bid", "no_ask", "mid", "spread",
                "strategy", "action", "side", "price", "qty", "reason",
                "position_id", "pnl",
            ])
    
    def _log(self, prices: Dict, strategy: str, action: str, side: str = "",
             price: int = 0, qty: int = 0, reason: str = "", 
             position_id: str = "", pnl: float = 0):
        now = utc_now()
        secs = (self.close_time - now).total_seconds()
        
        yes_bid = prices.get("best_yes_bid", "")
        yes_ask = prices.get("imp_yes_ask", "")
        no_bid = prices.get("best_no_bid", "")
        no_ask = prices.get("imp_no_ask", "")
        
        mid = ""
        spread = ""
        if yes_bid and yes_ask:
            mid = (yes_bid + yes_ask) / 2
            spread = yes_ask - yes_bid
        
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                now.isoformat(), self.ticker, int(secs),
                yes_bid, yes_ask, no_bid, no_ask, 
                f"{mid:.1f}" if mid else "", spread,
                strategy, action, side, price, qty, reason,
                position_id, f"{pnl:.2f}" if pnl else "",
            ])
    
    def _execute_entry(self, strategy: BaseStrategy, side: str, price: int, qty: int, reason: str):
        print_status(f"[{self.label}][{strategy.name}] ENTRY {side.upper()} {qty}x @ {price}c - {reason}")
        
        try:
            order_id = place_limit_buy(self.private_key, self.ticker, side, price, qty)
            filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, side, max_wait_secs=15)
            
            if filled > 0:
                actual_price = vwap if vwap else float(price)
                pos = strategy.record_entry(side, actual_price, filled, reason)
                print_status(f"[{self.label}][{strategy.name}] FILLED {filled}x @ {actual_price:.1f}c")
                return pos
            else:
                print_status(f"[{self.label}][{strategy.name}] NO FILL")
                return None
                
        except Exception as e:
            print_status(f"[{self.label}][{strategy.name}] ENTRY ERROR: {e}")
            return None
    
    def _execute_exit(self, strategy: BaseStrategy, position: Position, 
                      exit_type: str, price: int, reason: str):
        print_status(f"[{self.label}][{strategy.name}] EXIT {position.side.upper()} @ {price}c - {reason}")
        
        lock_price = None
        
        try:
            if exit_type == "lock":
                lock_side = "no" if position.side == "yes" else "yes"
                order_id = place_limit_buy(self.private_key, self.ticker, lock_side, price, position.qty)
                filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, lock_side, max_wait_secs=15)
                
                if filled > 0:
                    lock_price = vwap if vwap else float(price)
                    actual_exit_price = lock_price
                else:
                    print_status(f"[{self.label}][{strategy.name}] LOCK NO FILL")
                    return None
            else:
                order_id = place_limit_sell(self.private_key, self.ticker, position.side, price, position.qty)
                filled, vwap = wait_for_fill_or_timeout(self.private_key, order_id, position.side, max_wait_secs=15)
                
                if filled > 0:
                    actual_exit_price = vwap if vwap else float(price)
                else:
                    print_status(f"[{self.label}][{strategy.name}] EXIT NO FILL")
                    return None
            
            closed = strategy.record_exit(position, exit_type, actual_exit_price, lock_price)
            print_status(f"[{self.label}][{strategy.name}] CLOSED: net = {closed.net_pnl:.1f}c")
            return closed
            
        except Exception as e:
            print_status(f"[{self.label}][{strategy.name}] EXIT ERROR: {e}")
            return None
    
    def run(self):
        print_status(f"[{self.label}] Starting - Strategies: {[s.name for s in self.strategies]}")
        print_status(f"[{self.label}] Close time: {self.close_time}")
        
        while True:
            now = utc_now()
            secs_to_close = (self.close_time - now).total_seconds()
            
            if secs_to_close <= 0:
                print_status(f"[{self.label}] Market closed")
                break
            
            if secs_to_close <= STOP_TRADING_BEFORE_CLOSE_SECS:
                print_status(f"[{self.label}] Near close - stopping")
                break
            
            try:
                ob = fetch_orderbook(self.ticker)
                prices = derive_prices(ob)
            except Exception as e:
                print_status(f"[{self.label}] Orderbook error: {e}")
                time.sleep(POLL_INTERVAL_SECS)
                continue
            
            self.snapshots += 1
            
            # Log tick
            self._log(prices, "all", "tick")
            
            # Status every 30 ticks
            if self.snapshots % 30 == 0:
                yes_bid = prices.get("best_yes_bid", "?")
                no_bid = prices.get("best_no_bid", "?")
                open_count = sum(len(s.positions) for s in self.strategies)
                print_status(
                    f"[{self.label}] YES:{yes_bid} NO:{no_bid} | "
                    f"secs:{int(secs_to_close)} | open:{open_count}"
                )
            
            context = {"secs_to_close": secs_to_close}
            
            # Process each strategy
            for strategy in self.strategies:
                # Check exits first
                for pos in list(strategy.positions):
                    should_exit, exit_type, exit_price, reason = strategy.evaluate_exit(
                        pos, prices, int(secs_to_close), context
                    )
                    if should_exit:
                        closed = self._execute_exit(strategy, pos, exit_type, exit_price, reason)
                        if closed:
                            self._log(prices, strategy.name, f"exit_{exit_type}",
                                     pos.side, exit_price, pos.qty, reason,
                                     closed.id, closed.net_pnl)
                
                # Check entry
                can_enter, _ = strategy.can_enter()
                if can_enter:
                    should_enter, side, price, qty, reason = strategy.evaluate_entry(
                        prices, int(secs_to_close), context
                    )
                    if should_enter:
                        pos = self._execute_entry(strategy, side, price, qty, reason)
                        if pos:
                            self._log(prices, strategy.name, "entry",
                                     side, price, qty, reason, pos.id)
            
            time.sleep(POLL_INTERVAL_SECS)
        
        # Summary
        print_status(f"[{self.label}] === SUMMARY ===")
        for strategy in self.strategies:
            stats = strategy.get_stats()
            print_status(
                f"[{self.label}][{strategy.name}] "
                f"Trades:{stats['trades']} | "
                f"Locks:{stats.get('locks',0)} | "
                f"Stops:{stats.get('stops',0)} | "
                f"Net:{stats['net_pnl']:.1f}c"
            )
        
        print_status(f"[{self.label}] Log: {self.log_path}")
        return {s.name: s.get_stats() for s in self.strategies}

# =============================================================================
# MARKET RESOLUTION
# =============================================================================

def find_market(private_key, team_name: str) -> Tuple[str, Dict]:
    markets = get_markets_in_series(private_key, SERIES_TICKER)
    
    team_l = team_name.lower()
    candidates = []
    
    for m in markets:
        ticker = (m.get("ticker") or "").upper()
        title = (m.get("title") or "").lower()
        
        if ticker.endswith(f"-{team_name.upper()}"):
            candidates.append(m)
        elif team_l in title:
            candidates.append(m)
    
    if not candidates:
        raise RuntimeError(f"No market for '{team_name}'")
    
    now = utc_now()
    future = [m for m in candidates if parse_iso(m["close_time"]) > now]
    if future:
        candidates = future
    
    candidates.sort(key=lambda m: parse_iso(m["close_time"]))
    return candidates[0]["ticker"], candidates[0]

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  MULTI-STRATEGY TESTING - February 5, 2026")
    print("  Testing 3 strategies in parallel per game:")
    print("    1. model_edge     - Your current approach (improved)")
    print("    2. mean_reversion - Pure price action, no model")
    print("    3. spread_capture - Market making lite")
    print("="*70 + "\n")
    
    # Load credentials
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    
    if not api_key or not key_path:
        print("ERROR: Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        return
    
    private_key = _load_private_key(key_path)
    
    # Check balance
    try:
        resp = _get(private_key, "/trade-api/v2/portfolio/balance")
        balance = int(resp.get("balance", 0)) / 100
        print_status(f"Account balance: ${balance:.2f}")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    total_per_game = sum(STRATEGY_ALLOCATIONS.values())
    
    print(f"\nPer-game allocation: ${total_per_game:.2f}")
    for name, alloc in STRATEGY_ALLOCATIONS.items():
        print(f"  {name}: ${alloc:.2f}")
    
    print("\nGAMES:")
    for game in GAMES:
        print(f"  {game['label']}: {game['team_name']} @ {game['model_p_win']:.0%}")
        print(f"    Tip: {game['tip_time']} | Segment: {game['segment']}")
    
    print("\n" + "="*70)
    input("Press ENTER to start (Ctrl+C to abort)...")
    print("="*70 + "\n")
    
    # Run games
    threads = []
    results = {}
    
    def run_game(game: Dict):
        label = game["label"]
        try:
            ticker, market = find_market(private_key, game["team_name"])
            print_status(f"[{label}] Found: {ticker}")
            
            strategies = [
                ModelEdgeStrategy(
                    STRATEGY_ALLOCATIONS["model_edge"],
                    int(game["model_p_win"] * 100)
                ),
                MeanReversionStrategy(
                    STRATEGY_ALLOCATIONS["mean_reversion"]
                ),
                SpreadCaptureStrategy(
                    STRATEGY_ALLOCATIONS["spread_capture"]
                ),
            ]
            
            runner = GameRunner(label, ticker, market, strategies, private_key)
            results[label] = runner.run()
            
        except Exception as e:
            print_status(f"[{label}] ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    for game in GAMES:
        t = threading.Thread(target=run_game, args=(game,), name=game["label"])
        t.start()
        threads.append(t)
        print_status(f"Started: {game['label']}")
    
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print_status("\nInterrupted")
    
    # Final results
    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    
    for game_label, game_results in results.items():
        print(f"\n{game_label}:")
        for strat_name, stats in game_results.items():
            print(f"  {strat_name}:")
            print(f"    Trades: {stats['trades']} | Locks: {stats.get('locks',0)} | Stops: {stats.get('stops',0)}")
            print(f"    Net P&L: {stats['net_pnl']:.1f}c (${stats['net_pnl']/100:.2f})")
    
    print("\n" + "="*70)
    print("Logs saved to ./logs/")
    print("Upload multistrat_*.csv files tomorrow for analysis!")


if __name__ == "__main__":
    main()