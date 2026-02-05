# production_multistrat.py
# Production multi-strategy testing with all bug fixes
#
# Key improvements:
# - Fixed SpreadCapture lock math
# - Liquidity checks before all entries
# - Proper fee rounding
# - ESPN clock integration
# - Position sizing based on capital
# - Centralized execution
# - Comprehensive logging

import os
import sys
import time
import json
import math
import threading
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import deque
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

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

from espn_game_clock import EspnGameClock

# =============================================================================
# CONFIGURATION
# =============================================================================

POLL_INTERVAL_SECS = 3.0
STOP_TRADING_BEFORE_CLOSE_SECS = 300
MIN_LIQUIDITY_CONTRACTS = 1

# =============================================================================
# FEE CALCULATIONS (WITH PROPER ROUNDING)
# =============================================================================

def calc_taker_fee_cents(price_cents: float, qty: int = 1) -> float:
    """Kalshi taker fee: 0.07 × C × P × (1-P), max 2¢ per contract"""
    price_cents = round(price_cents)  # FIXED: round instead of truncate
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    p = price_cents / 100.0
    fee_per = min(0.07 * p * (1 - p), 0.02)
    return fee_per * qty * 100

def calc_maker_fee_cents(price_cents: float, qty: int = 1) -> float:
    """Maker fee is 1/4 of taker"""
    return calc_taker_fee_cents(price_cents, qty) / 4

# =============================================================================
# LIQUIDITY CHECKING (CRITICAL FIX)
# =============================================================================

def _cum_qty_at_or_above(levels: List[List[int]], price_threshold: int) -> int:
    """Sum quantity for all price levels >= threshold"""
    cum = 0
    for p, q in levels:
        if int(p) >= int(price_threshold):
            cum += int(q)
        else:
            break
    return cum

def has_fill_liquidity_for_implied_buy(
    prices: Dict[str, Any], 
    side_to_buy: str, 
    buy_price_c: int, 
    min_qty: int
) -> bool:
    """
    CRITICAL: For implied ask buy, check OPPOSITE book liquidity.
    
    - Buying YES at price p fills against NO bids >= (100 - p)
    - Buying NO at price q fills against YES bids >= (100 - q)
    """
    yes_levels = prices.get("yes_levels", [])
    no_levels = prices.get("no_levels", [])
    
    if side_to_buy == "yes":
        needed_no_bid = 100 - int(buy_price_c)
        return _cum_qty_at_or_above(no_levels, needed_no_bid) >= min_qty
    else:
        needed_yes_bid = 100 - int(buy_price_c)
        return _cum_qty_at_or_above(yes_levels, needed_yes_bid) >= min_qty

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    """Open position tracking"""
    id: str
    strategy: str
    side: str
    entry_price: float
    qty: int
    entry_time: dt.datetime
    entry_reason: str
    entry_fee: float = 0.0
    entry_edge: float = 0.0
    order_id: str = ""

@dataclass
class ClosedPosition:
    """Completed trade record"""
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
    lock_side: Optional[str] = None
    entry_edge: float = 0.0
    exit_reason: str = ""

@dataclass
class TradeEvent:
    """Individual trade action for detailed logging"""
    timestamp: str
    strategy: str
    action: str  # "entry_attempt", "entry_fill", "exit_attempt", "exit_fill", "cancel"
    side: str
    price: float
    qty: int
    filled_qty: int = 0
    vwap: Optional[float] = None
    fee_cents: float = 0.0
    reason: str = ""
    order_id: str = ""
    position_id: str = ""
    liquidity_ok: bool = True
    was_maker_attempt: bool = False

# =============================================================================
# CENTRALIZED EXECUTION MODULE
# =============================================================================

class OrderExecutor:
    """Centralized order execution with proper checks and logging"""
    
    def __init__(self, private_key, ticker: str, game_label: str):
        self.private_key = private_key
        self.ticker = ticker
        self.game_label = game_label
        self.trade_events: List[TradeEvent] = []
    
    def execute_entry(
        self,
        strategy_name: str,
        side: str,
        price: int,
        qty: int,
        prices: Dict[str, Any],
        reason: str,
        position_id: str
    ) -> Tuple[int, Optional[float], float]:
        """
        Execute entry order with liquidity check.
        
        Returns: (filled_qty, vwap, fee_cents)
        """
        # CRITICAL: Check liquidity before attempting
        has_liquidity = has_fill_liquidity_for_implied_buy(
            prices, side, price, max(MIN_LIQUIDITY_CONTRACTS, qty)
        )
        
        event = TradeEvent(
            timestamp=utc_now().isoformat(),
            strategy=strategy_name,
            action="entry_attempt",
            side=side,
            price=price,
            qty=qty,
            reason=reason,
            position_id=position_id,
            liquidity_ok=has_liquidity,
        )
        
        if not has_liquidity:
            print_status(
                f"[{self.game_label}][{strategy_name}] SKIP ENTRY: "
                f"{side.upper()} @{price}¢ - insufficient opposite-book liquidity"
            )
            self.trade_events.append(event)
            return 0, None, 0.0
        
        print_status(
            f"[{self.game_label}][{strategy_name}] ENTRY: "
            f"BUY {side.upper()} {qty}x @{price}¢ - {reason}"
        )
        
        try:
            order_id = place_limit_buy(self.private_key, self.ticker, side, price, qty)
            event.order_id = order_id
            
            filled, vwap = wait_for_fill_or_timeout(
                self.private_key, order_id, side, max_wait_secs=15, poll_secs=2
            )
            
            if filled > 0:
                actual_price = vwap if vwap else float(price)
                fee = calc_taker_fee_cents(actual_price, filled)
                
                fill_event = TradeEvent(
                    timestamp=utc_now().isoformat(),
                    strategy=strategy_name,
                    action="entry_fill",
                    side=side,
                    price=price,
                    qty=qty,
                    filled_qty=filled,
                    vwap=actual_price,
                    fee_cents=fee,
                    reason=reason,
                    order_id=order_id,
                    position_id=position_id,
                )
                self.trade_events.append(fill_event)
                
                print_status(
                    f"[{self.game_label}][{strategy_name}] FILLED: "
                    f"{filled}x @{actual_price:.2f}¢ (fee: {fee:.2f}¢)"
                )
                
                return filled, actual_price, fee
            else:
                print_status(
                    f"[{self.game_label}][{strategy_name}] NO FILL (timeout)"
                )
                self.trade_events.append(event)
                return 0, None, 0.0
                
        except Exception as e:
            print_status(
                f"[{self.game_label}][{strategy_name}] ENTRY ERROR: {e}"
            )
            event.reason = f"error: {e}"
            self.trade_events.append(event)
            return 0, None, 0.0
    
    def execute_exit(
        self,
        strategy_name: str,
        position: Position,
        exit_type: str,
        exit_price: int,
        reason: str,
        prices: Dict[str, Any]
    ) -> Tuple[int, Optional[float], float, Optional[float], Optional[str]]:
        """
        Execute exit order (sell or lock).
        
        Returns: (filled_qty, exit_vwap, exit_fee, lock_price, lock_side)
        """
        print_status(
            f"[{self.game_label}][{strategy_name}] EXIT: "
            f"{position.side.upper()} @{exit_price}¢ ({exit_type}) - {reason}"
        )
        
        lock_price = None
        lock_side = None
        
        try:
            if exit_type == "lock":
                # Lock = buy the opposite side
                lock_side = "no" if position.side == "yes" else "yes"
                
                # Check liquidity for lock
                has_liquidity = has_fill_liquidity_for_implied_buy(
                    prices, lock_side, exit_price, position.qty
                )
                
                if not has_liquidity:
                    print_status(
                        f"[{self.game_label}][{strategy_name}] SKIP LOCK: "
                        f"insufficient liquidity for {lock_side.upper()} @{exit_price}¢"
                    )
                    return 0, None, 0.0, None, None
                
                order_id = place_limit_buy(
                    self.private_key, self.ticker, lock_side, exit_price, position.qty
                )
                
                filled, vwap = wait_for_fill_or_timeout(
                    self.private_key, order_id, lock_side, max_wait_secs=15, poll_secs=2
                )
                
                if filled > 0:
                    lock_price = vwap if vwap else float(exit_price)
                    fee = calc_taker_fee_cents(lock_price, filled)
                    
                    event = TradeEvent(
                        timestamp=utc_now().isoformat(),
                        strategy=strategy_name,
                        action="exit_fill",
                        side=lock_side,
                        price=exit_price,
                        qty=position.qty,
                        filled_qty=filled,
                        vwap=lock_price,
                        fee_cents=fee,
                        reason=f"lock_{reason}",
                        order_id=order_id,
                        position_id=position.id,
                    )
                    self.trade_events.append(event)
                    
                    return filled, lock_price, fee, lock_price, lock_side
                else:
                    print_status(
                        f"[{self.game_label}][{strategy_name}] LOCK NO FILL"
                    )
                    return 0, None, 0.0, None, None
            
            else:
                # Regular exit = sell current position
                order_id = place_limit_sell(
                    self.private_key, self.ticker, position.side, exit_price, position.qty
                )
                
                filled, vwap = wait_for_fill_or_timeout(
                    self.private_key, order_id, position.side, max_wait_secs=15, poll_secs=2
                )
                
                if filled > 0:
                    exit_vwap = vwap if vwap else float(exit_price)
                    fee = calc_taker_fee_cents(exit_vwap, filled)
                    
                    event = TradeEvent(
                        timestamp=utc_now().isoformat(),
                        strategy=strategy_name,
                        action="exit_fill",
                        side=position.side,
                        price=exit_price,
                        qty=position.qty,
                        filled_qty=filled,
                        vwap=exit_vwap,
                        fee_cents=fee,
                        reason=reason,
                        order_id=order_id,
                        position_id=position.id,
                    )
                    self.trade_events.append(event)
                    
                    return filled, exit_vwap, fee, None, None
                else:
                    print_status(
                        f"[{self.game_label}][{strategy_name}] EXIT NO FILL"
                    )
                    return 0, None, 0.0, None, None
                    
        except Exception as e:
            print_status(
                f"[{self.game_label}][{strategy_name}] EXIT ERROR: {e}"
            )
            return 0, None, 0.0, None, None

# =============================================================================
# BASE STRATEGY CLASS (WITH FIXES)
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
        if self.capital_used >= self.max_capital * 0.95:  # 95% buffer
            return False, f"capital:{self.capital_used:.2f}/{self.max_capital:.2f}"
        
        if len(self.positions) >= self.params.get("max_positions", 2):
            return False, f"max_positions:{len(self.positions)}"
        
        if self.last_entry_time:
            elapsed = time.time() - self.last_entry_time
            if elapsed < self.min_entry_gap_secs:
                return False, f"throttle:{elapsed:.0f}s<{self.min_entry_gap_secs}s"
        
        return True, "ok"
    
    def _size_position(self, edge_cents: float, price: int, side: str) -> int:
        """Size position based on edge and available capital"""
        # Calculate collateral needed per contract
        if side == "yes":
            collateral_per = (100 - price) / 100
        else:
            collateral_per = price / 100
        
        # Max affordable
        available_capital = self.max_capital - self.capital_used
        max_affordable = int(available_capital / collateral_per) if collateral_per > 0 else 0
        
        # Edge-based sizing
        if edge_cents < 8:
            base_qty = 1
        elif edge_cents < 12:
            base_qty = 2
        elif edge_cents < 18:
            base_qty = 3
        else:
            base_qty = 4
        
        # Cap at max
        return min(base_qty, max_affordable, 5)
    
    @abstractmethod
    def evaluate_entry(
        self,
        prices: Dict[str, Any],
        game_context: Dict[str, Any]
    ) -> Tuple[bool, str, int, int, str, float]:
        """
        Returns: (should_enter, side, price, qty, reason, edge)
        """
        pass
    
    @abstractmethod
    def evaluate_exit(
        self,
        position: Position,
        prices: Dict[str, Any],
        game_context: Dict[str, Any]
    ) -> Tuple[bool, str, int, str]:
        """
        Returns: (should_exit, exit_type, exit_price, reason)
        """
        pass
    
    def record_entry(
        self,
        side: str,
        price: float,
        qty: int,
        reason: str,
        fee: float,
        edge: float,
        order_id: str = ""
    ) -> Position:
        """Record new position"""
        pos = Position(
            id=self._next_position_id(),
            strategy=self.name,
            side=side,
            entry_price=price,
            qty=qty,
            entry_time=utc_now(),
            entry_reason=reason,
            entry_fee=fee,
            entry_edge=edge,
            order_id=order_id,
        )
        
        self.positions.append(pos)
        self.last_entry_time = time.time()
        self.total_fees += fee
        
        # Update capital
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
        exit_fee: float,
        reason: str,
        lock_price: Optional[float] = None,
        lock_side: Optional[str] = None
    ) -> ClosedPosition:
        """Record position exit"""
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
            lock_side=lock_side,
            entry_edge=position.entry_edge,
            exit_reason=reason,
        )
        
        self.closed.append(closed)
        self.positions.remove(position)
        self.total_fees += exit_fee
        
        # Free capital
        if position.side == "yes":
            self.capital_used -= (100 - position.entry_price) * position.qty / 100
        else:
            self.capital_used -= position.entry_price * position.qty / 100
        
        return closed
    
    def get_stats(self) -> Dict[str, Any]:
        """Performance statistics"""
        if not self.closed:
            return {
                "strategy": self.name,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "locks": 0,
                "stops": 0,
                "gross_pnl": 0.0,
                "fees": self.total_fees,
                "net_pnl": -self.total_fees,
                "open_positions": len(self.positions),
                "capital_used": self.capital_used,
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
            "win_rate": len(wins) / len(self.closed) if self.closed else 0,
            "locks": len(locks),
            "lock_rate": len(locks) / len(self.closed) if self.closed else 0,
            "stops": len(stops),
            "gross_pnl": gross,
            "fees": self.total_fees,
            "net_pnl": net,
            "avg_hold_secs": sum(c.hold_secs for c in self.closed) / len(self.closed),
            "avg_win": sum(c.net_pnl for c in wins) / len(wins) if wins else 0,
            "avg_loss": sum(c.net_pnl for c in losses) / len(losses) if losses else 0,
            "open_positions": len(self.positions),
            "capital_used": self.capital_used,
        }

# =============================================================================
# STRATEGY IMPLEMENTATIONS (WITH FIXES)
# =============================================================================

class ModelEdgeStrategy(BaseStrategy):
    """Model-based edge strategy with dynamic fair price"""
    
    def __init__(self, max_capital: float, model_fair_cents: int):
        params = {
            "min_entry_edge": 10,
            "min_lock_profit": 10,
            "stop_loss": 18,
            "take_profit": 12,
            "max_positions": 2,
            "min_entry_gap_secs": 45,
        }
        super().__init__("model_edge", max_capital, params)
        self.model_fair = model_fair_cents
        self.price_history: deque = deque(maxlen=30)
    
    def _get_dynamic_fair(
        self, 
        current_mid: float, 
        game_secs_remaining: Optional[int]
    ) -> float:
        """Blend model with market based on REAL game time"""
        self.price_history.append(current_mid)
        
        if game_secs_remaining is None:
            # No game clock - use moderate blend
            if len(self.price_history) < 5:
                return float(self.model_fair)
            market_avg = sum(self.price_history) / len(self.price_history)
            return 0.6 * self.model_fair + 0.4 * market_avg
        
        # Use actual game progress
        total_game_secs = 2400  # 40 minutes
        game_elapsed = max(0, total_game_secs - game_secs_remaining)
        game_progress = min(1.0, game_elapsed / total_game_secs)
        
        # Exponential decay of model confidence
        model_weight = math.exp(-2 * game_progress)
        market_avg = sum(self.price_history) / len(self.price_history)
        
        return model_weight * self.model_fair + (1 - model_weight) * market_avg
    
    def evaluate_entry(
        self,
        prices: Dict[str, Any],
        game_context: Dict[str, Any]
    ) -> Tuple[bool, str, int, int, str, float]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")
        
        if not all([yes_bid, yes_ask]):
            return False, "", 0, 0, "no_prices", 0.0
        
        mid = (yes_bid + yes_ask) / 2
        game_secs = game_context.get("game_secs_remaining")
        fair = self._get_dynamic_fair(mid, game_secs)
        
        # Check YES side
        yes_edge = fair - yes_ask
        if yes_edge >= self.params["min_entry_edge"]:
            qty = self._size_position(yes_edge, yes_ask, "yes")
            if qty > 0:
                return True, "yes", yes_ask, qty, f"edge:{yes_edge:.1f}¢,fair:{fair:.1f}¢", yes_edge
        
        # Check NO side
        if no_ask:
            no_fair = 100 - fair
            no_edge = no_fair - no_ask
            if no_edge >= self.params["min_entry_edge"]:
                qty = self._size_position(no_edge, no_ask, "no")
                if qty > 0:
                    return True, "no", no_ask, qty, f"edge:{no_edge:.1f}¢,fair:{fair:.1f}¢", no_edge
        
        return False, "", 0, 0, "no_edge", 0.0
    
    def evaluate_exit(
        self,
        position: Position,
        prices: Dict[str, Any],
        game_context: Dict[str, Any]
    ) -> Tuple[bool, str, int, str]:
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        no_ask = prices.get("imp_no_ask")
        yes_ask = prices.get("imp_yes_ask")
        
        if position.side == "yes":
            # Try to lock first (FIXED: use correct imp_no_ask)
            if no_ask:
                profit = 100 - position.entry_price - no_ask
                if profit >= self.params["min_lock_profit"]:
                    return True, "lock", no_ask, f"lock:{profit:.0f}¢"
            
            # Stop/take profit
            if yes_bid:
                pnl = yes_bid - position.entry_price
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", yes_bid - 1, f"stop:{pnl:.0f}¢"
                if pnl >= self.params["take_profit"]:
                    return True, "take_profit", yes_bid - 1, f"tp:{pnl:.0f}¢"
        
        else:  # NO position
            # Try to lock (FIXED: use correct imp_yes_ask)
            if yes_ask:
                profit = 100 - position.entry_price - yes_ask
                if profit >= self.params["min_lock_profit"]:
                    return True, "lock", yes_ask, f"lock:{profit:.0f}¢"
            
            # Stop/take profit
            if no_bid:
                pnl = no_bid - position.entry_price
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", no_bid - 1, f"stop:{pnl:.0f}¢"
                if pnl >= self.params["take_profit"]:
                    return True, "take_profit", no_bid - 1, f"tp:{pnl:.0f}¢"
        
        return False, "", 0, "hold"


class MeanReversionStrategy(BaseStrategy):
    """Pure price-based mean reversion"""
    
    def __init__(self, max_capital: float):
        params = {
            "lookback": 60,  # FIXED: Increased from 20
            "entry_std_mult": 1.5,
            "stop_loss": 15,
            "max_positions": 2,
            "min_entry_gap_secs": 30,
        }
        super().__init__("mean_reversion", max_capital, params)
        self.prices: deque = deque(maxlen=params["lookback"])
    
    def _calc_stats(self) -> Tuple[float, float]:
        if len(self.prices) < 10:
            return 50.0, 10.0
        
        prices = list(self.prices)
        mean = sum(prices) / len(prices)
        var = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = math.sqrt(var) if var > 0 else 5.0
        
        return mean, std
    
    def evaluate_entry(
        self,
        prices: Dict[str, Any],
        game_context: Dict[str, Any]
    ) -> Tuple[bool, str, int, int, str, float]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")
        
        # FIXED: Skip early game volatility
        game_secs = game_context.get("game_secs_remaining")
        if game_secs and game_secs > 2200:
            return False, "", 0, 0, "game_too_early", 0.0
        
        if not yes_bid or not yes_ask:
            return False, "", 0, 0, "no_prices", 0.0
        
        mid = (yes_bid + yes_ask) / 2
        self.prices.append(mid)
        
        if len(self.prices) < 20:  # Need 20 samples minimum
            return False, "", 0, 0, f"warming:{len(self.prices)}/20", 0.0
        
        mean, std = self._calc_stats()
        threshold = self.params["entry_std_mult"] * std
        
        # Price dropped - buy YES
        if mid < mean - threshold:
            edge = mean - mid
            # Fee-aware: need enough edge to overcome fees
            min_edge = 8 + calc_taker_fee_cents(yes_ask, 1) * 2
            if edge >= min_edge:
                qty = self._size_position(edge, yes_ask, "yes")
                if qty > 0:
                    return True, "yes", yes_ask, qty, f"below:{edge:.0f}¢({edge/std:.1f}σ)", edge
        
        # Price spiked - buy NO
        if mid > mean + threshold and no_ask:
            edge = mid - mean
            min_edge = 8 + calc_taker_fee_cents(no_ask, 1) * 2
            if edge >= min_edge:
                qty = self._size_position(edge, no_ask, "no")
                if qty > 0:
                    return True, "no", no_ask, qty, f"above:{edge:.0f}¢({edge/std:.1f}σ)", edge
        
        return False, "", 0, 0, "in_range", 0.0
    
    def evaluate_exit(
        self,
        position: Position,
        prices: Dict[str, Any],
        game_context: Dict[str, Any]
    ) -> Tuple[bool, str, int, str]:
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
                return True, "take_profit", yes_bid - 1, f"reverted:{pnl:.0f}¢"
            
            if pnl <= -self.params["stop_loss"]:
                return True, "stop", yes_bid - 1, f"stop:{pnl:.0f}¢"
        
        else:
            if no_bid:
                pnl = no_bid - position.entry_price
                
                if mid <= mean + 2:
                    return True, "take_profit", no_bid - 1, f"reverted:{pnl:.0f}¢"
                
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", no_bid - 1, f"stop:{pnl:.0f}¢"
        
        return False, "", 0, "hold"


class SpreadCaptureStrategy(BaseStrategy):
    """Spread capture / market making lite"""
    
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
    
    def evaluate_entry(
        self,
        prices: Dict[str, Any],
        game_context: Dict[str, Any]
    ) -> Tuple[bool, str, int, int, str, float]:
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_bid = prices.get("best_no_bid")
        
        if not all([yes_bid, yes_ask, no_bid]):
            return False, "", 0, 0, "no_prices", 0.0
        
        spread = yes_ask - yes_bid
        
        if spread < self.params["min_spread"]:
            return False, "", 0, 0, f"spread:{spread}¢", 0.0
        
        # Try to improve YES bid
        our_bid = yes_bid + self.params["improve_cents"]
        
        # FIXED: Correct lock calculation using imp_no_ask
        imp_no_ask = 100 - yes_bid  # This is the price to lock NO
        potential_lock_cost = our_bid + imp_no_ask
        potential_profit = 100 - potential_lock_cost
        
        if potential_profit >= self.params["min_lock_profit"]:
            qty = self._size_position(potential_profit, our_bid, "yes")
            if qty > 0:
                return True, "yes", our_bid, qty, \
                    f"spread:{spread}¢,pot_lock:{potential_profit:.0f}¢", potential_profit
        
        return False, "", 0, 0, "no_opportunity", 0.0
    
    def evaluate_exit(
        self,
        position: Position,
        prices: Dict[str, Any],
        game_context: Dict[str, Any]
    ) -> Tuple[bool, str, int, str]:
        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_ask = prices.get("imp_no_ask")
        
        if position.side == "yes":
            # Try to lock (FIXED: use imp_no_ask)
            if no_ask:
                profit = 100 - position.entry_price - no_ask
                if profit >= self.params["min_lock_profit"]:
                    return True, "lock", no_ask, f"lock:{profit:.0f}¢"
            
            # Stop loss
            if yes_bid:
                pnl = yes_bid - position.entry_price
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", yes_bid - 1, f"stop:{pnl:.0f}¢"
        
        else:
            # Try to lock (FIXED: use imp_yes_ask)
            if yes_ask:
                profit = 100 - position.entry_price - yes_ask
                if profit >= self.params["min_lock_profit"]:
                    return True, "lock", yes_ask, f"lock:{profit:.0f}¢"
            
            # Stop loss
            if no_bid:
                pnl = no_bid - position.entry_price
                if pnl <= -self.params["stop_loss"]:
                    return True, "stop", no_bid - 1, f"stop:{pnl:.0f}¢"
        
        return False, "", 0, "waiting"

# Continue in next file...
# production_multistrat_part2.py
# Continuation: GameRunner, Logging, and Data Collection

# =============================================================================
# COMPREHENSIVE DATA LOGGER
# =============================================================================

class ComprehensiveLogger:
    """Multi-file logging system for complete data capture"""
    
    def __init__(self, game_label: str, ticker: str):
        self.game_label = game_label
        self.ticker = ticker
        
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        ts = utc_now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{game_label}_{ts}"
        
        # Multiple log files for different purposes
        self.snapshots_path = f"logs/snapshots_{self.session_id}.csv"
        self.trades_path = f"logs/trades_{self.session_id}.csv"
        self.positions_path = f"logs/positions_{self.session_id}.csv"
        self.events_path = f"logs/events_{self.session_id}.csv"
        self.summary_path = f"logs/summary_{self.session_id}.json"
        
        self._init_snapshots()
        self._init_trades()
        self._init_positions()
        self._init_events()
        
        self.tick_count = 0
    
    def _init_snapshots(self):
        """Market snapshot log (every tick)"""
        with open(self.snapshots_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "tick", "ticker",
                "yes_bid", "yes_ask", "yes_imp_ask",
                "no_bid", "no_ask", "no_imp_ask",
                "mid", "spread", 
                "yes_depth_1", "yes_depth_2", "yes_depth_3",
                "no_depth_1", "no_depth_2", "no_depth_3",
                "secs_to_kalshi_close", "secs_to_game_end", "clock_source",
            ])
    
    def _init_trades(self):
        """Individual trade attempts and fills"""
        with open(self.trades_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "strategy", "action", "side", "price", "qty",
                "filled_qty", "vwap", "fee_cents", "reason",
                "order_id", "position_id", "liquidity_ok", "was_maker"
            ])
    
    def _init_positions(self):
        """Complete position lifecycle"""
        with open(self.positions_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "position_id", "strategy", "side",
                "entry_time", "entry_price", "entry_qty", "entry_fee", "entry_edge", "entry_reason",
                "exit_time", "exit_type", "exit_price", "exit_fee", "exit_reason",
                "lock_price", "lock_side",
                "gross_pnl", "net_pnl", "hold_secs",
            ])
    
    def _init_events(self):
        """Strategy decision events (entries, exits, skips)"""
        with open(self.events_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "tick", "strategy", "event_type", "decision",
                "side", "price", "qty", "reason", "position_id"
            ])
    
    def log_snapshot(
        self,
        prices: Dict[str, Any],
        secs_to_kalshi: int,
        game_secs: Optional[int],
        clock_source: str
    ):
        """Log market snapshot"""
        self.tick_count += 1
        
        yes_levels = prices.get("yes_levels", [])
        no_levels = prices.get("no_levels", [])
        
        yes_depths = [q for p, q in yes_levels[:3]] + [0, 0, 0]
        no_depths = [q for p, q in no_levels[:3]] + [0, 0, 0]
        
        yes_bid = prices.get("best_yes_bid", "")
        yes_ask = prices.get("best_yes_ask", "")
        yes_imp_ask = prices.get("imp_yes_ask", "")
        no_bid = prices.get("best_no_bid", "")
        no_ask = prices.get("best_no_ask", "")
        no_imp_ask = prices.get("imp_no_ask", "")
        
        mid = ""
        spread = ""
        if yes_bid and yes_imp_ask:
            mid = (yes_bid + yes_imp_ask) / 2
            spread = yes_imp_ask - yes_bid
        
        with open(self.snapshots_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                utc_now().isoformat(), self.tick_count, self.ticker,
                yes_bid, yes_ask, yes_imp_ask,
                no_bid, no_ask, no_imp_ask,
                f"{mid:.2f}" if mid else "", spread,
                yes_depths[0], yes_depths[1], yes_depths[2],
                no_depths[0], no_depths[1], no_depths[2],
                secs_to_kalshi, game_secs or "", clock_source,
            ])
    
    def log_trade_event(self, event: TradeEvent):
        """Log individual trade attempt/fill"""
        with open(self.trades_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                event.timestamp, event.strategy, event.action, event.side,
                event.price, event.qty, event.filled_qty,
                f"{event.vwap:.2f}" if event.vwap else "",
                f"{event.fee_cents:.2f}",
                event.reason, event.order_id, event.position_id,
                event.liquidity_ok, event.was_maker_attempt,
            ])
    
    def log_position(self, pos: ClosedPosition):
        """Log completed position"""
        with open(self.positions_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                pos.id, pos.strategy, pos.side,
                pos.entry_time, f"{pos.entry_price:.2f}", pos.qty,
                f"{pos.entry_fee:.2f}", f"{pos.entry_edge:.2f}", "",
                pos.exit_time, pos.exit_type, f"{pos.exit_price:.2f}",
                f"{pos.exit_fee:.2f}", pos.exit_reason,
                f"{pos.lock_price:.2f}" if pos.lock_price else "",
                pos.lock_side or "",
                f"{pos.gross_pnl:.2f}", f"{pos.net_pnl:.2f}", pos.hold_secs,
            ])
    
    def log_event(
        self,
        strategy: str,
        event_type: str,
        decision: str,
        side: str = "",
        price: int = 0,
        qty: int = 0,
        reason: str = "",
        position_id: str = ""
    ):
        """Log strategy decision event"""
        with open(self.events_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                utc_now().isoformat(), self.tick_count, strategy,
                event_type, decision, side, price, qty, reason, position_id
            ])
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save final summary as JSON"""
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    
    def get_paths(self) -> Dict[str, str]:
        """Return all log file paths"""
        return {
            "snapshots": self.snapshots_path,
            "trades": self.trades_path,
            "positions": self.positions_path,
            "events": self.events_path,
            "summary": self.summary_path,
        }

# =============================================================================
# GAME RUNNER (WITH ESPN CLOCK)
# =============================================================================

class GameRunner:
    """Runs multiple strategies on one game with full logging"""
    
    def __init__(
        self,
        game_label: str,
        ticker: str,
        market: Dict[str, Any],
        strategies: List[BaseStrategy],
        private_key,
        espn_clock: Optional[EspnGameClock] = None,
    ):
        self.label = game_label
        self.ticker = ticker
        self.market = market
        self.strategies = strategies
        self.private_key = private_key
        self.espn_clock = espn_clock
        
        self.close_time = parse_iso(market["close_time"])
        self.executor = OrderExecutor(private_key, ticker, game_label)
        self.logger = ComprehensiveLogger(game_label, ticker)
        
        self.start_time = utc_now()
        self.last_status_tick = 0
    
    def _get_game_context(self) -> Dict[str, Any]:
        """Build game context including real-time clock info"""
        now = utc_now()
        kalshi_secs = (self.close_time - now).total_seconds()
        
        game_secs = None
        clock_source = "kalshi_only"
        
        if self.espn_clock:
            try:
                game_secs, status = self.espn_clock.get_secs_to_game_end()
                if game_secs is not None:
                    clock_source = f"espn:{status}"
            except Exception as e:
                print_status(f"[{self.label}] ESPN clock error: {e}")
        
        # ESPN is source of truth when available
        if game_secs is not None:
            secs_to_close = float(game_secs)
            is_espn_final = (game_secs == 0 and "final" in status.lower())
        else:
            secs_to_close = kalshi_secs
            is_espn_final = False
        
           
        
        return {
            "secs_to_close": int(secs_to_close),
            "kalshi_secs_remaining": int(kalshi_secs),
            "game_secs_remaining": game_secs,
            "clock_source": clock_source,
            "is_espn_final": is_espn_final,
        }
    
    def _print_status(self, prices: Dict[str, Any], context: Dict[str, Any]):
        """Print periodic status update"""
        yes_bid = prices.get("best_yes_bid", "?")
        yes_ask = prices.get("imp_yes_ask", "?")
        no_bid = prices.get("best_no_bid", "?")
        no_ask = prices.get("imp_no_ask", "?")
        
        open_count = sum(len(s.positions) for s in self.strategies)
        total_net_pnl = sum(s.get_stats()["net_pnl"] for s in self.strategies)
        
        clock_info = f"K:{context['kalshi_secs_remaining']}"
        if context['game_secs_remaining']:
            clock_info += f"|G:{context['game_secs_remaining']}"
        
        print_status(
            f"[{self.label}] YES:{yes_bid}/{yes_ask} NO:{no_bid}/{no_ask} | "
            f"Open:{open_count} | NetPnL:{total_net_pnl:.1f}¢ | "
            f"Clocks:{clock_info} | Src:{context['clock_source']}"
        )
    
    def run(self) -> Dict[str, Any]:
        """Main trading loop"""
        print_status(f"\n{'='*70}")
        print_status(f"[{self.label}] STARTING")
        print_status(f"  Ticker: {self.ticker}")
        print_status(f"  Market: {self.market.get('title', 'N/A')}")
        print_status(f"  Close: {self.close_time}")
        print_status(f"  Strategies: {[s.name for s in self.strategies]}")
        if self.espn_clock:
            print_status(f"  ESPN Clock: ENABLED")
        print_status(f"{'='*70}\n")
        
        while True:
            context = self._get_game_context()
            
            # Check if we should stop
            if context.get("is_espn_final", False):
                print_status(f"[{self.label}] Game final per ESPN - stopping")
                self._close_all_positions()
                break
            
            if context["secs_to_close"] <= 0:
                print_status(f"[{self.label}] Market closed (time)")
                break
            
            # if context["secs_to_close"] <= STOP_TRADING_BEFORE_CLOSE_SECS:
            #     print_status(
            #         f"[{self.label}] Near close ({context['secs_to_close']}s) - "
            #         f"stopping new trades"
            #     )
            #     # Close any remaining positions
            #     self._close_all_positions()
            #     break
            
            # Fetch orderbook
            try:
                ob = fetch_orderbook(self.ticker)
                prices = derive_prices(ob)
            except Exception as e:
                print_status(f"[{self.label}] Orderbook error: {e}")
                time.sleep(POLL_INTERVAL_SECS)
                continue
            
            # Log snapshot
            self.logger.log_snapshot(
                prices,
                context["kalshi_secs_remaining"],
                context["game_secs_remaining"],
                context["clock_source"]
            )
            
            # Status every 20 ticks (~1 minute)
            if self.logger.tick_count - self.last_status_tick >= 20:
                self._print_status(prices, context)
                self.last_status_tick = self.logger.tick_count
            
            # Process each strategy
            for strategy in self.strategies:
                self._process_strategy(strategy, prices, context)
            
            # Add jitter to avoid synchronized API calls
            import random
            jitter = random.uniform(0, 0.7)
            time.sleep(POLL_INTERVAL_SECS + jitter)
        
        # Final summary
        return self._generate_summary()
    
    def _process_strategy(
        self,
        strategy: BaseStrategy,
        prices: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Process one strategy's decisions"""
        
        # Check exits first
        for pos in list(strategy.positions):
            should_exit, exit_type, exit_price, reason = strategy.evaluate_exit(
                pos, prices, context
            )
            
            if should_exit:
                self.logger.log_event(
                    strategy.name, "exit_eval", "yes",
                    pos.side, exit_price, pos.qty, reason, pos.id
                )
                
                filled, vwap, fee, lock_price, lock_side = self.executor.execute_exit(
                    strategy.name, pos, exit_type, exit_price, reason, prices
                )
                
                if filled > 0:
                    actual_exit_price = vwap if vwap else float(exit_price)
                    closed = strategy.record_exit(
                        pos, exit_type, actual_exit_price, fee,
                        reason, lock_price, lock_side
                    )
                    self.logger.log_position(closed)
                    
                    print_status(
                        f"[{self.label}][{strategy.name}] CLOSED {pos.id}: "
                        f"{closed.exit_type} | Net: {closed.net_pnl:.1f}¢"
                    )
            else:
                if self.logger.tick_count % 60 == 0:  # Log holds occasionally
                    self.logger.log_event(
                        strategy.name, "exit_eval", "hold",
                        pos.side, 0, pos.qty, reason, pos.id
                    )
        
        # Check entry
        can_enter, throttle_reason = strategy.can_enter()
        
        if not can_enter:
            if self.logger.tick_count % 60 == 0:
                self.logger.log_event(
                    strategy.name, "entry_eval", "throttled",
                    "", 0, 0, throttle_reason
                )
            return
        
        should_enter, side, price, qty, reason, edge = strategy.evaluate_entry(
            prices, context
        )
        
        if should_enter:
            position_id = f"{strategy.name}_{strategy.position_counter + 1}"
            
            self.logger.log_event(
                strategy.name, "entry_eval", "yes",
                side, price, qty, reason, position_id
            )
            
            filled, vwap, fee = self.executor.execute_entry(
                strategy.name, side, price, qty, prices, reason, position_id
            )
            
            if filled > 0:
                actual_price = vwap if vwap else float(price)
                pos = strategy.record_entry(
                    side, actual_price, filled, reason, fee, edge
                )
                
                print_status(
                    f"[{self.label}][{strategy.name}] OPENED {pos.id}: "
                    f"{side.upper()} {filled}x @{actual_price:.2f}¢ | "
                    f"Edge: {edge:.1f}¢"
                )
        else:
            if self.logger.tick_count % 60 == 0:
                self.logger.log_event(
                    strategy.name, "entry_eval", "no",
                    "", 0, 0, reason
                )
    
    def _close_all_positions(self):
        """Emergency close all open positions"""
        print_status(f"[{self.label}] Closing all open positions...")
        
        try:
            ob = fetch_orderbook(self.ticker)
            prices = derive_prices(ob)
        except:
            print_status(f"[{self.label}] Cannot fetch prices for emergency close")
            return
        
        for strategy in self.strategies:
            for pos in list(strategy.positions):
                yes_bid = prices.get("best_yes_bid")
                no_bid = prices.get("best_no_bid")
                
                if pos.side == "yes" and yes_bid:
                    exit_price = max(1, yes_bid - 2)
                elif pos.side == "no" and no_bid:
                    exit_price = max(1, no_bid - 2)
                else:
                    continue
                
                filled, vwap, fee, _, _ = self.executor.execute_exit(
                    strategy.name, pos, "timeout", exit_price,
                    "market_closing", prices
                )
                
                if filled > 0:
                    actual_price = vwap if vwap else float(exit_price)
                    closed = strategy.record_exit(
                        pos, "timeout", actual_price, fee, "market_closing"
                    )
                    self.logger.log_position(closed)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate final summary"""
        print_status(f"\n{'='*70}")
        print_status(f"[{self.label}] FINAL SUMMARY")
        print_status(f"{'='*70}")
        
        strategy_results = {}
        
        for strategy in self.strategies:
            stats = strategy.get_stats()
            strategy_results[strategy.name] = stats
            
            print_status(f"\n[{strategy.name}]")
            print_status(f"  Trades: {stats['trades']} ({stats['wins']}W / {stats['losses']}L)")
            print_status(f"  Win Rate: {stats['win_rate']:.1%}")
            print_status(f"  Locks: {stats['locks']} ({stats['lock_rate']:.1%})")
            print_status(f"  Stops: {stats['stops']}")
            print_status(f"  Gross P&L: {stats['gross_pnl']:.1f}¢ (${stats['gross_pnl']/100:.2f})")
            print_status(f"  Fees: {stats['fees']:.1f}¢ (${stats['fees']/100:.2f})")
            print_status(f"  Net P&L: {stats['net_pnl']:.1f}¢ (${stats['net_pnl']/100:.2f})")
            if stats['trades'] > 0:
                print_status(f"  Avg Win: {stats['avg_win']:.1f}¢")
                print_status(f"  Avg Loss: {stats['avg_loss']:.1f}¢")
                print_status(f"  Avg Hold: {stats['avg_hold_secs']/60:.1f} min")
        
        # Log all trade events
        for event in self.executor.trade_events:
            self.logger.log_trade_event(event)
        
        summary = {
            "game_label": self.label,
            "ticker": self.ticker,
            "market_title": self.market.get("title", ""),
            "start_time": self.start_time.isoformat(),
            "end_time": utc_now().isoformat(),
            "duration_secs": int((utc_now() - self.start_time).total_seconds()),
            "strategies": strategy_results,
            "log_files": self.logger.get_paths(),
        }
        
        self.logger.save_summary(summary)
        
        print_status(f"\nLog files:")
        for name, path in self.logger.get_paths().items():
            print_status(f"  {name}: {path}")
        
        print_status(f"{'='*70}\n")
        
        return summary

# Export for use in other files
__all__ = [
    'BaseStrategy',
    'ModelEdgeStrategy',
    'MeanReversionStrategy',
    'SpreadCaptureStrategy',
    'GameRunner',
    'OrderExecutor',
    'ComprehensiveLogger',
]
