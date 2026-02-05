# kalshi_strategy_lab.py
# Multi-strategy testing framework for Kalshi live basketball trading
#
# This module provides:
# 1. Enhanced data collection (captures everything needed for analysis)
# 2. Multiple strategy implementations that can run in parallel
# 3. Performance comparison tools
#
# Philosophy: Test multiple strategies with small allocations, let data decide

import os
import time
import json
import csv
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import deque
from enum import Enum
import datetime as dt

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StrategyConfig:
    """Configuration for a single strategy instance"""
    name: str
    enabled: bool = True
    max_collateral_dollars: float = 5.0
    max_position_qty: int = 5
    max_open_positions: int = 2
    
    # Entry parameters
    min_entry_edge_cents: int = 8
    min_spread_cents: int = 2
    max_spread_cents: int = 10
    
    # Exit parameters
    stop_loss_cents: int = 15
    take_profit_cents: int = 10
    min_lock_profit_cents: int = 8
    
    # Timing
    entry_timeout_secs: int = 15
    min_secs_between_entries: int = 30
    stop_trading_before_close_secs: int = 300


# =============================================================================
# ENHANCED DATA COLLECTION
# =============================================================================

@dataclass
class MarketSnapshot:
    """Complete snapshot of market state at a point in time"""
    timestamp: str
    ticker: str
    
    # Best prices
    yes_best_bid: Optional[int]
    yes_best_ask: Optional[int]  # Direct ask if available
    yes_imp_ask: Optional[int]   # Implied from NO bid
    no_best_bid: Optional[int]
    no_best_ask: Optional[int]
    no_imp_ask: Optional[int]
    
    # Depth (top 3 levels)
    yes_bid_depth: List[Tuple[int, int]] = field(default_factory=list)  # [(price, qty), ...]
    no_bid_depth: List[Tuple[int, int]] = field(default_factory=list)
    
    # Derived
    midpoint: Optional[float] = None
    spread_cents: Optional[int] = None
    
    # Timing
    secs_to_close: int = 0
    clock_source: str = ""
    
    def __post_init__(self):
        # Calculate derived fields
        if self.yes_best_bid is not None and self.yes_imp_ask is not None:
            self.midpoint = (self.yes_best_bid + self.yes_imp_ask) / 2
            self.spread_cents = self.yes_imp_ask - self.yes_best_bid


@dataclass
class TradeEvent:
    """Record of a trade execution"""
    timestamp: str
    ticker: str
    strategy: str
    
    # Trade details
    action: str  # "buy", "sell"
    side: str    # "yes", "no"
    intended_price: int
    fill_price: float
    qty_intended: int
    qty_filled: int
    
    # Context
    reason: str  # Why this trade was made
    edge_at_entry: float = 0.0
    fair_price_estimate: float = 0.0
    volatility_estimate: float = 0.0
    secs_to_close: int = 0
    
    # Fees
    estimated_fee_cents: float = 0.0
    is_maker: bool = False
    
    # Linking
    order_id: str = ""
    position_id: str = ""  # Links entries to exits


@dataclass  
class PositionRecord:
    """Complete lifecycle of a position"""
    position_id: str
    ticker: str
    strategy: str
    
    # Entry
    entry_time: str
    entry_side: str
    entry_price: float
    entry_qty: int
    entry_edge: float
    entry_fee: float
    
    # Exit (filled when closed)
    exit_time: Optional[str] = None
    exit_type: Optional[str] = None  # "lock", "stop", "take_profit", "timeout", "manual"
    exit_price: Optional[float] = None
    exit_qty: Optional[int] = None
    exit_fee: Optional[float] = None
    
    # For locked pairs
    lock_price: Optional[float] = None
    lock_side: Optional[str] = None
    
    # P&L
    gross_pnl_cents: Optional[float] = None
    total_fees_cents: Optional[float] = None
    net_pnl_cents: Optional[float] = None
    
    # Duration
    hold_time_secs: Optional[int] = None


class DataCollector:
    """
    Enhanced data collector that captures everything needed for analysis.
    Saves to multiple files for different analysis purposes.
    """
    
    def __init__(self, game_label: str, output_dir: str = "data"):
        self.game_label = game_label
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{game_label}_{ts}"
        
        # File paths
        self.snapshots_path = os.path.join(output_dir, f"snapshots_{self.session_id}.csv")
        self.trades_path = os.path.join(output_dir, f"trades_{self.session_id}.csv")
        self.positions_path = os.path.join(output_dir, f"positions_{self.session_id}.csv")
        
        # Initialize files with headers
        self._init_snapshots_file()
        self._init_trades_file()
        self._init_positions_file()
        
        # In-memory buffers for efficiency
        self.snapshot_buffer: List[MarketSnapshot] = []
        self.buffer_size = 50
        
        # Thread safety
        self.lock = threading.Lock()
        
    def _init_snapshots_file(self):
        with open(self.snapshots_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "ticker",
                "yes_best_bid", "yes_imp_ask", "no_best_bid", "no_imp_ask",
                "midpoint", "spread_cents",
                "yes_depth_1_qty", "yes_depth_2_qty", "yes_depth_3_qty",
                "no_depth_1_qty", "no_depth_2_qty", "no_depth_3_qty",
                "secs_to_close", "clock_source"
            ])
    
    def _init_trades_file(self):
        with open(self.trades_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "ticker", "strategy",
                "action", "side", "intended_price", "fill_price",
                "qty_intended", "qty_filled",
                "reason", "edge_at_entry", "fair_price_estimate",
                "volatility_estimate", "secs_to_close",
                "estimated_fee_cents", "is_maker",
                "order_id", "position_id"
            ])
    
    def _init_positions_file(self):
        with open(self.positions_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "position_id", "ticker", "strategy",
                "entry_time", "entry_side", "entry_price", "entry_qty",
                "entry_edge", "entry_fee",
                "exit_time", "exit_type", "exit_price", "exit_qty", "exit_fee",
                "lock_price", "lock_side",
                "gross_pnl_cents", "total_fees_cents", "net_pnl_cents",
                "hold_time_secs"
            ])
    
    def record_snapshot(self, snapshot: MarketSnapshot):
        """Record a market snapshot (buffered for efficiency)"""
        with self.lock:
            self.snapshot_buffer.append(snapshot)
            if len(self.snapshot_buffer) >= self.buffer_size:
                self._flush_snapshots()
    
    def _flush_snapshots(self):
        """Write buffered snapshots to disk"""
        if not self.snapshot_buffer:
            return
            
        with open(self.snapshots_path, "a", newline="") as f:
            w = csv.writer(f)
            for s in self.snapshot_buffer:
                # Extract depth quantities
                yes_depths = [q for _, q in s.yes_bid_depth[:3]] + [0, 0, 0]
                no_depths = [q for _, q in s.no_bid_depth[:3]] + [0, 0, 0]
                
                w.writerow([
                    s.timestamp, s.ticker,
                    s.yes_best_bid, s.yes_imp_ask, s.no_best_bid, s.no_imp_ask,
                    f"{s.midpoint:.1f}" if s.midpoint else "",
                    s.spread_cents,
                    yes_depths[0], yes_depths[1], yes_depths[2],
                    no_depths[0], no_depths[1], no_depths[2],
                    s.secs_to_close, s.clock_source
                ])
        
        self.snapshot_buffer.clear()
    
    def record_trade(self, trade: TradeEvent):
        """Record a trade execution"""
        with self.lock:
            with open(self.trades_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    trade.timestamp, trade.ticker, trade.strategy,
                    trade.action, trade.side, trade.intended_price, trade.fill_price,
                    trade.qty_intended, trade.qty_filled,
                    trade.reason, trade.edge_at_entry, trade.fair_price_estimate,
                    trade.volatility_estimate, trade.secs_to_close,
                    trade.estimated_fee_cents, trade.is_maker,
                    trade.order_id, trade.position_id
                ])
    
    def record_position(self, position: PositionRecord):
        """Record a completed position"""
        with self.lock:
            with open(self.positions_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    position.position_id, position.ticker, position.strategy,
                    position.entry_time, position.entry_side, position.entry_price,
                    position.entry_qty, position.entry_edge, position.entry_fee,
                    position.exit_time, position.exit_type, position.exit_price,
                    position.exit_qty, position.exit_fee,
                    position.lock_price, position.lock_side,
                    position.gross_pnl_cents, position.total_fees_cents, position.net_pnl_cents,
                    position.hold_time_secs
                ])
    
    def flush(self):
        """Flush all buffers to disk"""
        with self.lock:
            self._flush_snapshots()
    
    def get_paths(self) -> Dict[str, str]:
        """Return paths to all data files"""
        return {
            "snapshots": self.snapshots_path,
            "trades": self.trades_path,
            "positions": self.positions_path,
        }


# =============================================================================
# STRATEGY BASE CLASS
# =============================================================================

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Each strategy implementation handles its own entry/exit logic.
    """
    
    def __init__(self, config: StrategyConfig, data_collector: DataCollector):
        self.config = config
        self.collector = data_collector
        self.name = config.name
        
        # State
        self.open_positions: List[PositionRecord] = []
        self.closed_positions: List[PositionRecord] = []
        self.last_entry_time: Optional[float] = None
        self.total_collateral_used: float = 0.0
        
        # Tracking
        self.position_counter = 0
        
    def _generate_position_id(self) -> str:
        self.position_counter += 1
        return f"{self.name}_{self.position_counter}"
    
    @abstractmethod
    def should_enter(
        self,
        snapshot: MarketSnapshot,
        fair_price: float,
        volatility: float
    ) -> Tuple[bool, str, int, int, str]:
        """
        Decide whether to enter a position.
        
        Returns: (should_enter, side, price, qty, reason)
        """
        pass
    
    @abstractmethod
    def should_exit(
        self,
        position: PositionRecord,
        snapshot: MarketSnapshot,
        fair_price: float
    ) -> Tuple[bool, str, int, str]:
        """
        Decide whether to exit a position.
        
        Returns: (should_exit, exit_type, exit_price, reason)
        """
        pass
    
    def can_enter_new_position(self, snapshot: MarketSnapshot) -> Tuple[bool, str]:
        """Check if strategy constraints allow a new entry"""
        
        # Check position limit
        if len(self.open_positions) >= self.config.max_open_positions:
            return False, "max_positions_reached"
        
        # Check collateral limit
        if self.total_collateral_used >= self.config.max_collateral_dollars:
            return False, "max_collateral_reached"
        
        # Check time between entries
        if self.last_entry_time is not None:
            elapsed = time.time() - self.last_entry_time
            if elapsed < self.config.min_secs_between_entries:
                return False, f"throttle: {elapsed:.0f}s < {self.config.min_secs_between_entries}s"
        
        # Check time to close
        if snapshot.secs_to_close < self.config.stop_trading_before_close_secs:
            return False, f"near_close: {snapshot.secs_to_close}s remaining"
        
        return True, "ok"
    
    def record_entry(
        self,
        snapshot: MarketSnapshot,
        side: str,
        price: float,
        qty: int,
        edge: float,
        fee: float
    ) -> PositionRecord:
        """Record a new position entry"""
        position = PositionRecord(
            position_id=self._generate_position_id(),
            ticker=snapshot.ticker,
            strategy=self.name,
            entry_time=snapshot.timestamp,
            entry_side=side,
            entry_price=price,
            entry_qty=qty,
            entry_edge=edge,
            entry_fee=fee,
        )
        
        self.open_positions.append(position)
        self.last_entry_time = time.time()
        
        # Update collateral tracking
        if side == "yes":
            self.total_collateral_used += (100 - price) * qty / 100
        else:
            self.total_collateral_used += price * qty / 100
        
        return position
    
    def record_exit(
        self,
        position: PositionRecord,
        exit_type: str,
        exit_price: float,
        exit_qty: int,
        exit_fee: float,
        lock_price: Optional[float] = None,
        lock_side: Optional[str] = None
    ):
        """Record position exit and calculate P&L"""
        position.exit_time = dt.datetime.now(dt.timezone.utc).isoformat()
        position.exit_type = exit_type
        position.exit_price = exit_price
        position.exit_qty = exit_qty
        position.exit_fee = exit_fee
        position.lock_price = lock_price
        position.lock_side = lock_side
        
        # Calculate P&L
        if exit_type == "lock":
            # Locked pair: profit = 100 - entry - lock
            position.gross_pnl_cents = (100 - position.entry_price - lock_price) * exit_qty
        else:
            # Regular exit: profit/loss = exit - entry
            if position.entry_side == "yes":
                position.gross_pnl_cents = (exit_price - position.entry_price) * exit_qty
            else:
                position.gross_pnl_cents = (exit_price - position.entry_price) * exit_qty
        
        position.total_fees_cents = position.entry_fee + exit_fee
        position.net_pnl_cents = position.gross_pnl_cents - position.total_fees_cents
        
        # Calculate hold time
        entry_dt = dt.datetime.fromisoformat(position.entry_time.replace("Z", "+00:00"))
        exit_dt = dt.datetime.fromisoformat(position.exit_time.replace("Z", "+00:00"))
        position.hold_time_secs = int((exit_dt - entry_dt).total_seconds())
        
        # Move to closed
        self.open_positions.remove(position)
        self.closed_positions.append(position)
        
        # Update collateral
        if position.entry_side == "yes":
            self.total_collateral_used -= (100 - position.entry_price) * position.entry_qty / 100
        else:
            self.total_collateral_used -= position.entry_price * position.entry_qty / 100
        
        # Record to collector
        self.collector.record_position(position)
        
        return position
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        all_positions = self.closed_positions
        
        if not all_positions:
            return {
                "strategy": self.name,
                "trades": 0,
                "net_pnl_cents": 0,
            }
        
        wins = [p for p in all_positions if (p.net_pnl_cents or 0) > 0]
        losses = [p for p in all_positions if (p.net_pnl_cents or 0) < 0]
        
        return {
            "strategy": self.name,
            "trades": len(all_positions),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(all_positions) if all_positions else 0,
            "gross_pnl_cents": sum(p.gross_pnl_cents or 0 for p in all_positions),
            "total_fees_cents": sum(p.total_fees_cents or 0 for p in all_positions),
            "net_pnl_cents": sum(p.net_pnl_cents or 0 for p in all_positions),
            "avg_win_cents": sum(p.net_pnl_cents or 0 for p in wins) / len(wins) if wins else 0,
            "avg_loss_cents": sum(p.net_pnl_cents or 0 for p in losses) / len(losses) if losses else 0,
            "avg_hold_secs": sum(p.hold_time_secs or 0 for p in all_positions) / len(all_positions),
            "open_positions": len(self.open_positions),
        }


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

class MeanReversionStrategy(BaseStrategy):
    """
    Pure mean reversion: Buy when price drops sharply, sell when it recovers.
    No directional model needed - just assumes large moves revert.
    """
    
    def __init__(self, config: StrategyConfig, data_collector: DataCollector):
        super().__init__(config, data_collector)
        self.price_history: deque = deque(maxlen=30)
        self.recent_high: Optional[float] = None
        self.recent_low: Optional[float] = None
    
    def should_enter(
        self,
        snapshot: MarketSnapshot,
        fair_price: float,  # Ignored - we use our own
        volatility: float
    ) -> Tuple[bool, str, int, int, str]:
        
        if snapshot.midpoint is None:
            return False, "", 0, 0, "no_midpoint"
        
        # Update price history
        self.price_history.append(snapshot.midpoint)
        
        if len(self.price_history) < 10:
            return False, "", 0, 0, "insufficient_history"
        
        # Calculate recent range
        prices = list(self.price_history)
        self.recent_high = max(prices[-10:])
        self.recent_low = min(prices[-10:])
        
        # Moving average
        sma = sum(prices) / len(prices)
        
        # Standard deviation
        variance = sum((p - sma) ** 2 for p in prices) / len(prices)
        std = math.sqrt(variance) if variance > 0 else 5
        
        current = snapshot.midpoint
        
        # Entry signals: price is >1.5 std from mean
        if current < sma - 1.5 * std:
            # Price dropped - buy YES (betting on reversion up)
            if snapshot.yes_imp_ask is not None:
                edge = sma - current
                if edge >= self.config.min_entry_edge_cents:
                    return True, "yes", snapshot.yes_imp_ask, 1, f"mean_rev_buy: {edge:.1f}c below SMA"
        
        if current > sma + 1.5 * std:
            # Price spiked - buy NO (betting on reversion down)
            if snapshot.no_imp_ask is not None:
                edge = current - sma
                if edge >= self.config.min_entry_edge_cents:
                    return True, "no", snapshot.no_imp_ask, 1, f"mean_rev_sell: {edge:.1f}c above SMA"
        
        return False, "", 0, 0, "no_signal"
    
    def should_exit(
        self,
        position: PositionRecord,
        snapshot: MarketSnapshot,
        fair_price: float
    ) -> Tuple[bool, str, int, str]:
        
        if snapshot.midpoint is None:
            return False, "", 0, "no_midpoint"
        
        prices = list(self.price_history) if self.price_history else [snapshot.midpoint]
        sma = sum(prices) / len(prices)
        
        # For YES positions: exit when price reverts to mean or above
        if position.entry_side == "yes":
            current_bid = snapshot.yes_best_bid
            if current_bid is None:
                return False, "", 0, "no_bid"
            
            pnl = current_bid - position.entry_price
            
            # Take profit: reverted to mean or better
            if snapshot.midpoint >= sma - 2:
                return True, "take_profit", current_bid - 1, f"reverted_to_mean: {pnl:.0f}c"
            
            # Stop loss
            if pnl <= -self.config.stop_loss_cents:
                return True, "stop", current_bid - 1, f"stop_loss: {pnl:.0f}c"
        
        # For NO positions: exit when price reverts to mean or below
        else:
            current_bid = snapshot.no_best_bid
            if current_bid is None:
                return False, "", 0, "no_bid"
            
            pnl = current_bid - position.entry_price
            
            if snapshot.midpoint <= sma + 2:
                return True, "take_profit", current_bid - 1, f"reverted_to_mean: {pnl:.0f}c"
            
            if pnl <= -self.config.stop_loss_cents:
                return True, "stop", current_bid - 1, f"stop_loss: {pnl:.0f}c"
        
        return False, "", 0, "hold"


class SpreadCaptureStrategy(BaseStrategy):
    """
    Capture wide spreads by placing maker orders on both sides.
    Profit when both orders fill and the spread is captured.
    """
    
    def __init__(self, config: StrategyConfig, data_collector: DataCollector):
        super().__init__(config, data_collector)
        self.target_spread_cents = 6  # Minimum spread to target
    
    def should_enter(
        self,
        snapshot: MarketSnapshot,
        fair_price: float,
        volatility: float
    ) -> Tuple[bool, str, int, int, str]:
        
        if snapshot.spread_cents is None or snapshot.spread_cents < self.target_spread_cents:
            return False, "", 0, 0, f"spread_too_tight: {snapshot.spread_cents}c"
        
        # Look for opportunities where we can improve the best bid
        # and still have edge if filled
        
        if snapshot.yes_best_bid is not None and snapshot.yes_imp_ask is not None:
            # Try to buy YES by improving the bid
            our_bid = snapshot.yes_best_bid + 1
            potential_profit = snapshot.yes_imp_ask - our_bid
            
            if potential_profit >= self.config.min_entry_edge_cents:
                return True, "yes", our_bid, 1, f"spread_capture: {potential_profit}c potential"
        
        return False, "", 0, 0, "no_opportunity"
    
    def should_exit(
        self,
        position: PositionRecord,
        snapshot: MarketSnapshot,
        fair_price: float
    ) -> Tuple[bool, str, int, str]:
        
        # For spread capture, we want to lock quickly
        if position.entry_side == "yes":
            if snapshot.no_imp_ask is not None:
                lock_cost = position.entry_price + snapshot.no_imp_ask
                if lock_cost <= 100 - self.config.min_lock_profit_cents:
                    return True, "lock", snapshot.no_imp_ask, f"lock_available: {100 - lock_cost}c profit"
        else:
            if snapshot.yes_imp_ask is not None:
                lock_cost = position.entry_price + snapshot.yes_imp_ask
                if lock_cost <= 100 - self.config.min_lock_profit_cents:
                    return True, "lock", snapshot.yes_imp_ask, f"lock_available: {100 - lock_cost}c profit"
        
        # Stop loss
        if position.entry_side == "yes":
            current_bid = snapshot.yes_best_bid
            if current_bid is not None:
                pnl = current_bid - position.entry_price
                if pnl <= -self.config.stop_loss_cents:
                    return True, "stop", current_bid - 1, f"stop_loss: {pnl:.0f}c"
        else:
            current_bid = snapshot.no_best_bid
            if current_bid is not None:
                pnl = current_bid - position.entry_price
                if pnl <= -self.config.stop_loss_cents:
                    return True, "stop", current_bid - 1, f"stop_loss: {pnl:.0f}c"
        
        return False, "", 0, "hold"


class ModelEdgeStrategy(BaseStrategy):
    """
    Your original strategy: Enter when market price differs from model fair price.
    Enhanced with fee awareness and dynamic adjustments.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        data_collector: DataCollector,
        pregame_fair_cents: int
    ):
        super().__init__(config, data_collector)
        self.pregame_fair = pregame_fair_cents
        self.price_history: deque = deque(maxlen=30)
    
    def _get_dynamic_fair(self, snapshot: MarketSnapshot) -> float:
        """Blend pregame model with market price as game progresses"""
        if snapshot.midpoint is None:
            return float(self.pregame_fair)
        
        # Update history
        self.price_history.append(snapshot.midpoint)
        
        # Calculate game progress (assuming 2400s = 40 min game)
        total_game_secs = 2400
        elapsed = max(0, total_game_secs - snapshot.secs_to_close)
        progress = min(1.0, elapsed / total_game_secs)
        
        # Exponential decay of model confidence
        model_weight = math.exp(-2 * progress)
        market_midpoint = sum(self.price_history) / len(self.price_history)
        
        return model_weight * self.pregame_fair + (1 - model_weight) * market_midpoint
    
    def should_enter(
        self,
        snapshot: MarketSnapshot,
        fair_price: float,  # External fair price (may override)
        volatility: float
    ) -> Tuple[bool, str, int, int, str]:
        
        # Use our dynamic fair price
        fair = self._get_dynamic_fair(snapshot)
        
        candidates = []
        
        # Check YES side
        if snapshot.yes_imp_ask is not None:
            yes_edge = fair - snapshot.yes_imp_ask
            if yes_edge >= self.config.min_entry_edge_cents:
                candidates.append(("yes", snapshot.yes_imp_ask, yes_edge))
        
        # Check NO side
        if snapshot.no_imp_ask is not None:
            no_fair = 100 - fair
            no_edge = no_fair - snapshot.no_imp_ask
            if no_edge >= self.config.min_entry_edge_cents:
                candidates.append(("no", snapshot.no_imp_ask, no_edge))
        
        if not candidates:
            return False, "", 0, 0, "no_edge"
        
        # Take best edge
        candidates.sort(key=lambda x: x[2], reverse=True)
        side, price, edge = candidates[0]
        
        # Size based on edge
        qty = 1 if edge < 10 else (2 if edge < 15 else 3)
        qty = min(qty, self.config.max_position_qty)
        
        return True, side, price, qty, f"model_edge: {edge:.1f}c vs fair={fair:.1f}c"
    
    def should_exit(
        self,
        position: PositionRecord,
        snapshot: MarketSnapshot,
        fair_price: float
    ) -> Tuple[bool, str, int, str]:
        
        # Try to lock first
        if position.entry_side == "yes":
            if snapshot.no_imp_ask is not None:
                lock_cost = position.entry_price + snapshot.no_imp_ask
                profit = 100 - lock_cost
                if profit >= self.config.min_lock_profit_cents:
                    return True, "lock", snapshot.no_imp_ask, f"lock: {profit:.0f}c"
            
            # Stop/take profit
            current_bid = snapshot.yes_best_bid
            if current_bid is not None:
                pnl = current_bid - position.entry_price
                if pnl >= self.config.take_profit_cents:
                    return True, "take_profit", current_bid - 1, f"take_profit: {pnl:.0f}c"
                if pnl <= -self.config.stop_loss_cents:
                    return True, "stop", current_bid - 1, f"stop_loss: {pnl:.0f}c"
        
        else:  # NO position
            if snapshot.yes_imp_ask is not None:
                lock_cost = position.entry_price + snapshot.yes_imp_ask
                profit = 100 - lock_cost
                if profit >= self.config.min_lock_profit_cents:
                    return True, "lock", snapshot.yes_imp_ask, f"lock: {profit:.0f}c"
            
            current_bid = snapshot.no_best_bid
            if current_bid is not None:
                pnl = current_bid - position.entry_price
                if pnl >= self.config.take_profit_cents:
                    return True, "take_profit", current_bid - 1, f"take_profit: {pnl:.0f}c"
                if pnl <= -self.config.stop_loss_cents:
                    return True, "stop", current_bid - 1, f"stop_loss: {pnl:.0f}c"
        
        return False, "", 0, "hold"


# =============================================================================
# STRATEGY MANAGER (Run multiple strategies in parallel)
# =============================================================================

class StrategyManager:
    """
    Manages multiple strategies running on the same market.
    Handles resource allocation and performance comparison.
    """
    
    def __init__(self, game_label: str, total_capital: float = 10.0):
        self.game_label = game_label
        self.total_capital = total_capital
        self.collector = DataCollector(game_label)
        self.strategies: List[BaseStrategy] = []
        
    def add_strategy(self, strategy: BaseStrategy):
        self.strategies.append(strategy)
        
    def process_snapshot(self, snapshot: MarketSnapshot, fair_price: float, volatility: float):
        """
        Process a market snapshot across all strategies.
        In production, this would also handle order execution.
        """
        # Record the snapshot
        self.collector.record_snapshot(snapshot)
        
        # Check each strategy
        for strategy in self.strategies:
            if not strategy.config.enabled:
                continue
            
            # Check existing positions for exits
            for position in list(strategy.open_positions):
                should_exit, exit_type, exit_price, reason = strategy.should_exit(
                    position, snapshot, fair_price
                )
                if should_exit:
                    # In production: execute exit order here
                    # For now, just record the decision
                    print(f"[{strategy.name}] EXIT {position.entry_side.upper()} "
                          f"@ {exit_price}c ({exit_type}): {reason}")
            
            # Check for new entries
            can_enter, throttle_reason = strategy.can_enter_new_position(snapshot)
            if can_enter:
                should_enter, side, price, qty, reason = strategy.should_enter(
                    snapshot, fair_price, volatility
                )
                if should_enter:
                    # In production: execute entry order here
                    print(f"[{strategy.name}] ENTER {side.upper()} {qty}x @ {price}c: {reason}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary across all strategies"""
        return {
            "game": self.game_label,
            "strategies": [s.get_stats() for s in self.strategies],
            "data_files": self.collector.get_paths(),
        }
    
    def flush_data(self):
        """Flush all data to disk"""
        self.collector.flush()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_setup():
    """
    Example of how to set up multi-strategy testing.
    """
    # Create manager
    manager = StrategyManager("test_game", total_capital=10.0)
    
    # Add mean reversion strategy
    mr_config = StrategyConfig(
        name="mean_reversion",
        max_collateral_dollars=3.0,
        min_entry_edge_cents=8,
        stop_loss_cents=12,
    )
    manager.add_strategy(MeanReversionStrategy(mr_config, manager.collector))
    
    # Add spread capture strategy
    sc_config = StrategyConfig(
        name="spread_capture",
        max_collateral_dollars=3.0,
        min_entry_edge_cents=6,
        stop_loss_cents=10,
    )
    manager.add_strategy(SpreadCaptureStrategy(sc_config, manager.collector))
    
    # Add model edge strategy
    me_config = StrategyConfig(
        name="model_edge",
        max_collateral_dollars=4.0,
        min_entry_edge_cents=10,  # Higher threshold due to fees
        stop_loss_cents=15,
    )
    manager.add_strategy(ModelEdgeStrategy(me_config, manager.collector, pregame_fair_cents=55))
    
    print("Strategy Lab initialized with:")
    for s in manager.strategies:
        print(f"  - {s.name}: ${s.config.max_collateral_dollars} allocation")
    
    return manager


if __name__ == "__main__":
    print("Kalshi Strategy Lab")
    print("="*40)
    manager = example_setup()
    
    # Simulate a few snapshots
    for i in range(5):
        snapshot = MarketSnapshot(
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            ticker="TEST-TICKER",
            yes_best_bid=50 + i,
            yes_best_ask=None,
            yes_imp_ask=52 + i,
            no_best_bid=48 - i,
            no_best_ask=None,
            no_imp_ask=50 - i,
            secs_to_close=2000 - i*100,
            clock_source="test",
        )
        
        manager.process_snapshot(snapshot, fair_price=55.0, volatility=5.0)
    
    manager.flush_data()
    
    print("\nData files created:")
    for name, path in manager.collector.get_paths().items():
        print(f"  {name}: {path}")