# kalshi_improvements.py
# Drop-in improvements for combo_vnext.py
# These can be imported and used to enhance the existing strategy

import math
import time
from typing import Dict, Any, Optional, Tuple, List, Callable
from collections import deque
from dataclasses import dataclass, field
import datetime as dt

# =============================================================================
# FEE CALCULATIONS
# =============================================================================

def calculate_taker_fee_cents(price_cents: int, contracts: int = 1) -> float:
    """
    Kalshi taker fee formula: 0.07 × C × P × (1-P)
    Max 2c per contract at 50c prices
    
    Args:
        price_cents: Price in cents (1-99)
        contracts: Number of contracts
        
    Returns:
        Fee in cents
    """
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    
    p = price_cents / 100.0
    fee_per_contract = 0.07 * p * (1 - p)
    fee_per_contract = min(fee_per_contract, 0.02)  # Max 2c at 50/50
    return fee_per_contract * contracts * 100  # Return in cents


def calculate_maker_fee_cents(price_cents: int, contracts: int = 1) -> float:
    """Maker fee is 1/4 of taker fee"""
    return calculate_taker_fee_cents(price_cents, contracts) / 4


def round_trip_fee_cents(
    entry_price: int,
    exit_price: int,
    contracts: int = 1,
    entry_is_taker: bool = True,
    exit_is_taker: bool = True
) -> float:
    """Calculate total fees for a round-trip trade"""
    entry_fee = (calculate_taker_fee_cents if entry_is_taker else calculate_maker_fee_cents)(
        entry_price, contracts
    )
    exit_fee = (calculate_taker_fee_cents if exit_is_taker else calculate_maker_fee_cents)(
        exit_price, contracts
    )
    return entry_fee + exit_fee


# =============================================================================
# FEE-AWARE ENTRY THRESHOLDS
# =============================================================================

def minimum_edge_for_profit(
    price_cents: int,
    lock_probability: float = 0.6,
    stop_loss_cents: int = 15,
    min_lock_profit_cents: int = 8
) -> float:
    """
    Calculate minimum edge needed to be +EV after fees.
    
    The key insight: if you enter at a price, you'll either:
    1. Lock a pair (probability = lock_probability)
    2. Hit stop loss (probability = 1 - lock_probability)
    
    EV = P(lock) × (lock_profit - entry_fee - lock_fee)
       + P(stop) × (-stop_loss - entry_fee - exit_fee)
       
    For EV > 0, we need sufficient edge.
    """
    entry_fee = calculate_taker_fee_cents(price_cents, 1)
    lock_fee = calculate_taker_fee_cents(100 - price_cents, 1)  # Approximate
    exit_fee = calculate_taker_fee_cents(price_cents, 1)
    
    p_stop = 1 - lock_probability
    
    # Net profit if we lock
    lock_net = min_lock_profit_cents - entry_fee - lock_fee
    
    # Net loss if we stop
    stop_net = stop_loss_cents + entry_fee + exit_fee
    
    # For EV = 0:
    # lock_probability × lock_net = p_stop × stop_net
    # We need lock_net > (p_stop × stop_net) / lock_probability
    
    min_lock_needed = (p_stop * stop_net) / lock_probability
    
    # The edge we need = price we're willing to pay - fair price
    # If we can lock at lock_px, total cost = our_px + lock_px
    # Profit per pair = 100 - our_px - lock_px
    # We need this profit > min_lock_needed
    # So our_px + lock_px < 100 - min_lock_needed
    
    return min_lock_needed


def recommended_entry_threshold(
    market_price_cents: int,
    lock_probability: float = 0.6,
    stop_loss_cents: int = 15
) -> int:
    """
    Returns the recommended minimum edge in cents to enter a trade.
    This accounts for fees and stop-loss probability.
    """
    min_profit = minimum_edge_for_profit(
        market_price_cents, lock_probability, stop_loss_cents
    )
    
    # Add buffer for execution slippage
    buffer_cents = 2
    
    return int(math.ceil(min_profit / 2 + buffer_cents))


# =============================================================================
# DYNAMIC FAIR PRICE
# =============================================================================

def update_fair_price(
    pregame_fair_cents: int,
    current_midpoint_cents: float,
    secs_elapsed: int,
    total_game_secs: int = 2400
) -> float:
    """
    Blend pre-game model with market-implied price as game progresses.
    
    Rationale: Early in the game, your pre-game model still has value.
    As the game progresses, the market sees the score and you don't,
    so you should trust the market more.
    
    Uses exponential decay of model confidence.
    """
    if secs_elapsed <= 0:
        return float(pregame_fair_cents)
    
    game_progress = min(1.0, secs_elapsed / total_game_secs)
    
    # Exponential decay: model weight drops from 1.0 to ~0.14 by end
    decay_rate = 2.0
    model_weight = math.exp(-decay_rate * game_progress)
    market_weight = 1 - model_weight
    
    blended = model_weight * pregame_fair_cents + market_weight * current_midpoint_cents
    
    return blended


class FairPriceTracker:
    """
    Tracks fair price evolution during a game.
    Provides smoothed estimates and confidence intervals.
    """
    
    def __init__(
        self,
        pregame_fair_cents: int,
        total_game_secs: int = 2400,
        smoothing_window: int = 10
    ):
        self.pregame_fair = pregame_fair_cents
        self.total_game_secs = total_game_secs
        self.midpoints: deque = deque(maxlen=smoothing_window)
        self.start_time: Optional[float] = None
        
    def update(self, yes_bid: int, no_bid: int) -> float:
        """
        Update with new market data and return current fair price estimate.
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        # Calculate midpoint
        if yes_bid is not None and no_bid is not None:
            midpoint = (yes_bid + (100 - no_bid)) / 2
            self.midpoints.append(midpoint)
        
        # Get smoothed midpoint
        if self.midpoints:
            smoothed_midpoint = sum(self.midpoints) / len(self.midpoints)
        else:
            smoothed_midpoint = self.pregame_fair
            
        # Calculate elapsed time
        secs_elapsed = time.time() - self.start_time
        
        return update_fair_price(
            self.pregame_fair,
            smoothed_midpoint,
            int(secs_elapsed),
            self.total_game_secs
        )
    
    def get_confidence(self) -> float:
        """
        Returns confidence in fair price estimate (0-1).
        Lower when market is volatile or moving fast.
        """
        if len(self.midpoints) < 3:
            return 0.5
            
        # Use coefficient of variation
        midpoints = list(self.midpoints)
        mean = sum(midpoints) / len(midpoints)
        if mean == 0:
            return 0.5
            
        variance = sum((x - mean) ** 2 for x in midpoints) / len(midpoints)
        std = math.sqrt(variance)
        cv = std / mean
        
        # Higher CV = lower confidence
        confidence = max(0.1, 1 - cv * 5)
        return min(1.0, confidence)


# =============================================================================
# VOLATILITY TRACKING
# =============================================================================

class VolatilityTracker:
    """
    Tracks recent price volatility for dynamic stop-loss and entry decisions.
    """
    
    def __init__(self, window_size: int = 30, poll_interval_secs: float = 4.0):
        self.prices: deque = deque(maxlen=window_size)
        self.poll_interval = poll_interval_secs
        
    def update(self, midpoint: float) -> None:
        self.prices.append(midpoint)
        
    def get_volatility(self) -> float:
        """Returns estimated volatility (standard deviation of recent prices)"""
        if len(self.prices) < 5:
            return 10.0  # Default high volatility
            
        prices = list(self.prices)
        mean = sum(prices) / len(prices)
        variance = sum((x - mean) ** 2 for x in prices) / len(prices)
        return math.sqrt(variance)
    
    def get_dynamic_stop(self, base_stop_cents: int = 15) -> int:
        """
        Returns volatility-adjusted stop loss in cents.
        In volatile markets, use wider stops to avoid noise.
        """
        vol = self.get_volatility()
        
        # Stop should be at least 2 standard deviations
        vol_based_stop = vol * 2
        
        # But not less than base stop
        return max(base_stop_cents, int(math.ceil(vol_based_stop)))
    
    def is_trending(self, lookback: int = 10) -> Tuple[bool, str]:
        """
        Detect if price is trending (vs mean-reverting).
        Returns (is_trending, direction)
        """
        if len(self.prices) < lookback:
            return False, "neutral"
            
        recent = list(self.prices)[-lookback:]
        
        # Simple linear regression
        n = len(recent)
        sum_x = sum(range(n))
        sum_y = sum(recent)
        sum_xy = sum(i * y for i, y in enumerate(recent))
        sum_x2 = sum(i ** 2 for i in range(n))
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return False, "neutral"
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Significant trend if slope > 0.5 cents per observation
        if abs(slope) > 0.5:
            direction = "up" if slope > 0 else "down"
            return True, direction
            
        return False, "neutral"


# =============================================================================
# ENTRY THROTTLING
# =============================================================================

class EntryThrottle:
    """
    Prevents rapid-fire entries that can compound losses.
    Enforces minimum time between entries and max entries per period.
    """
    
    def __init__(
        self,
        min_gap_secs: int = 30,
        max_per_period: int = 3,
        period_secs: int = 300
    ):
        self.entry_times: List[float] = []
        self.min_gap_secs = min_gap_secs
        self.max_per_period = max_per_period
        self.period_secs = period_secs
        
    def can_enter(self) -> Tuple[bool, str]:
        """
        Returns (can_enter, reason)
        """
        now = time.time()
        
        # Clean old entries
        self.entry_times = [
            t for t in self.entry_times 
            if now - t < self.period_secs
        ]
        
        # Check rate limit
        if len(self.entry_times) >= self.max_per_period:
            return False, f"rate_limit: {len(self.entry_times)}/{self.max_per_period} in period"
            
        # Check minimum gap
        if self.entry_times and now - self.entry_times[-1] < self.min_gap_secs:
            gap = now - self.entry_times[-1]
            return False, f"min_gap: {gap:.0f}s < {self.min_gap_secs}s"
            
        return True, "ok"
        
    def record_entry(self) -> None:
        self.entry_times.append(time.time())
        
    def reset(self) -> None:
        self.entry_times.clear()


# =============================================================================
# ENHANCED TRADE LOGGING
# =============================================================================

@dataclass
class TradeRecord:
    """Detailed record of a single trade"""
    timestamp: str
    action: str  # "entry", "lock", "stop", "take_profit", "dead_cap_exit"
    side: str
    qty: int
    intended_price: int
    actual_price: float
    fee_cents: float
    order_id: str = ""
    edge_at_entry: float = 0.0
    volatility_at_entry: float = 0.0
    secs_to_close: int = 0
    notes: str = ""


@dataclass 
class EnhancedStrategyState:
    """
    Enhanced state tracking with fees, detailed trade log, and analytics.
    Can be used alongside or as a replacement for StrategyState.
    """
    # Position tracking
    open_side: Optional[str] = None
    open_vwap_c: Optional[float] = None
    open_qty: int = 0
    open_opened_at: Optional[dt.datetime] = None
    open_entry_edge: float = 0.0
    
    # Locked pairs: (yes_px, no_px, qty, entry_ts)
    pairs: List[Tuple[float, float, int, str]] = field(default_factory=list)
    
    # PnL tracking WITH FEES
    realized_pnl_cents: int = 0
    total_fees_cents: float = 0.0
    
    # Blocked sides
    stop_out_sides: set = field(default_factory=set)
    
    # Detailed trade log
    trades: List[TradeRecord] = field(default_factory=list)
    
    # Analytics
    entries_count: int = 0
    locks_count: int = 0
    stops_count: int = 0
    take_profits_count: int = 0
    
    def record_trade(
        self,
        action: str,
        side: str,
        qty: int,
        intended_price: int,
        actual_price: float,
        fee_cents: float,
        **kwargs
    ) -> None:
        """Record a trade with full details"""
        self.total_fees_cents += fee_cents
        
        record = TradeRecord(
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            action=action,
            side=side,
            qty=qty,
            intended_price=intended_price,
            actual_price=actual_price,
            fee_cents=fee_cents,
            **kwargs
        )
        self.trades.append(record)
        
        # Update counters
        if action == "entry":
            self.entries_count += 1
        elif action == "lock":
            self.locks_count += 1
        elif action == "stop":
            self.stops_count += 1
        elif action == "take_profit":
            self.take_profits_count += 1
            
    def locked_profit_cents(self) -> int:
        """Total locked profit across all pairs"""
        total = 0
        for py, pn, q, _ in self.pairs:
            total += int(round((100 - py - pn) * q))
        return total
    
    def gross_pnl_cents(self, best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> int:
        """PnL before fees"""
        locked = self.locked_profit_cents()
        mtm = self._mark_to_market(best_yes_bid, best_no_bid)
        return self.realized_pnl_cents + locked + mtm
    
    def net_pnl_cents(self, best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> float:
        """PnL after fees"""
        return self.gross_pnl_cents(best_yes_bid, best_no_bid) - self.total_fees_cents
    
    def _mark_to_market(self, best_yes_bid: Optional[int], best_no_bid: Optional[int]) -> int:
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
    
    def lock_success_rate(self) -> float:
        """Percentage of entries that result in locked pairs"""
        if self.entries_count == 0:
            return 0.0
        return self.locks_count / self.entries_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of trading performance"""
        return {
            "entries": self.entries_count,
            "locks": self.locks_count,
            "stops": self.stops_count,
            "take_profits": self.take_profits_count,
            "lock_rate": f"{self.lock_success_rate():.1%}",
            "gross_pnl_cents": "N/A (need current prices)",
            "total_fees_cents": f"{self.total_fees_cents:.1f}",
            "pairs_locked": len(self.pairs),
            "locked_profit_cents": self.locked_profit_cents(),
        }


# =============================================================================
# GAME PHASE DETECTION
# =============================================================================

def detect_game_phase(secs_to_end: int) -> str:
    """
    Categorize the current game phase for strategy adjustment.
    
    Returns one of: "early", "middle", "late", "final"
    """
    if secs_to_end > 2100:  # More than 35 min left
        return "early"
    elif secs_to_end > 1200:  # 20-35 min left
        return "middle"  
    elif secs_to_end > 300:  # 5-20 min left
        return "late"
    else:  # Final 5 minutes
        return "final"


def get_phase_params(phase: str) -> Dict[str, Any]:
    """
    Get recommended strategy parameters for each game phase.
    """
    params = {
        "early": {
            "min_entry_edge": 5,
            "min_lock_profit": 10,
            "max_spread": 8,
            "stop_loss": 20,
            "take_profit": 12,
            "trade_enabled": True,
            "notes": "Market still forming, more opportunities but also more noise"
        },
        "middle": {
            "min_entry_edge": 6,
            "min_lock_profit": 10,
            "max_spread": 6,
            "stop_loss": 18,
            "take_profit": 10,
            "trade_enabled": True,
            "notes": "Market more efficient, moderate approach"
        },
        "late": {
            "min_entry_edge": 8,
            "min_lock_profit": 8,
            "max_spread": 5,
            "stop_loss": 15,
            "take_profit": 8,
            "trade_enabled": True,
            "notes": "Higher volatility, need larger edge"
        },
        "final": {
            "min_entry_edge": 15,
            "min_lock_profit": 6,
            "max_spread": 4,
            "stop_loss": 10,
            "take_profit": 6,
            "trade_enabled": False,  # Often better to not trade
            "notes": "Extreme volatility, binary outcomes. Consider closing positions."
        }
    }
    return params.get(phase, params["middle"])


# =============================================================================
# IMPROVED ENTRY DECISION
# =============================================================================

def should_enter_position(
    prices: Dict[str, Any],
    fair_price_tracker: FairPriceTracker,
    volatility_tracker: VolatilityTracker,
    entry_throttle: EntryThrottle,
    secs_to_end: int,
    current_side_blocked: set
) -> Tuple[bool, str, int, int, str]:
    """
    Comprehensive entry decision incorporating all factors.
    
    Returns:
        (should_enter, side, price, qty, reason)
    """
    # Check throttle
    can_enter, throttle_reason = entry_throttle.can_enter()
    if not can_enter:
        return False, "", 0, 0, f"throttled: {throttle_reason}"
    
    # Get game phase parameters
    phase = detect_game_phase(secs_to_end)
    params = get_phase_params(phase)
    
    if not params["trade_enabled"]:
        return False, "", 0, 0, f"phase={phase}: trading disabled"
    
    # Get current fair price
    yes_bid = prices.get("best_yes_bid")
    no_bid = prices.get("best_no_bid")
    
    if yes_bid is None or no_bid is None:
        return False, "", 0, 0, "no_market_data"
    
    fair = fair_price_tracker.update(yes_bid, no_bid)
    confidence = fair_price_tracker.get_confidence()
    
    # Check trend - don't fight strong trends
    is_trending, trend_dir = volatility_tracker.is_trending()
    
    # Calculate implied asks
    imp_yes_ask = prices.get("imp_yes_ask")
    imp_no_ask = prices.get("imp_no_ask")
    
    if imp_yes_ask is None or imp_no_ask is None:
        return False, "", 0, 0, "no_implied_ask"
    
    # Calculate spread
    spread = imp_yes_ask + imp_no_ask - 100
    if spread > params["max_spread"]:
        return False, "", 0, 0, f"spread_too_wide: {spread}c"
    
    # Evaluate YES side
    yes_edge = fair - imp_yes_ask
    
    # Evaluate NO side  
    no_fair = 100 - fair
    no_edge = no_fair - imp_no_ask
    
    # Get fee-aware minimum edge
    min_edge_yes = recommended_entry_threshold(
        int(imp_yes_ask),
        lock_probability=0.6,
        stop_loss_cents=params["stop_loss"]
    )
    min_edge_no = recommended_entry_threshold(
        int(imp_no_ask),
        lock_probability=0.6,
        stop_loss_cents=params["stop_loss"]
    )
    
    # Also use phase-based minimum
    min_edge_yes = max(min_edge_yes, params["min_entry_edge"])
    min_edge_no = max(min_edge_no, params["min_entry_edge"])
    
    candidates = []
    
    if yes_edge >= min_edge_yes and "yes" not in current_side_blocked:
        # Don't buy YES if trending strongly down
        if not (is_trending and trend_dir == "down"):
            candidates.append(("yes", int(imp_yes_ask), yes_edge))
            
    if no_edge >= min_edge_no and "no" not in current_side_blocked:
        # Don't buy NO if trending strongly up
        if not (is_trending and trend_dir == "up"):
            candidates.append(("no", int(imp_no_ask), no_edge))
    
    if not candidates:
        return False, "", 0, 0, "no_edge"
    
    # Take best edge
    candidates.sort(key=lambda x: x[2], reverse=True)
    side, price, edge = candidates[0]
    
    # Simple qty based on edge (can be improved)
    qty = 1 if edge < 10 else (2 if edge < 15 else 3)
    
    reason = f"edge={edge:.1f}c, fair={fair:.1f}c, phase={phase}, conf={confidence:.2f}"
    return True, side, price, qty, reason


# =============================================================================
# EXAMPLE INTEGRATION
# =============================================================================

def example_main_loop_integration():
    """
    Example of how to integrate these improvements into the main trading loop.
    This is pseudocode showing the structure.
    """
    
    # Initialize trackers
    fair_tracker = FairPriceTracker(pregame_fair_cents=58, total_game_secs=2400)
    vol_tracker = VolatilityTracker(window_size=30)
    throttle = EntryThrottle(min_gap_secs=30, max_per_period=3)
    state = EnhancedStrategyState()
    
    while True:  # Main loop
        # Fetch prices
        # prices = fetch_orderbook(...)
        # secs_to_end = ...
        
        # Update trackers
        # midpoint = (yes_bid + (100 - no_bid)) / 2
        # vol_tracker.update(midpoint)
        
        # Dynamic stop
        # dynamic_stop = vol_tracker.get_dynamic_stop(base_stop_cents=15)
        
        # Entry decision
        # should_enter, side, price, qty, reason = should_enter_position(
        #     prices, fair_tracker, vol_tracker, throttle, secs_to_end, state.stop_out_sides
        # )
        
        # if should_enter:
        #     # Execute entry with fee tracking
        #     fee = calculate_taker_fee_cents(price, qty)
        #     # ... execute order ...
        #     state.record_trade("entry", side, qty, price, actual_fill_price, fee)
        #     throttle.record_entry()
        
        # ... rest of loop (lock, stop, etc.) ...
        
        pass


if __name__ == "__main__":
    # Quick test of fee calculations
    print("Fee calculation tests:")
    for price in [25, 50, 75]:
        taker = calculate_taker_fee_cents(price, 1)
        maker = calculate_maker_fee_cents(price, 1)
        print(f"  Price {price}c: taker={taker:.3f}c, maker={maker:.3f}c")
    
    print("\nMinimum profitable edge tests:")
    for price in [40, 50, 60]:
        for lock_prob in [0.5, 0.6, 0.7]:
            min_edge = recommended_entry_threshold(price, lock_prob)
            print(f"  Price {price}c, P(lock)={lock_prob}: min_edge={min_edge}c")
    
    print("\nDynamic fair price tests:")
    for secs in [0, 600, 1200, 1800, 2400]:
        fair = update_fair_price(
            pregame_fair_cents=58,
            current_midpoint_cents=45,
            secs_elapsed=secs
        )
        print(f"  At {secs}s: blended_fair={fair:.1f}c (pregame=58, market=45)")