# superbowl_mr_strategy.py
# Mean Reversion Strategy adapted for Super Bowl LIX
# Patriots vs Seahawks - Feb 8, 2026
#
# Key adaptations from NCAAM:
#   - Longer lookback (NFL scoring is slower)
#   - Higher entry threshold (tighter spreads, more competition)
#   - Larger position sizes (more liquidity)
#   - NFL-specific clock handling
#   - Conservative parameters for first live test

import os
import sys
import math
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import datetime as dt

from production_strategies import calc_taker_fee


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from combo_vnext import (
    _load_private_key,
    _get,
    fetch_orderbook,
    derive_prices,
    place_limit_buy,
    place_limit_sell,
    wait_for_fill_or_timeout,
    print_status,
    utc_now,
    parse_iso,
)


# =============================================================================
# SUPER BOWL SPECIFIC PARAMETERS
# =============================================================================

@dataclass
class SuperBowlMRConfig:
    """Configuration tuned for Super Bowl markets"""
    
    # Entry parameters (more conservative than NCAAM)
    lookback_window: int = 40              # Longer: NFL scoring is slower
    entry_std_mult: float = 2.0            # Higher: need more extreme moves
    min_entry_edge_cents: int = 8          # Higher: tighter spreads, more fees
    
    # Position management
    max_positions: int = 3                 # Can handle more due to liquidity
    max_position_size: int = 5             # Larger sizes viable
    max_capital_dollars: float = 70.0      # Actual allocation (via SB_MAX_CAPITAL env var)
    
    # Exit parameters (NO stop loss - same as NCAAM learning)
    use_stop_loss: bool = False
    take_profit_std_mult: float = 0.5      # Exit when back to ~1σ from mean
    min_hold_secs_after_entry: int = 90
    pnl_floor_cents: float = -4.0          # ← new, same as above


    # Timing
    min_entry_gap_secs: int = 60           # Slower - less frequent opportunities
    warmup_observations: int = 15          # More warmup for stability
    
    # Late game behavior (final 5 minutes)
    final_period_secs: int = 300
    final_flatten_mode: bool = True        # Flatten positions in final 5 min
    
    # Market quality filters
    min_spread_cents: int = 2              # Skip if spread too tight (fees)
    max_spread_cents: int = 12             # Skip if spread too wide (illiquid)
    
    # NFL-specific
    game_duration_secs: int = 3600         # 60 minutes

    # --- SIZING (mirrors CBB MeanReversionStrategy) ---
    # Sizing scaled for $75 allocation (vs original $20 assumption)
    size_base_frac: float = 0.35           # fraction of remaining capital at min σ
    size_max_frac: float = 0.80            # fraction at 3σ+
    size_sigma_floor: float = 1.5          # σ that maps to base_frac
    size_sigma_cap: float = 3.0            # σ that maps to max_frac
    size_min_qty: int = 3                  # never smaller than this (matches CBB)
    max_order_qty: int = 80                # cap scaled for $75 allocation (~60% max)

class SuperBowlMRStrategy:
    """
    Mean Reversion for Super Bowl - key differences from NCAAM:
    
    1. More conservative entry (2.0σ vs 1.5σ)
    2. Longer lookback (40 vs 20-30)
    3. Higher minimum edge (8¢ vs 5-6¢)
    4. Larger position sizes (liquidity supports it)
    5. NO stop loss (learned from Feb 5/6)
    6. Flatten in final 5 minutes
    """
    
    def __init__(
        self, 
        max_capital: float,
        preferred_side: Optional[str] = None,
        config: Optional[SuperBowlMRConfig] = None
    ):
        self.name = "superbowl_mr"
        self.config = config or SuperBowlMRConfig()
        self.max_capital = max_capital
        self.preferred_side = preferred_side  # "yes"/"no" or None
        
        # State
        self.positions: List[Dict] = []
        self.closed: List[Dict] = []
        self.capital_used = 0.0
        self.position_counter = 0
        
        # Price tracking
        self.midpoints: deque = deque(maxlen=self.config.lookback_window)
        self.volumes: deque = deque(maxlen=self.config.lookback_window)
        
        # Throttling
        self.last_entry_time: Optional[float] = None
        
        # Stats
        self.total_fees = 0.0
        self.snapshot_count = 0
        
        # Market quality tracking
        self.recent_spreads: deque = deque(maxlen=10)
        self.recent_volumes: deque = deque(maxlen=10)
    
    def _calc_stats(self) -> Tuple[float, float]:
        """Calculate mean and std of recent midpoints"""
        if len(self.midpoints) < 5:
            return 50.0, 10.0
        
        prices = list(self.midpoints)
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = math.sqrt(variance) if variance > 0 else 5.0
        
        return mean, std
    
    def _is_market_quality_acceptable(self, prices: Dict) -> Tuple[bool, str]:
        """
        P3 fix: Pre-filter market quality
        Don't trade in dead/illiquid markets
        """
        # Need warmup first
        if self.snapshot_count < self.config.warmup_observations:
            return True, "warmup"
        
        spread = prices.get("spread_cents", 0)
        self.recent_spreads.append(spread)
        
        # Check spread bounds
        if spread < self.config.min_spread_cents:
            return False, f"spread_too_tight:{spread}¢"
        
        if spread > self.config.max_spread_cents:
            return False, f"spread_too_wide:{spread}¢"
        
        # Check if spread is widening (market getting worse)
        if len(self.recent_spreads) >= 5:
            avg_recent = sum(list(self.recent_spreads)[-5:]) / 5
            if avg_recent > self.config.max_spread_cents * 0.8:
                return False, f"deteriorating_liquidity:{avg_recent:.1f}¢"
        
        return True, "ok"

    def _collateral_per_contract(self, side: str, price_cents: int) -> float:
        """
        Approx collateral in dollars per contract.

        IMPORTANT: Match CBB behavior:
          - Kalshi cash required to BUY:
              YES @ p → you pay p cents
              NO  @ p → you pay p cents
        """
        return price_cents / 100.0

    def _size_qty(self, side: str, price_cents: int, sigma_mult: float = 1.5) -> int:
        """
        σ-scaled tranche sizing (mirrors CBB MeanReversionStrategy):

          - Compute remaining capital in dollars
          - Choose a fraction of that capital based on signal strength (σ)
          - 1.5σ  → base_frac of remaining
            3σ+   → max_frac of remaining
          - Floors and caps via size_min_qty / max_order_qty.
        """
        remaining = max(0.0, float(self.max_capital) - float(self.capital_used))
        per = self._collateral_per_contract(side, price_cents)
        if per <= 0:
            return 0

        # How many contracts could we buy with ALL remaining capital?
        max_possible = int(remaining / per)
        if max_possible <= 0:
            return 0

        sigma_floor = float(self.config.size_sigma_floor)
        sigma_cap = float(self.config.size_sigma_cap)
        base_frac = float(self.config.size_base_frac)
        max_frac = float(self.config.size_max_frac)

        # Clamp sigma_mult into [floor, cap] then interpolate
        t = max(0.0, min(1.0, (sigma_mult - sigma_floor) /
                         max(0.01, sigma_cap - sigma_floor)))
        frac = base_frac + t * (max_frac - base_frac)

        qty = int(max_possible * frac)

        # Apply floor and cap
        min_qty = int(self.config.size_min_qty)
        max_qty = int(self.config.max_order_qty)

        # Floor, but don't exceed what we can afford
        qty = max(min(min_qty, max_possible), qty)
        qty = min(qty, max_qty)
        qty = min(qty, max_possible)  # final safety

        return max(0, qty)
  
    def evaluate_entry(
        self,
        prices: Dict[str, Any],
        secs_to_close: int,
        context: Dict[str, Any]
    ) -> Tuple[bool, str, int, int, str]:
        """
        Evaluate whether to enter a new position.
        Returns: (should_enter, side, price, qty, reason)
        """
        self.snapshot_count += 1
        
        # Check throttle
        if self.last_entry_time:
            elapsed = time.time() - self.last_entry_time
            if elapsed < self.config.min_entry_gap_secs:
                return False, "", 0, 0, f"throttle:{elapsed:.0f}s"
        
        # Check capital
        if self.capital_used >= self.max_capital:
            return False, "", 0, 0, "max_capital"
        
        if len(self.positions) >= self.config.max_positions:
            return False, "", 0, 0, "max_positions"
        
        # Check if in final period
        in_final = secs_to_close <= self.config.final_period_secs
        
        # Get prices
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        no_bid = prices.get("best_no_bid")
        no_ask = prices.get("imp_no_ask")
        
        if not all([yes_bid, yes_ask, no_bid, no_ask]):
            return False, "", 0, 0, "incomplete_prices"
        
        # Check market quality
        quality_ok, quality_reason = self._is_market_quality_acceptable(prices)
        if not quality_ok:
            return False, "", 0, 0, quality_reason
        
        # Calculate midpoint and update history
        mid = (yes_bid + yes_ask) / 2
        self.midpoints.append(mid)
        
        # Need warmup
        if len(self.midpoints) < self.config.warmup_observations:
            return False, "", 0, 0, f"warmup:{len(self.midpoints)}/{self.config.warmup_observations}"
        
        # Calculate stats
        mean, std = self._calc_stats()
        threshold = self.config.entry_std_mult * std
        edge_from_mean = abs(mid - mean)
        
        # Check for entry signal
        if mid < mean - threshold:
            # Price dropped significantly - potential buy YES
            side = "yes"
            price = yes_ask
            edge = mean - mid
            
        elif mid > mean + threshold:
            # Price spiked - potential buy NO
            side = "no" 
            price = no_ask
            edge = mid - mean
            
        else:
            return False, "", 0, 0, f"in_range:{edge_from_mean:.1f}¢({edge_from_mean/std:.2f}σ)"
        
        # Final period: only enter if matches preferred side
        if in_final:
            if self.config.final_flatten_mode:
                if self.preferred_side and side != self.preferred_side:
                    return False, "", 0, 0, f"final_period_wrong_side"
        
        # Check minimum edge
        if edge < self.config.min_entry_edge_cents:
            return False, "", 0, 0, f"edge_too_small:{edge:.1f}¢"
        
        # Size the position using the SAME σ-scaled logic as CBB MR
        sigma_mult = edge / std if std > 0 else self.config.entry_std_mult
        qty = self._size_qty(side, int(price), sigma_mult=sigma_mult)
        if qty <= 0:
            return False, "", 0, 0, "no_capital"

        reason = (
            f"mr_entry:{edge:.1f}¢({edge/std:.2f}σ)|"
            f"mean={mean:.1f}|std={std:.1f}|"
            f"final={in_final}|qty={qty}"
        )

        
        return True, side, int(price), qty, reason
    
    def evaluate_exit(
        self,
        position: Dict,
        prices: Dict[str, Any],
        secs_to_close: int,
        context: Dict[str, Any]
    ) -> Tuple[bool, str, int, str]:
        """
        Patient + CBB-style pnl_floor protection.
        Never exit on a tiny bounce if we're still deeply underwater.
        """
        in_final = secs_to_close <= self.config.final_period_secs

        # 1. Final-period flatten (unchanged)
        if in_final and self.config.final_flatten_mode:
            if self.preferred_side and position["side"] != self.preferred_side:
                side = position["side"]
                exit_price = prices.get("best_yes_bid" if side == "yes" else "best_no_bid")
                return True, "flatten_final", exit_price, "final_flatten_wrong_side"

        # 2. Minimum hold time (we already have this — keep it)
        hold_time = (utc_now() - position["entry_time"]).total_seconds()
        if hold_time < self.config.min_hold_secs_after_entry:
            return False, "", 0, f"hold_too_soon:{int(hold_time)}s"

        yes_bid = prices.get("best_yes_bid")
        no_bid = prices.get("best_no_bid")
        if not yes_bid or not no_bid:
            return False, "", 0, "no_bid"

        mean, std = self._calc_stats()

        # === NEW: CBB-style pnl_floor guard (this is the fix you want) ===
        pnl_floor = -6.0          # ← tune this: -3 = aggressive, -6 = very patient
                                  # negative = still allowed to be underwater a bit

        if position["side"] == "yes":
            exit_price = yes_bid
            current_value = yes_bid
            pnl = current_value - position["entry_price"]

            if pnl < pnl_floor:                                 # ← key guard
                return False, "", 0, f"hold_pnl_floor:{pnl:.1f}c < {pnl_floor:.1f}c"

            # Require real reversion: at least 3–4¢ back toward mean (not just σ)
            if current_value >= mean - 5:                       # fixed cents, not σ
                return True, "take_profit", exit_price, f"reverted:pnl={pnl:.1f}¢"

        else:  # NO position
            exit_price = no_bid
            current_value = no_bid
            pnl = current_value - position["entry_price"]

            if pnl < pnl_floor:
                return False, "", 0, f"hold_pnl_floor:{pnl:.1f}c < {pnl_floor:.1f}c"

            if current_value >= mean - 3:                       # same logic for NO
                return True, "take_profit", exit_price, f"reverted:pnl={pnl:.1f}¢"

        return False, "", 0, "hold"
    def can_enter(self) -> Tuple[bool, str]:
        """Check if we can enter based on constraints"""
        if self.capital_used >= self.max_capital:
            return False, "max_capital"
        if len(self.positions) >= self.config.max_positions:
            return False, "max_positions"
        return True, "ok"
    
    # In superbowl_mr_strategy.py

    def record_entry(self, side: str, price: float, qty: int, reason: str):
        self.position_counter += 1
        pos = {
            "id": f"{self.name}_{self.position_counter}",
            "side": side,
            "entry_price": price,
            "qty": qty,
            "entry_time": utc_now(),
            "reason": reason,
        }
        self.positions.append(pos)
        self.last_entry_time = time.time()

        # Add fee
        fee_cents = calc_taker_fee(int(price), qty)
        total_cost = (price * qty + fee_cents) / 100.0
        self.capital_used += total_cost
        self.total_fees += fee_cents / 100.0

        return pos


    def record_exit(self, position: Dict, exit_type: str, exit_price: float):
        pnl_cents = (exit_price - position["entry_price"]) * position["qty"]
        fee_cents = calc_taker_fee(int(exit_price), position["qty"])
        pnl_after_fee_cents = pnl_cents - fee_cents

        closed = {
            **position,
            "exit_type": exit_type,
            "exit_price": exit_price,
            "exit_time": utc_now(),
            "pnl": pnl_after_fee_cents / 100.0,
            "fee": fee_cents / 100.0,
        }

        self.closed.append(closed)
        self.positions.remove(position)

        # Free collateral (fee already deducted on entry)
        self.capital_used -= (position["entry_price"] * position["qty"]) / 100.0
        self.capital_used = max(0.0, self.capital_used)

        self.total_fees += fee_cents / 100.0

        return closed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance stats"""
        if not self.closed:
            return {
                "strategy": self.name,
                "trades": 0,
                "net_pnl": 0,
                "fees": self.total_fees,
            }
        
        wins = [c for c in self.closed if c["pnl"] > 0]
        
        return {
            "strategy": self.name,
            "trades": len(self.closed),
            "wins": len(wins),
            "losses": len(self.closed) - len(wins),
            "win_rate": len(wins) / len(self.closed),
            "gross_pnl": sum(c["pnl"] for c in self.closed),
            "fees": self.total_fees,
            "net_pnl": sum(c["pnl"] for c in self.closed) - self.total_fees,
            "avg_pnl": sum(c["pnl"] for c in self.closed) / len(self.closed),
            "open_positions": len(self.positions),
        }


# =============================================================================
# NFL ESPN CLOCK
# =============================================================================

class NFLGameClock:
    """
    Simplified NFL clock - for Super Bowl we can hardcode the teams.
    In production, would make this more general.
    """
    
    def __init__(self):
        # Super Bowl LIX: Patriots vs Seahawks
        self.espn_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        self.game_id = None  # Will auto-detect
        self.cache_ttl = 10
        self.last_fetch = 0
        self.cached_data = None
    
    def _fetch_scoreboard(self):
        """Fetch current NFL scoreboard"""
        now = time.time()
        if self.cached_data and (now - self.last_fetch) < self.cache_ttl:
            return self.cached_data
        
        try:
            import requests
            resp = requests.get(self.espn_url, timeout=5)
            if resp.status_code == 200:
                self.cached_data = resp.json()
                self.last_fetch = now
                return self.cached_data
        except:
            pass
        
        return None
    
    def get_secs_to_game_end(self) -> Tuple[Optional[int], str]:
        """
        Returns (seconds_remaining, status_string)
        """
        data = self._fetch_scoreboard()
        if not data:
            return None, "espn_unavailable"
        
        events = data.get("events", [])
        if not events:
            return None, "no_games"
        
        # For Super Bowl, should only be one game today
        # In production, would match by team names
        game = events[0]
        
        status = game.get("status", {})
        state = status.get("type", {}).get("state", "").lower()
        
        if state == "post":
            return 0, "final"
        
        if state == "pre":
            return None, "pregame"
        
        # Parse clock
        period = int(status.get("period", 1))
        clock_str = status.get("displayClock", "15:00")
        
        try:
            parts = clock_str.split(":")
            mins = int(parts[0])
            secs = int(parts[1])
            clock_secs = mins * 60 + secs
        except:
            return None, "clock_parse_error"
        
        # NFL: 4 quarters x 15 min, OT varies
        if period == 1:
            remaining = (15 * 60 * 3) + clock_secs  # 3 quarters + current
        elif period == 2:
            remaining = (15 * 60 * 2) + clock_secs  # 2 quarters + current
        elif period == 3:
            remaining = (15 * 60 * 1) + clock_secs  # 1 quarter + current
        elif period == 4:
            remaining = clock_secs
        else:  # OT
            remaining = clock_secs
        
        return remaining, f"Q{period}:{clock_str}"


# =============================================================================
# EXAMPLE: HOW TO RUN
# =============================================================================

def run_superbowl_mr(
    ticker: str,
    market: Dict[str, Any],
    preferred_side: Optional[str] = None,
    max_capital: float = 20.0
):
    """
    Run MR strategy on Super Bowl market.
    
    Args:
        ticker: Kalshi market ticker
        market: Market object from API
        preferred_side: "yes" or "no" or None (your model's lean)
        max_capital: How much to allocate
    """
    print_status("="*80)
    print_status("SUPER BOWL LIX - MEAN REVERSION STRATEGY")
    print_status("Patriots vs Seahawks")
    print_status(f"Market: {ticker}")
    print_status(f"Allocation: ${max_capital:.2f}")
    if preferred_side:
        print_status(f"Preferred side: {preferred_side.upper()}")
    print_status("="*80)
    
    # Load credentials
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    
    if not api_key or not key_path:
        raise RuntimeError("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
    
    private_key = _load_private_key(key_path)
    
    # Initialize strategy
    config = SuperBowlMRConfig()
    strategy = SuperBowlMRStrategy(
        max_capital=max_capital,
        preferred_side=preferred_side,
        config=config
    )
    
    # Initialize NFL clock
    nfl_clock = NFLGameClock()
    
    # Get close time
    close_time = parse_iso(market["close_time"])
    
    print_status(f"Market closes at: {close_time}")
    print_status(f"Strategy config:")
    print_status(f"  Lookback: {config.lookback_window}")
    print_status(f"  Entry threshold: {config.entry_std_mult}σ, min edge {config.min_entry_edge_cents}¢")
    print_status(f"  Max positions: {config.max_positions}")
    print_status(f"  Position size: 2-{config.max_position_size} contracts")
    print_status(f"  Stop loss: {'DISABLED' if not config.use_stop_loss else 'ENABLED'}")
    print_status("")
    
    poll_interval = 3.0
    snapshot_count = 0
    
    try:
        while True:
            now = utc_now()
            kalshi_secs = (close_time - now).total_seconds()
            
            # Try to get ESPN clock
            espn_secs, espn_status = nfl_clock.get_secs_to_game_end()
            
            if espn_secs is not None:
                secs_to_close = min(kalshi_secs, espn_secs)
                clock_source = f"espn:{espn_status}"
            else:
                secs_to_close = kalshi_secs
                clock_source = "kalshi_only"
            
            if secs_to_close <= 0:
                print_status("Game ended")
                break
            
            # Fetch orderbook
            try:
                ob = fetch_orderbook(ticker)
                prices = derive_prices(ob)
                
                # Add spread to prices dict for quality check
                yes_bid = prices.get("best_yes_bid")
                yes_ask = prices.get("imp_yes_ask")
                if yes_bid and yes_ask:
                    prices["spread_cents"] = yes_ask - yes_bid
                
            except Exception as e:
                print_status(f"Orderbook error: {e}")
                time.sleep(poll_interval)
                continue
            
            snapshot_count += 1
            
            # Status every 20 ticks
            if snapshot_count % 20 == 0:
                mean, std = strategy._calc_stats() if len(strategy.midpoints) >= 5 else (50, 10)
                mid = (prices.get("best_yes_bid", 50) + prices.get("imp_yes_ask", 50)) / 2
                
                print_status(
                    f"[{snapshot_count}] "
                    f"YES {prices.get('best_yes_bid')}/{prices.get('imp_yes_ask')} | "
                    f"NO {prices.get('best_no_bid')}/{prices.get('imp_no_ask')} | "
                    f"Mid={mid:.1f} Mean={mean:.1f}±{std:.1f} | "
                    f"Open={len(strategy.positions)} | "
                    f"{int(secs_to_close)}s ({clock_source})"
                )
            
            # Check exits
            for pos in list(strategy.positions):
                should_exit, exit_type, exit_price, reason = strategy.evaluate_exit(
                    pos, prices, int(secs_to_close), {}
                )
                
                if should_exit:
                    print_status(f"EXIT {pos['side'].upper()} @ {exit_price}¢ - {reason}")
                    
                    try:
                        order_id = place_limit_sell(
                            private_key, ticker, pos["side"], exit_price, pos["qty"]
                        )
                        filled, vwap = wait_for_fill_or_timeout(
                            private_key, order_id, pos["side"], max_wait_secs=12
                        )
                        
                        if filled > 0:
                            actual_price = vwap if vwap else exit_price
                            closed = strategy.record_exit(pos, exit_type, actual_price)
                            print_status(f"  Filled @ {actual_price:.1f}¢ | P&L: {closed['pnl']:.1f}¢")
                        else:
                            print_status(f"  No fill")
                            
                    except Exception as e:
                        print_status(f"  Exit error: {e}")
            
            # Check entry
            can_enter, _ = strategy.can_enter()
            if can_enter:
                should_enter, side, price, qty, reason = strategy.evaluate_entry(
                    prices, int(secs_to_close), {}
                )
                
                if should_enter:
                    print_status(f"ENTRY {side.upper()} {qty}x @ {price}¢ - {reason}")
                    
                    try:
                        order_id = place_limit_buy(private_key, ticker, side, price, qty)
                        filled, vwap = wait_for_fill_or_timeout(
                            private_key, order_id, side, max_wait_secs=15
                        )
                        
                        if filled > 0:
                            actual_price = vwap if vwap else price
                            pos = strategy.record_entry(side, actual_price, filled, reason)
                            print_status(f"  Filled {filled}x @ {actual_price:.1f}¢")
                        else:
                            print_status(f"  No fill")
                            
                    except Exception as e:
                        print_status(f"  Entry error: {e}")
            
            time.sleep(poll_interval)
    
    except KeyboardInterrupt:
        print_status("\nInterrupted by user")
    
    # Final summary
    print_status("")
    print_status("="*80)
    print_status("FINAL SUMMARY")
    print_status("="*80)
    
    stats = strategy.get_stats()
    print_status(f"Trades: {stats['trades']}")
    print_status(f"Win rate: {stats.get('win_rate', 0):.1%}")
    print_status(f"Gross P&L: {stats.get('gross_pnl', 0):.1f}¢")
    print_status(f"Fees: {stats.get('fees', 0):.1f}¢")
    print_status(f"Net P&L: {stats['net_pnl']:.1f}¢ (${stats['net_pnl']/100:.2f})")
    print_status(f"Open positions: {stats['open_positions']}")
    
    if strategy.positions:
        print_status("\nOpen positions:")
        for pos in strategy.positions:
            print_status(f"  {pos['side'].upper()} {pos['qty']}x @ {pos['entry_price']:.1f}¢")


if __name__ == "__main__":
    # EXAMPLE: Find and run on Super Bowl ML market
    
    print("Super Bowl Mean Reversion Strategy")
    print("Finding Patriots vs Seahawks market...")
    
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    private_key = _load_private_key(key_path)
    
    # Find the market (adjust series ticker for NFL)
    # For Super Bowl, the series might be NFLSB or similar
    
    # PLACEHOLDER - you'll need to find the actual ticker
    # Option 1: Use TICKER_OVERRIDE env var
    # Option 2: Search for it programmatically
    
    ticker = os.getenv("SUPERBOWL_TICKER", "").strip()
    if not ticker:
        print("Set SUPERBOWL_TICKER environment variable to the market ticker")
        print("Example: export SUPERBOWL_TICKER=NFLSB-26FEB08-NEPAT")
        sys.exit(1)
    
    # Fetch market
    from combo_vnext import fetch_market
    market = fetch_market(private_key, ticker)
    
    # Optional: Set preferred side based on your model
    # preferred = "yes"  # If you think Patriots win
    # preferred = "no"   # If you think Seahawks win
    preferred = None  # Let MR trade both sides
    
    run_superbowl_mr(
        ticker=ticker,
        market=market,
        preferred_side=preferred,
        max_capital=20.0  # Adjust based on your risk tolerance
    )