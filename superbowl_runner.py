# superbowl_runner.py
# Production runner for Super Bowl LIX - Feb 8, 2026
# New England Patriots vs Seattle Seahawks
#
# Updated: robust fill handling, proper fee tracking, fixed preferred_side

import os
import sys
import time
from typing import Dict, Any, Optional, Tuple

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
    fetch_fills_for_order,
)

from superbowl_mr_strategy import (
    SuperBowlMRStrategy,
    SuperBowlMRConfig,
    NFLGameClock,
)

# For fee calculations
from production_strategies import calc_taker_fee

# =============================================================================
# SUPER BOWL MARKET DETAILS
# =============================================================================

SERIES_TICKER = "KXSB"
EVENT_TICKER = "KXSB-26"

MARKET_NE = "KXSB-26-NE"    # Patriots (underdog)
MARKET_SEA = "KXSB-26-SEA"  # Seahawks (favorite)

MARKETS_INFO = {
    "NE": {
        "ticker": MARKET_NE,
        "team": "New England",
        "full_name": "New England Patriots",
        "espn_code": "NE",
    },
    "SEA": {
        "ticker": MARKET_SEA,
        "team": "Seattle",
        "full_name": "Seattle Seahawks",
        "espn_code": "SEA",
    }
}

# =============================================================================
# MARKET SELECTION
# =============================================================================

def get_market_prices(private_key, ticker: str) -> Dict[str, Any]:
    try:
        resp = _get(private_key, f"/trade-api/v2/markets/{ticker}")
        market = resp.get("market", resp)
        return {
            "ticker": ticker,
            "title": market.get("title", ""),
            "yes_bid": market.get("yes_bid", 0),
            "yes_ask": market.get("yes_ask", 100),
            "volume": market.get("volume", 0),
            "open_interest": market.get("open_interest", 0),
            "liquidity": market.get("liquidity", 0),
            "status": market.get("status", "unknown"),
            "close_time": market.get("close_time", ""),
        }
    except Exception as e:
        print_status(f"Error fetching {ticker}: {e}")
        return {}


def select_market(private_key) -> Tuple[str, Dict[str, Any]]:
    print_status("Fetching Super Bowl market data...")

    ne_prices = get_market_prices(private_key, MARKET_NE)
    sea_prices = get_market_prices(private_key, MARKET_SEA)

    print_status("\nMARKET OVERVIEW:")
    print_status("=" * 70)

    print_status(f"\nNew England Patriots (KXSB-26-NE):")
    print_status(f"  Price: {ne_prices.get('yes_bid', 0)}-{ne_prices.get('yes_ask', 0)}¢")
    print_status(f"  Volume: {ne_prices.get('volume', 0):,} contracts")
    print_status(f"  OI: {ne_prices.get('open_interest', 0):,}")

    print_status(f"\nSeattle Seahawks (KXSB-26-SEA):")
    print_status(f"  Price: {sea_prices.get('yes_bid', 0)}-{sea_prices.get('yes_ask', 0)}¢")
    print_status(f"  Volume: {sea_prices.get('volume', 0):,} contracts")
    print_status(f"  OI: {sea_prices.get('open_interest', 0):,}")

    print_status("\n" + "=" * 70)

    # Default to favorite (SEA) - more liquidity and action
    chosen_ticker = MARKET_SEA
    chosen_team = "SEA"

    override = os.getenv("SB_MARKET_OVERRIDE", "").upper()
    if override in ["NE", "SEA"]:
        chosen_ticker = MARKETS_INFO[override]["ticker"]
        chosen_team = override
        print_status(f"✓ Market override: {override}")
    else:
        print_status("✓ Auto-selected: Seattle (favorite / higher liquidity)")

    info = MARKETS_INFO[chosen_team]
    print_status(f"  Trading: {info['full_name']} ({chosen_ticker})")

    # Get full market object
    resp = _get(private_key, f"/trade-api/v2/markets/{chosen_ticker}")
    market = resp.get("market", resp)

    return chosen_ticker, market


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("  SUPER BOWL LIX - MEAN REVERSION STRATEGY")
    print("  Patriots vs Seahawks — February 8, 2026")
    print("=" * 80)

    # Credentials
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()

    if not api_key or not key_path:
        print("\n✗ Missing KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH")
        return 1

    try:
        private_key = _load_private_key(key_path)
        print("\n✓ Private key loaded")
    except Exception as e:
        print(f"\n✗ Key loading failed: {e}")
        return 1

    # Balance check
    try:
        resp = _get(private_key, "/trade-api/v2/portfolio/balance")
        balance = int(resp.get("balance", 0)) / 100
        print(f"✓ Account balance: ${balance:,.2f}")
    except Exception as e:
        print(f"\n✗ Balance check failed: {e}")
        return 1

    # Select market
    try:
        ticker, market = select_market(private_key)
    except Exception as e:
        print(f"\n✗ Market selection failed: {e}")
        return 1

    # Allocation & side preference
    max_capital = float(os.getenv("SB_MAX_CAPITAL", "75.0"))

    # Preferred side: read from env (yes / no / empty=None)
    preferred_side = os.getenv("SB_PREFERRED_SIDE", "").strip().lower()
    if preferred_side not in ("yes", "no", ""):
        preferred_side = None
    elif preferred_side == "":
        preferred_side = None

    print("\nSTRATEGY CONFIGURATION:")
    print(f"  Allocation:       ${max_capital:,.2f}")
    print(f"  Market:           {ticker}")
    print(f"  Preferred side:   {preferred_side.upper() if preferred_side else 'BOTH'}")

    # Strategy config overrides via env
    config = SuperBowlMRConfig()

    if os.getenv("SB_ENTRY_THRESHOLD"):
        config.entry_std_mult = float(os.getenv("SB_ENTRY_THRESHOLD"))
    if os.getenv("SB_MIN_EDGE"):
        config.min_entry_edge_cents = int(os.getenv("SB_MIN_EDGE"))
    if os.getenv("SB_LOOKBACK"):
        config.lookback_window = int(os.getenv("SB_LOOKBACK"))

    print(f"  Entry threshold:  {config.entry_std_mult:.1f}σ")
    print(f"  Min edge:         {config.min_entry_edge_cents}¢")
    print(f"  Lookback:         {config.lookback_window} ticks")

    # Pre-flight checks
    print("\nPRE-FLIGHT CHECKS:")
    print(f"  • fill_price_fix imported?     {'fill_price_fix' in sys.modules}")
    print(f"  • calc_taker_fee available?     {calc_taker_fee is not None}")
    print(f"  • max_capital set to           ${max_capital:.2f}")
    print(f"  • preferred_side               {preferred_side!r}")

    print("\n" + "=" * 80)
    input("Press ENTER to begin trading (Ctrl+C to cancel)...\n")
    print("=" * 80 + "\n")

    # Initialize strategy
    strategy = SuperBowlMRStrategy(
        max_capital=max_capital,
        preferred_side=preferred_side,
        config=config,
    )

    nfl_clock = NFLGameClock()
    close_time = parse_iso(market["close_time"])

    print_status(f"Market closes at: {close_time}")
    print_status("Trading loop started...\n")

    poll_interval = 3.0
    snapshot_count = 0

    try:
        while True:
            now = utc_now()
            kalshi_secs = (close_time - now).total_seconds()

            espn_secs, espn_status = nfl_clock.get_secs_to_game_end()
            if espn_secs is not None:
                secs_to_close = min(kalshi_secs, espn_secs)
                clock_source = f"espn:{espn_status}"
            else:
                secs_to_close = kalshi_secs
                clock_source = "kalshi"

            if secs_to_close <= 0:
                print_status("\n🏈 Game over")
                break

            # Get orderbook
            try:
                ob = fetch_orderbook(ticker)
                prices = derive_prices(ob)

                yes_bid = prices.get("best_yes_bid")
                yes_ask = prices.get("imp_yes_ask")
                if yes_bid is not None and yes_ask is not None:
                    prices["spread_cents"] = yes_ask - yes_bid

            except Exception as e:
                print_status(f"Orderbook fetch failed: {e}")
                time.sleep(poll_interval)
                continue

            snapshot_count += 1

            # Periodic status
            if snapshot_count % 30 == 0:
                realized = sum(c["pnl"] for c in strategy.closed)

                unrealized = 0.0
                yes_bid = prices.get("best_yes_bid")
                no_bid = prices.get("best_no_bid")

                for pos in strategy.positions:
                    if pos["side"] == "yes" and yes_bid:
                        gross = (yes_bid - pos["entry_price"]) * pos["qty"]
                        fee_est = calc_taker_fee(int(yes_bid), pos["qty"])
                        unrealized += (gross - fee_est)
                    elif pos["side"] == "no" and no_bid:
                        gross = (no_bid - pos["entry_price"]) * pos["qty"]
                        fee_est = calc_taker_fee(int(no_bid), pos["qty"])
                        unrealized += (gross - fee_est)

                total_pnl = realized + unrealized

                yes_qty = sum(p["qty"] for p in strategy.positions if p["side"] == "yes")
                no_qty = sum(p["qty"] for p in strategy.positions if p["side"] == "no")

                pref = strategy.preferred_side.upper() if strategy.preferred_side else ""

                print_status(
                    f"[SB] Y:{yes_bid}/{yes_ask} | "
                    f"secs:{int(secs_to_close)} {clock_source} | "
                    f"open:{len(strategy.positions)} | "
                    f"inv YES:{yes_qty} NO:{no_qty} pref:{pref} | "
                    f"PnL R:{realized:.1f}c U:{unrealized:.1f}c T:{total_pnl:.1f}c "
                    f"(${total_pnl/100:.2f})"
                )

            # === EXITS ===
            for pos in list(strategy.positions):
                should_exit, exit_type, exit_price, reason = strategy.evaluate_exit(
                    pos, prices, int(secs_to_close), {}
                )

                if should_exit:
                    print_status(f"[SB][mr] EXIT {pos['side'].upper()} @ {exit_price}¢ — {reason}")

                    try:
                        order_id = place_limit_sell(
                            private_key, ticker, pos["side"], exit_price, pos["qty"]
                        )
                        filled, vwap = wait_for_fill_or_timeout(
                            private_key, order_id, pos["side"], max_wait_secs=12
                        )

                        if filled > 0:
                            actual_price = vwap

                            if actual_price is None:
                                print_status("[exit] VWAP None → using robust fill extraction")
                                try:
                                    from fill_price_fix import calculate_vwap_robust
                                    fills = fetch_fills_for_order(private_key, order_id)
                                    if not fills:
                                        print_status("[exit] CRITICAL: No exit fills returned")
                                        continue

                                    actual_price = calculate_vwap_robust(fills, pos["side"])
                                    if actual_price is None:
                                        print_status("[exit] CRITICAL: Could not extract exit price")
                                        if fills:
                                            print_status(f"[exit DEBUG] Sample fill: {fills[0]}")
                                        continue

                                    print_status(f"[exit] Robust price: {actual_price:.2f}¢")

                                except Exception as e:
                                    print_status(f"[exit] Robust extraction failed: {e}")
                                    actual_price = exit_price  # fallback

                            closed = strategy.record_exit(pos, exit_type, actual_price)
                            print_status(f"[SB][mr] CLOSED: net={closed['pnl']:.2f}c")
                        else:
                            print_status(f"[SB][mr] EXIT NO FILL")

                    except Exception as e:
                        print_status(f"[SB][mr] EXIT ERROR: {e}")
                        import traceback
                        traceback.print_exc()

            # === ENTRIES ===
            can_enter, _ = strategy.can_enter()
            if can_enter:
                should_enter, side, price, qty, reason = strategy.evaluate_entry(
                    prices, int(secs_to_close), {}
                )

                if should_enter:
                    print_status(f"[SB][mr] ENTRY {side.upper()} {qty}x @ {price}¢ — {reason}")

                    try:
                        order_id = place_limit_buy(private_key, ticker, side, price, qty)
                        filled, vwap = wait_for_fill_or_timeout(
                            private_key, order_id, side, max_wait_secs=15
                        )

                        if filled > 0:
                            actual_price = vwap

                            if actual_price is None:
                                print_status("[entry] VWAP None → using robust fill extraction")
                                try:
                                    from fill_price_fix import calculate_vwap_robust
                                    fills = fetch_fills_for_order(private_key, order_id)
                                    if not fills:
                                        print_status("[entry] CRITICAL: No fills returned")
                                        continue

                                    actual_price = calculate_vwap_robust(fills, side)
                                    if actual_price is None:
                                        print_status("[entry] CRITICAL: Robust extraction failed")
                                        if fills:
                                            print_status(f"[entry DEBUG] Sample fill: {fills[0]}")
                                        continue

                                    print_status(f"[entry] Robust price: {actual_price:.2f}¢")

                                except Exception as e:
                                    print_status(f"[entry] Robust extraction failed: {e}")
                                    actual_price = price  # last resort

                            pos = strategy.record_entry(side, actual_price, filled, reason)
                            print_status(f"[SB][mr] FILLED {filled}x @ {actual_price:.2f}¢")
                        else:
                            print_status(f"[SB][mr] ENTRY NO FILL")

                    except Exception as e:
                        print_status(f"[SB][mr] ENTRY ERROR: {e}")
                        import traceback
                        traceback.print_exc()

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print_status("\nInterrupted by user")

    # Final summary
    print_status("\n" + "=" * 60)
    print_status("[SuperBowl] === FINAL SUMMARY ===")
    print_status("=" * 60)

    stats = strategy.get_stats()

    print_status(
        f"[SB][mr] "
        f"Trades: {stats['trades']}   "
        f"W:{stats.get('wins', 0)} / L:{stats.get('losses', 0)}   "
        f"Net: {stats['net_pnl']:.1f}¢ (${stats['net_pnl']/100:.2f})   "
        f"Fees: {stats['fees']:.1f}¢"
    )

    if strategy.positions:
        print_status(f"Open positions remaining: {len(strategy.positions)} (will settle at outcome)")

    print_status(f"Log files: logs/superbowl_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())