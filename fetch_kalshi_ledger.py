#!/usr/bin/env python3
# fetch_kalshi_ledger.py
# Pull your ACTUAL trading ledger from Kalshi API
# Use this to verify bot logs, check fills, and reconcile P&L

import os
import json
import csv
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Import from combo_vnext for auth
from combo_vnext import _load_private_key, _get

# =============================================================================
# API ENDPOINTS
# =============================================================================

def fetch_fills(
    private_key,
    ticker: Optional[str] = None,
    min_ts: Optional[str] = None,
    max_ts: Optional[str] = None,
    limit: int = 500
) -> List[Dict[str, Any]]:
    """
    Fetch fill history from Kalshi.
    
    Args:
        ticker: Filter by market ticker (e.g., "KXNCAAMBGAME-26FEB05FAIR-FAIR")
        min_ts: Minimum timestamp (Unix milliseconds as string, e.g., "1738800000000")
        max_ts: Maximum timestamp (Unix milliseconds as string, e.g., "1738886400000")
        limit: Max results per page (up to 1000)
    
    Returns:
        List of fill records
    """
    fills = []
    cursor = None
    
    while True:
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        
        resp = _get(private_key, "/trade-api/v2/portfolio/fills", params=params)
        
        batch = resp.get("fills", []) or []
        fills.extend(batch)
        
        cursor = resp.get("cursor")
        if not cursor:
            break
        
        print(f"Fetched {len(fills)} fills so far...")
    
    return fills


def fetch_orders(
    private_key,
    ticker: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 500
) -> List[Dict[str, Any]]:
    """
    Fetch order history from Kalshi.
    
    Args:
        ticker: Filter by market ticker
        status: Filter by status ("resting", "canceled", "executed")
        limit: Max results per page
    
    Returns:
        List of order records
    """
    orders = []
    cursor = None
    
    while True:
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        
        resp = _get(private_key, "/trade-api/v2/portfolio/orders", params=params)
        
        batch = resp.get("orders", []) or []
        orders.extend(batch)
        
        cursor = resp.get("cursor")
        if not cursor:
            break
        
        print(f"Fetched {len(orders)} orders so far...")
    
    return orders


def fetch_positions(private_key) -> List[Dict[str, Any]]:
    """Fetch current open positions"""
    resp = _get(private_key, "/trade-api/v2/portfolio/positions")
    return resp.get("positions", []) or []


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def calculate_fill_fee(fill: Dict[str, Any]) -> float:
    """
    Calculate fee for a fill in cents.
    Kalshi charges: 0.07 × C × P × (1-P), max 2¢
    """
    count = int(fill.get("count", 0))
    
    # Try to get price
    side = fill.get("side", "")
    if side == "yes":
        price_cents = fill.get("yes_price")
    elif side == "no":
        price_cents = fill.get("no_price")
    else:
        price_cents = fill.get("price")
    
    if price_cents is None:
        return 0.0
    
    try:
        price_cents = float(price_cents)
    except (TypeError, ValueError):
        return 0.0
    
    p = price_cents / 100.0
    fee_per_contract = min(0.07 * p * (1 - p), 0.02)
    
    return fee_per_contract * count * 100  # Return in cents


def group_fills_by_order(fills: List[Dict]) -> Dict[str, List[Dict]]:
    """Group fills by order_id"""
    grouped = {}
    for fill in fills:
        oid = fill.get("order_id", "unknown")
        if oid not in grouped:
            grouped[oid] = []
        grouped[oid].append(fill)
    return grouped


def summarize_fills(fills: List[Dict]) -> Dict[str, Any]:
    """Generate summary statistics from fills"""
    if not fills:
        return {
            "total_fills": 0,
            "total_contracts": 0,
            "buy_contracts": 0,
            "sell_contracts": 0,
            "estimated_fees_cents": 0.0,
        }
    
    total_contracts = 0
    buy_contracts = 0
    sell_contracts = 0
    total_fees = 0.0
    
    for fill in fills:
        count = int(fill.get("count", 0))
        action = fill.get("action", "")
        
        total_contracts += count
        if action == "buy":
            buy_contracts += count
        else:
            sell_contracts += count
        
        total_fees += calculate_fill_fee(fill)
    
    return {
        "total_fills": len(fills),
        "total_contracts": total_contracts,
        "buy_contracts": buy_contracts,
        "sell_contracts": sell_contracts,
        "estimated_fees_cents": total_fees,
    }


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_fills_csv(fills: List[Dict], filename: str):
    """Export fills to CSV"""
    if not fills:
        print(f"No fills to export")
        return
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "created_time",
            "ticker",
            "order_id",
            "action",
            "side",
            "count",
            "yes_price",
            "no_price",
            "price",
            "fee_cents",
            "trade_id",
        ])
        
        # Rows
        for fill in fills:
            fee = calculate_fill_fee(fill)
            writer.writerow([
                fill.get("created_time", ""),
                fill.get("ticker", ""),
                fill.get("order_id", ""),
                fill.get("action", ""),
                fill.get("side", ""),
                fill.get("count", ""),
                fill.get("yes_price", ""),
                fill.get("no_price", ""),
                fill.get("price", ""),
                f"{fee:.2f}",
                fill.get("trade_id", ""),
            ])
    
    print(f"✓ Exported {len(fills)} fills to {filename}")


def export_orders_csv(orders: List[Dict], filename: str):
    """Export orders to CSV"""
    if not orders:
        print(f"No orders to export")
        return
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "created_time",
            "ticker",
            "order_id",
            "action",
            "side",
            "type",
            "status",
            "count",
            "yes_price",
            "no_price",
            "remaining_count",
            "expiration_time",
        ])
        
        # Rows
        for order in orders:
            writer.writerow([
                order.get("created_time", ""),
                order.get("ticker", ""),
                order.get("order_id", ""),
                order.get("action", ""),
                order.get("side", ""),
                order.get("type", ""),
                order.get("status", ""),
                order.get("count", ""),
                order.get("yes_price", ""),
                order.get("no_price", ""),
                order.get("remaining_count", ""),
                order.get("expiration_time", ""),
            ])
    
    print(f"✓ Exported {len(orders)} orders to {filename}")


def export_positions_csv(positions: List[Dict], filename: str):
    """Export current positions to CSV"""
    if not positions:
        print(f"No positions to export")
        return
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "ticker",
            "position_id",
            "side",
            "count",
            "total_cost",
            "fees_paid",
            "resting_order_count",
        ])
        
        # Rows
        for pos in positions:
            writer.writerow([
                pos.get("ticker", ""),
                pos.get("position_id", ""),
                pos.get("side", ""),
                pos.get("count", ""),
                pos.get("total_cost", ""),
                pos.get("fees_paid", ""),
                pos.get("resting_order_count", ""),
            ])
    
    print(f"✓ Exported {len(positions)} positions to {filename}")


# =============================================================================
# RECONCILIATION
# =============================================================================

def reconcile_with_bot_logs(
    fills: List[Dict],
    bot_trades_csv: str
) -> Dict[str, Any]:
    """
    Compare Kalshi fills with bot's trade log.
    
    Returns dict with:
    - matched_trades
    - missing_in_bot
    - missing_in_kalshi
    - discrepancies
    """
    # Load bot trades
    bot_trades = []
    try:
        with open(bot_trades_csv, "r") as f:
            reader = csv.DictReader(f)
            bot_trades = list(reader)
    except FileNotFoundError:
        print(f"⚠️  Bot log not found: {bot_trades_csv}")
        return {}
    
    print(f"\nReconciliation:")
    print(f"  Kalshi fills: {len(fills)}")
    print(f"  Bot trades: {len(bot_trades)}")
    
    # TODO: Implement detailed matching logic
    # This would compare timestamps, tickers, sides, quantities, prices
    
    return {
        "kalshi_count": len(fills),
        "bot_count": len(bot_trades),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  KALSHI LEDGER FETCHER")
    print("  Pull your actual trading history from Kalshi API")
    print("="*70 + "\n")
    
    # Load credentials
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    
    if not api_key or not key_path:
        print("ERROR: Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        return
    
    private_key = _load_private_key(key_path)
    
    # Ask for filters
    print("Filter options (press Enter to skip):")
    ticker = input("  Ticker (e.g., KXNCAAMBGAME-26FEB05FAIR-FAIR): ").strip()
    
    # Time range
    print("\n  Time range:")
    print("    Enter Unix timestamp in milliseconds (e.g., 1738800000000)")
    print("    Or press Enter to skip")
    min_ts_input = input("    Start time (empty = all): ").strip()
    max_ts_input = input("    End time (empty = now): ").strip()
    
    min_ts = min_ts_input if min_ts_input else None
    max_ts = max_ts_input if max_ts_input else None
    
    if not ticker:
        ticker = None
    
    print("\n" + "-"*70)
    print("Fetching data from Kalshi...")
    print("-"*70 + "\n")
    
    # Fetch fills
    print("📋 Fetching fills...")
    fills = fetch_fills(
        private_key,
        ticker=ticker,
        min_ts=min_ts,
        max_ts=max_ts
    )
    
    # Fetch orders
    print("\n📋 Fetching orders...")
    orders = fetch_orders(private_key, ticker=ticker)
    
    # Fetch positions
    print("\n📋 Fetching current positions...")
    positions = fetch_positions(private_key)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    fill_summary = summarize_fills(fills)
    print(f"\nFills:")
    print(f"  Total: {fill_summary['total_fills']}")
    print(f"  Contracts traded: {fill_summary['total_contracts']}")
    print(f"    Buys: {fill_summary['buy_contracts']}")
    print(f"    Sells: {fill_summary['sell_contracts']}")
    print(f"  Estimated fees: {fill_summary['estimated_fees_cents']:.2f}¢")
    
    print(f"\nOrders:")
    print(f"  Total: {len(orders)}")
    by_status = {}
    for o in orders:
        status = o.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    for status, count in sorted(by_status.items()):
        print(f"    {status}: {count}")
    
    print(f"\nOpen Positions:")
    print(f"  Total: {len(positions)}")
    for pos in positions:
        ticker = pos.get("ticker", "")
        side = pos.get("side", "")
        count = pos.get("count", 0)
        print(f"    {ticker}: {side.upper()} {count}x")
    
    # Export
    print("\n" + "="*70)
    print("EXPORT")
    print("="*70 + "\n")
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    fills_file = f"kalshi_fills_{ts}.csv"
    orders_file = f"kalshi_orders_{ts}.csv"
    positions_file = f"kalshi_positions_{ts}.csv"
    
    export_fills_csv(fills, fills_file)
    export_orders_csv(orders, orders_file)
    export_positions_csv(positions, positions_file)
    
    # Also export full JSON for debugging
    json_file = f"kalshi_ledger_{ts}.json"
    with open(json_file, "w") as f:
        json.dump({
            "fills": fills,
            "orders": orders,
            "positions": positions,
            "summary": fill_summary,
        }, f, indent=2)
    print(f"✓ Exported full data to {json_file}")
    
    print("\n" + "="*70)
    print(f"✓ Complete! Check the exported files.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()