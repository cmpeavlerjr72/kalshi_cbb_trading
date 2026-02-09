# fill_price_fix.py
# Improved fill price extraction for Kalshi trades
# 
# This fixes the VWAP calculation bug where fill prices weren't being found

from typing import List, Dict, Any, Optional

def extract_fill_price_robust(fill_obj: Dict[str, Any], side: str) -> Optional[float]:
    """
    Robustly extract fill price from Kalshi fill object.
    
    Tries multiple field names in order of preference:
    1. Side-specific fields (yes_price, no_price)
    2. Generic fields (price, fill_price, executed_price)
    3. Fallback calculations from other fields
    
    Returns price in cents, or None if not found.
    """
    # Try side-specific first
    if side == "yes":
        for field in ['yes_price', 'yesPrice']:
            if field in fill_obj and fill_obj[field] is not None:
                try:
                    return float(fill_obj[field])
                except (TypeError, ValueError):
                    pass
    elif side == "no":
        for field in ['no_price', 'noPrice']:
            if field in fill_obj and fill_obj[field] is not None:
                try:
                    return float(fill_obj[field])
                except (TypeError, ValueError):
                    pass
    
    # Try generic fields
    for field in ['price', 'fill_price', 'fillPrice', 'executed_price', 'executedPrice']:
        if field in fill_obj and fill_obj[field] is not None:
            try:
                return float(fill_obj[field])
            except (TypeError, ValueError):
                pass
    
    # Last resort: try to calculate from cost/count
    # For YES: cost = price * count
    # For NO: cost = (100 - price) * count (inverse)
    if 'cost' in fill_obj and 'count' in fill_obj:
        try:
            cost = float(fill_obj['cost'])
            count = int(fill_obj['count'])
            if count > 0:
                # Cost is in cents total
                price_per = cost / count
                # For NO side, Kalshi might store the complement
                # We'll return the raw value and let caller handle it
                return price_per
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    
    return None


def calculate_vwap_robust(fills: List[Dict[str, Any]], side: str) -> Optional[float]:
    """
    Calculate VWAP from fill objects with robust price extraction.
    
    Returns VWAP in cents, or None if no valid fills found.
    """
    total_qty = 0
    total_cost = 0.0
    
    for fill in fills:
        try:
            qty = int(fill.get('count', 0))
            if qty <= 0:
                continue
            
            price = extract_fill_price_robust(fill, side)
            if price is None:
                print(f"WARNING: Could not extract price from fill: {fill}")
                continue
            
            total_qty += qty
            total_cost += price * qty
            
        except Exception as e:
            print(f"ERROR parsing fill: {e}")
            continue
    
    if total_qty == 0:
        return None
    
    return total_cost / total_qty


# Diagnostic helper
def diagnose_fill_object(fill: Dict[str, Any], side: str) -> str:
    """Return diagnostic string showing what fields are present and their values"""
    lines = ["\n=== FILL OBJECT DIAGNOSTIC ==="]
    lines.append(f"Side: {side}")
    lines.append(f"All keys: {list(fill.keys())}")
    lines.append("\nPrice-related fields:")
    
    price_fields = [
        'yes_price', 'yesPrice', 'no_price', 'noPrice',
        'price', 'fill_price', 'fillPrice', 
        'executed_price', 'executedPrice',
        'cost', 'count'
    ]
    
    for field in price_fields:
        if field in fill:
            lines.append(f"  {field}: {fill[field]}")
    
    lines.append(f"\nExtracted price: {extract_fill_price_robust(fill, side)}")
    lines.append("=" * 30)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with different fill formats
    
    # Format 1: Standard
    fill1 = {'count': 80, 'no_price': 26.31}
    print(f"Test 1: {calculate_vwap_robust([fill1], 'no')}")
    
    # Format 2: Generic price field
    fill2 = {'count': 80, 'price': 25.61}
    print(f"Test 2: {calculate_vwap_robust([fill2], 'no')}")
    
    # Format 3: Cost-based
    fill3 = {'count': 80, 'cost': 2104.8}  # 80 * 26.31 = 2104.8
    print(f"Test 3: {calculate_vwap_robust([fill3], 'no')}")
    
    # Diagnostic
    print(diagnose_fill_object(fill1, 'no'))