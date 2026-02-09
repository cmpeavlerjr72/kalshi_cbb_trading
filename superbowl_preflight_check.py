#!/usr/bin/env python3
"""
superbowl_preflight_check.py

Pre-flight validation for Super Bowl LIX mean reversion trading
New England Patriots vs Seattle Seahawks - February 8, 2026

Run this BEFORE starting superbowl_runner.py to verify:
- Environment setup
- API authentication
- Market availability
- NFLGameClock functionality
- Strategy configuration
- File syntax
"""

import os
import sys
import ast
import time
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CHECK FUNCTIONS
# =============================================================================

def check_env_vars() -> bool:
    """Check required environment variables"""
    print("\n1. Checking environment variables...")
    
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    env = os.getenv("KALSHI_ENV", "DEMO").upper()
    
    if not api_key:
        print("   ✗ KALSHI_API_KEY_ID not set")
        return False
    print(f"   ✓ KALSHI_API_KEY_ID: {api_key[:8]}...")
    
    if not key_path:
        print("   ✗ KALSHI_PRIVATE_KEY_PATH not set")
        return False
    print(f"   ✓ KALSHI_PRIVATE_KEY_PATH: {key_path}")
    
    if not os.path.exists(key_path):
        print(f"   ✗ Private key file does not exist: {key_path}")
        return False
    print(f"   ✓ Private key file exists")
    
    print(f"   ✓ KALSHI_ENV: {env}")
    
    if env != "PROD":
        print("   ⚠  WARNING: Running in DEMO mode (change KALSHI_ENV=PROD for real trading)")
    
    return True


def check_private_key() -> bool:
    """Verify private key can be loaded"""
    print("\n2. Checking private key...")
    
    try:
        from combo_vnext import _load_private_key
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        private_key = _load_private_key(key_path)
        print("   ✓ Private key loaded successfully")
        print(f"   ✓ Key type: {type(private_key).__name__}")
        return True
    except Exception as e:
        print(f"   ✗ Failed to load private key: {e}")
        return False


def check_api_connection() -> bool:
    """Test API connection and authentication"""
    print("\n3. Testing API connection...")
    
    try:
        from combo_vnext import _load_private_key, _get
        
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        private_key = _load_private_key(key_path)
        
        # Test authentication
        resp = _get(private_key, "/trade-api/v2/portfolio/balance")
        balance = int(resp.get("balance", 0)) / 100
        
        print(f"   ✓ API authentication successful")
        print(f"   ✓ Account balance: ${balance:.2f}")
        
        # Check if balance is sufficient
        min_recommended = 20.0  # For $20 allocation
        if balance < min_recommended:
            print(f"   ⚠  WARNING: Balance ${balance:.2f} < Recommended ${min_recommended:.2f}")
            print(f"      Consider reducing SB_MAX_CAPITAL")
        else:
            print(f"   ✓ Sufficient balance for trading")
        
        return True
        
    except Exception as e:
        print(f"   ✗ API connection failed: {e}")
        return False


def check_dependencies() -> bool:
    """Check required Python packages"""
    print("\n4. Checking dependencies...")
    
    required = {
        "requests": "requests",
        "cryptography": "cryptography", 
        "python-dotenv": "dotenv",  # Import name differs from package name
    }
    
    missing = []
    for pkg_name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"   ✓ {pkg_name}")
        except ImportError:
            print(f"   ✗ {pkg_name} not installed")
            missing.append(pkg_name)
    
    if missing:
        print(f"\n   Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


def check_files() -> bool:
    """Check required code files exist and have valid syntax"""
    print("\n5. Checking code files...")
    
    required_files = {
        "superbowl_runner.py": "Main runner (entry point)",
        "superbowl_mr_strategy.py": "ML mean reversion strategy",
        "combo_vnext.py": "Kalshi API helpers",
    }
    
    all_ok = True
    
    for filename, description in required_files.items():
        if not os.path.exists(filename):
            print(f"   ✗ {filename} missing - {description}")
            all_ok = False
            continue
        
        print(f"   ✓ {filename} exists")
        
        # Check syntax (try UTF-8, fall back to latin-1 for Windows compatibility)
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # Fallback for Windows files with mixed encoding
                with open(filename, 'r', encoding='latin-1') as f:
                    content = f.read()
                print(f"     ⚠  File uses non-UTF-8 encoding (may have display issues)")
            except Exception as e:
                print(f"     ✗ Could not read file: {e}")
                all_ok = False
                continue
        except Exception as e:
            print(f"     ✗ Could not read file: {e}")
            all_ok = False
            continue
        
        try:
            ast.parse(content)
            print(f"     ✓ Syntax valid")
        except SyntaxError as e:
            print(f"     ✗ Syntax error at line {e.lineno}: {e.msg}")
            all_ok = False
    
    # Warn about corrupted spread file if it exists
    if os.path.exists("superbowl_spread_mr.py"):
        print(f"\n   ⚠  superbowl_spread_mr.py detected")
        try:
            # Try UTF-8 first, fall back to latin-1
            try:
                with open("superbowl_spread_mr.py", 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open("superbowl_spread_mr.py", 'r', encoding='latin-1') as f:
                    content = f.read()
            
            ast.parse(content)
            print(f"     ✓ Syntax valid (spread mode available)")
        except SyntaxError:
            print(f"     ✗ Syntax invalid (CORRUPTED - do not use spread mode)")
            print(f"     → Stick with ML mode (default)")
    
    return all_ok


def check_superbowl_markets() -> bool:
    """Check if Super Bowl markets are available"""
    print("\n6. Checking Super Bowl markets...")
    
    try:
        from combo_vnext import _load_private_key, _get
        
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        private_key = _load_private_key(key_path)
        
        # Expected markets
        markets_to_check = {
            "KXSB-26-SEA": "Seattle Seahawks",
            "KXSB-26-NE": "New England Patriots",
        }
        
        found_markets = {}
        
        for ticker, team in markets_to_check.items():
            try:
                resp = _get(private_key, f"/trade-api/v2/markets/{ticker}")
                market = resp.get("market", resp)
                
                yes_bid = market.get("yes_bid", 0)
                yes_ask = market.get("yes_ask", 100)
                volume = market.get("volume", 0)
                liquidity = market.get("liquidity", 0)
                status = market.get("status", "unknown")
                
                print(f"   ✓ {ticker} ({team})")
                print(f"     Price: {yes_bid}-{yes_ask}¢")
                print(f"     Volume: {volume:,} contracts")
                print(f"     Liquidity: ${liquidity/100:,.0f}")
                print(f"     Status: {status}")
                
                if status != "active":
                    print(f"     ⚠  Market is not active (status={status})")
                
                if yes_ask - yes_bid > 5:
                    print(f"     ⚠  Wide spread: {yes_ask - yes_bid}¢")
                
                found_markets[ticker] = market
                
            except Exception as e:
                print(f"   ✗ {ticker} ({team}) - Error: {e}")
                return False
        
        # Check if at least one market is good
        if len(found_markets) > 0:
            print(f"\n   ✓ Found {len(found_markets)}/2 Super Bowl markets")
            return True
        else:
            print(f"\n   ✗ No Super Bowl markets found")
            return False
            
    except Exception as e:
        print(f"   ✗ Market check failed: {e}")
        return False


def check_orderbook_access() -> bool:
    """Test orderbook fetch (unauthenticated endpoint)"""
    print("\n7. Testing orderbook access...")
    
    try:
        from combo_vnext import fetch_orderbook, derive_prices
        
        # Try to fetch Seattle market orderbook
        ticker = "KXSB-26-SEA"
        
        start = time.time()
        ob = fetch_orderbook(ticker)
        latency = (time.time() - start) * 1000
        
        print(f"   ✓ Orderbook fetch successful")
        print(f"   ✓ Latency: {latency:.1f}ms")
        
        if latency > 200:
            print(f"     ⚠  High latency (>{latency:.0f}ms) - consider running from cloud")
        
        # Parse prices
        prices = derive_prices(ob)
        yes_bid = prices.get("best_yes_bid")
        yes_ask = prices.get("imp_yes_ask")
        
        if yes_bid and yes_ask:
            spread = yes_ask - yes_bid
            print(f"   ✓ YES bid/ask: {yes_bid}/{yes_ask}¢ (spread: {spread}¢)")
            
            if spread == 1:
                print(f"     ⚠  1¢ spread detected - fees will be ~40-50% of edge")
                print(f"     → This is expected for Super Bowl (highest liquidity)")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Orderbook access failed: {e}")
        return False


def check_nfl_clock() -> bool:
    """Test NFLGameClock functionality"""
    print("\n8. Testing NFL game clock...")
    
    try:
        from superbowl_mr_strategy import NFLGameClock
        
        clock = NFLGameClock()
        
        print(f"   ✓ NFLGameClock imported successfully")
        
        # Try to get current status
        secs, status = clock.get_secs_to_game_end()
        
        print(f"   ✓ Clock query successful")
        print(f"     Status: {status}")
        
        if secs is not None:
            mins = int(secs / 60)
            print(f"     Time remaining: {mins}m {int(secs % 60)}s")
            
            if "pregame" in status.lower():
                print(f"     ℹ️  Game hasn't started yet")
            elif "final" in status.lower():
                print(f"     ⚠  Game is over")
            else:
                print(f"     ✓ Game is live")
        else:
            if "pregame" in status.lower():
                print(f"     ℹ️  Game hasn't started (expected)")
            else:
                print(f"     ⚠  Could not determine game time: {status}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ NFLGameClock test failed: {e}")
        print(f"     → Will fall back to Kalshi close_time only")
        return True  # Non-critical, return True


def check_configuration() -> bool:
    """Validate configuration parameters"""
    print("\n9. Checking configuration...")
    
    # Get config from environment
    max_capital = float(os.getenv("SB_MAX_CAPITAL", "20.0"))
    entry_threshold = float(os.getenv("SB_ENTRY_THRESHOLD", "2.0"))
    min_edge = int(os.getenv("SB_MIN_EDGE", "8"))
    lookback = int(os.getenv("SB_LOOKBACK", "40"))
    market_override = os.getenv("SB_MARKET_OVERRIDE", "").upper()
    preferred_side = os.getenv("SB_PREFERRED_SIDE", "").lower()
    
    print(f"   Configuration:")
    print(f"     SB_MAX_CAPITAL: ${max_capital:.2f}")
    print(f"     SB_ENTRY_THRESHOLD: {entry_threshold}σ")
    print(f"     SB_MIN_EDGE: {min_edge}¢")
    print(f"     SB_LOOKBACK: {lookback} ticks")
    
    if market_override:
        print(f"     SB_MARKET_OVERRIDE: {market_override}")
    else:
        print(f"     SB_MARKET_OVERRIDE: (auto-select Seattle)")
    
    if preferred_side:
        print(f"     SB_PREFERRED_SIDE: {preferred_side}")
    else:
        print(f"     SB_PREFERRED_SIDE: (both directions)")
    
    # Validation
    warnings = []
    
    if max_capital < 5.0:
        warnings.append("Very small allocation (<$5) - may not have enough capital for entries")
    
    if max_capital > 100.0:
        warnings.append("Large allocation (>$100) - verify this is intentional")
    
    if entry_threshold < 1.5:
        warnings.append("Low entry threshold (<1.5σ) - may overfit to noise")
    
    if entry_threshold > 3.0:
        warnings.append("High entry threshold (>3.0σ) - may miss opportunities")
    
    if min_edge < 5:
        warnings.append("Low min edge (<5¢) - fees may eat profits")
    
    if lookback < 20:
        warnings.append("Small lookback (<20) - statistics may be unreliable")
    
    if lookback > 60:
        warnings.append("Large lookback (>60) - may be slow to adapt")
    
    if warnings:
        print(f"\n   ⚠  Configuration warnings:")
        for w in warnings:
            print(f"     - {w}")
        print(f"\n   ℹ️  These are warnings, not errors. Review if needed.")
    else:
        print(f"\n   ✓ Configuration looks reasonable")
    
    return True


def check_directories() -> bool:
    """Check/create output directories"""
    print("\n10. Checking output directories...")
    
    dirs = ["logs"]
    
    for dirname in dirs:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print(f"   ✓ Created {dirname}/")
        else:
            print(f"   ✓ {dirname}/ exists")
    
    return True


def estimate_fees() -> bool:
    """Estimate trading fees for the configuration"""
    print("\n11. Fee estimation...")
    
    try:
        # Typical Super Bowl prices
        sea_price = 67  # Seattle at 67¢ (favorite)
        ne_price = 33   # New England at 33¢ (underdog)
        
        # Kalshi fee: 0.07 × C × P × (1-P), max 2¢ per contract
        def calc_fee(price_cents: int) -> float:
            p = price_cents / 100.0
            return min(0.07 * p * (1 - p), 0.02) * 100  # Return in cents
        
        sea_fee = calc_fee(sea_price)
        ne_fee = calc_fee(ne_price)
        
        print(f"   Taker fees (per contract):")
        print(f"     Seattle (~67¢): {sea_fee:.2f}¢")
        print(f"     New England (~33¢): {ne_fee:.2f}¢")
        
        # Round-trip estimate
        sea_roundtrip = sea_fee * 2
        ne_roundtrip = ne_fee * 2
        
        print(f"\n   Round-trip fees:")
        print(f"     Seattle: {sea_roundtrip:.2f}¢ per contract")
        print(f"     New England: {ne_roundtrip:.2f}¢ per contract")
        
        min_edge = int(os.getenv("SB_MIN_EDGE", "8"))
        
        print(f"\n   With SB_MIN_EDGE={min_edge}¢:")
        print(f"     Seattle: {min_edge}¢ edge - {sea_roundtrip:.1f}¢ fees = {min_edge - sea_roundtrip:.1f}¢ net")
        print(f"     New England: {min_edge}¢ edge - {ne_roundtrip:.1f}¢ fees = {min_edge - ne_roundtrip:.1f}¢ net")
        
        if min_edge < sea_roundtrip + 2:
            print(f"\n   ⚠  MIN_EDGE is close to round-trip fees")
            print(f"     → Need high win rate (>60%) to be profitable")
        else:
            print(f"\n   ✓ MIN_EDGE provides buffer over fees")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Fee estimation failed: {e}")
        return True  # Non-critical


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  SUPER BOWL LIX - PRE-FLIGHT CHECK")
    print("  New England Patriots vs Seattle Seahawks")
    print("  Mean Reversion Strategy - ML (Moneyline) Mode")
    print("=" * 80)
    
    checks = [
        ("Environment Variables", check_env_vars, True),
        ("Private Key", check_private_key, True),
        ("API Connection", check_api_connection, True),
        ("Dependencies", check_dependencies, True),
        ("Code Files", check_files, True),
        ("Super Bowl Markets", check_superbowl_markets, True),
        ("Orderbook Access", check_orderbook_access, True),
        ("NFL Game Clock", check_nfl_clock, False),  # Non-critical
        ("Configuration", check_configuration, False),  # Non-critical
        ("Directories", check_directories, False),  # Non-critical
        ("Fee Estimation", estimate_fees, False),  # Non-critical
    ]
    
    results = {}
    critical_failed = False
    
    for name, check_func, is_critical in checks:
        try:
            passed = check_func()
            results[name] = passed
            
            if not passed and is_critical:
                critical_failed = True
                
        except Exception as e:
            print(f"\n   ✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
            
            if is_critical:
                critical_failed = True
    
    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80 + "\n")
    
    for name, passed in results.items():
        # Find if critical
        is_critical = next((c[2] for c in checks if c[0] == name), False)
        critical_marker = " (CRITICAL)" if is_critical else ""
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:10s} {name}{critical_marker}")
    
    print("\n" + "=" * 80)
    
    if critical_failed:
        print("  ✗ CRITICAL CHECKS FAILED - FIX ISSUES ABOVE")
        print("=" * 80 + "\n")
        return 1
    
    if all(results.values()):
        print("  ✅ ALL CHECKS PASSED - READY TO TRADE")
        print("=" * 80 + "\n")
        print("  To start trading:")
        print("    python superbowl_runner.py")
        print()
        return 0
    else:
        print("  ⚠️  SOME NON-CRITICAL CHECKS FAILED")
        print("  You can proceed, but review warnings above")
        print("=" * 80 + "\n")
        print("  To start trading:")
        print("    python superbowl_runner.py")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())