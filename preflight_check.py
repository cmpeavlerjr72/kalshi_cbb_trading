# preflight_check.py
# Run this BEFORE tonight_runner.py to validate your setup

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def check_env_vars():
    """Check required environment variables"""
    print("\n1. Checking environment variables...")
    
    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    env = os.getenv("KALSHI_ENV", "DEMO").upper()
    
    if not api_key:
        print("   ✗ KALSHI_API_KEY_ID not set")
        return False
    else:
        print(f"   ✓ KALSHI_API_KEY_ID: {api_key[:8]}...")
    
    if not key_path:
        print("   ✗ KALSHI_PRIVATE_KEY_PATH not set")
        return False
    else:
        print(f"   ✓ KALSHI_PRIVATE_KEY_PATH: {key_path}")
    
    print(f"   ✓ KALSHI_ENV: {env}")
    
    if env != "PROD":
        print("   ⚠ WARNING: Running in DEMO mode")
    
    return True

def check_private_key():
    """Verify private key can be loaded"""
    print("\n2. Checking private key...")
    
    try:
        from combo_vnext import _load_private_key
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        _load_private_key(key_path)
        print("   ✓ Private key loaded successfully")
        return True
    except Exception as e:
        print(f"   ✗ Failed to load private key: {e}")
        return False

def check_api_connection():
    """Test API connection and authentication"""
    print("\n3. Testing API connection...")
    
    try:
        from combo_vnext import _load_private_key, _get
        
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        private_key = _load_private_key(key_path)
        
        resp = _get(private_key, "/trade-api/v2/portfolio/balance")
        balance = int(resp.get("balance", 0)) / 100
        
        print(f"   ✓ API authentication successful")
        print(f"   ✓ Account balance: ${balance:.2f}")
        
        # Check if we have enough
        required = 12.0  # 2 games × $6
        if balance < required:
            print(f"   ⚠ WARNING: Balance ${balance:.2f} < Required ${required:.2f}")
            print(f"      Consider reducing allocations in tonight_runner.py")
        
        return True
        
    except Exception as e:
        print(f"   ✗ API connection failed: {e}")
        return False

def check_dependencies():
    """Check required Python packages"""
    print("\n4. Checking dependencies...")
    
    required = [
        "requests",
        "cryptography",
        "python-dotenv",
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"   ✓ {pkg}")
        except ImportError:
            print(f"   ✗ {pkg} not installed")
            missing.append(pkg)
    
    if missing:
        print(f"\n   Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def check_files():
    """Check required code files exist"""
    print("\n5. Checking code files...")
    
    required_files = [
        "production_strategies.py",
        "tonight_runner.py",
        "combo_vnext.py",
        "espn_game_clock.py",
    ]
    
    missing = []
    for filename in required_files:
        if os.path.exists(filename):
            print(f"   ✓ {filename}")
        else:
            print(f"   ✗ {filename} missing")
            missing.append(filename)
    
    if missing:
        return False
    
    return True

def check_markets():
    """Check if tonight's markets are available"""
    print("\n6. Checking market availability...")
    
    try:
        from combo_vnext import _load_private_key, get_markets_in_series, SERIES_TICKER
        
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        private_key = _load_private_key(key_path)
        
        markets = get_markets_in_series(private_key, SERIES_TICKER)
        print(f"   ✓ Found {len(markets)} NCAAM markets")
        
        # Check for our specific teams
        teams = ["TARL", "FAIR"]
        found = {team: False for team in teams}
        
        for market in markets:
            ticker = market.get("ticker", "").upper()
            for team in teams:
                if ticker.endswith(f"-{team}"):
                    found[team] = True
        
        for team, exists in found.items():
            if exists:
                print(f"   ✓ {team} market found")
            else:
                print(f"   ⚠ {team} market not found (may not be open yet)")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Market check failed: {e}")
        return False

def check_directories():
    """Check/create output directories"""
    print("\n7. Checking output directories...")
    
    dirs = ["logs", "data"]
    
    for dirname in dirs:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print(f"   ✓ Created {dirname}/")
        else:
            print(f"   ✓ {dirname}/ exists")
    
    return True

def main():
    print("="*70)
    print("  KALSHI MULTI-STRATEGY - PRE-FLIGHT CHECK")
    print("="*70)
    
    checks = [
        ("Environment Variables", check_env_vars),
        ("Private Key", check_private_key),
        ("API Connection", check_api_connection),
        ("Dependencies", check_dependencies),
        ("Code Files", check_files),
        ("Markets", check_markets),
        ("Directories", check_directories),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n   ✗ Unexpected error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70 + "\n")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:10s} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("  ✓ ALL CHECKS PASSED - READY TO RUN")
        print("="*70 + "\n")
        print("Run: python tonight_runner.py\n")
        return 0
    else:
        print("  ✗ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("="*70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
