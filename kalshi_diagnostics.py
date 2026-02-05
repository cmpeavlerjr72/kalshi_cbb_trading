# kalshi_diagnostics.py
# Latency measurement and system diagnostics for Kalshi trading
#
# Run this before trading to understand your execution environment

import os
import time
import statistics
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

# Import auth helpers from your existing code
# If running standalone, we'll define minimal versions here

ENV = (os.getenv("KALSHI_ENV") or "DEMO").upper()
BASE_URL = "https://demo-api.kalshi.co" if ENV == "DEMO" else "https://api.elections.kalshi.com"
API_KEY_ID = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
PRIVATE_KEY_PATH = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()

try:
    from combo_vnext import _load_private_key, _get, fetch_orderbook
    HAS_COMBO = True
except ImportError:
    HAS_COMBO = False
    import requests
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    import base64

    def _load_private_key(path: str):
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _now_ms() -> str:
        return str(int(time.time() * 1000))

    def _sign_request(private_key, ts_ms: str, method: str, path: str) -> str:
        path_no_q = path.split("?")[0]
        msg = f"{ts_ms}{method.upper()}{path_no_q}".encode("utf-8")
        sig = private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _auth_headers(private_key, method: str, path: str) -> Dict[str, str]:
        ts_ms = _now_ms()
        signature = _sign_request(private_key, ts_ms, method, path)
        return {
            "KALSHI-ACCESS-KEY": API_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

    def _get(private_key, path: str, params=None) -> Any:
        url = BASE_URL + path
        headers = _auth_headers(private_key, "GET", path)
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"GET {path} failed: {resp.status_code} {resp.text}")
        return resp.json()

    def fetch_orderbook(ticker: str) -> Dict[str, Any]:
        import requests as req
        resp = req.get(f"{BASE_URL}/trade-api/v2/markets/{ticker}/orderbook", timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"Orderbook fetch failed: {resp.status_code}")
        return resp.json()


def measure_latency_authenticated(private_key, num_samples: int = 20) -> Dict[str, Any]:
    """
    Measure round-trip latency to Kalshi API for authenticated endpoints.
    This is what matters for order placement.
    """
    print(f"\n{'='*60}")
    print("AUTHENTICATED API LATENCY TEST")
    print(f"{'='*60}")
    print(f"Endpoint: {BASE_URL}")
    print(f"Testing: GET /trade-api/v2/portfolio/balance")
    print(f"Samples: {num_samples}")
    print()

    latencies: List[float] = []
    errors = 0

    for i in range(num_samples):
        try:
            start = time.perf_counter()
            _get(private_key, "/trade-api/v2/portfolio/balance")
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            print(f"  Sample {i+1:2d}: {latency_ms:6.1f} ms")
        except Exception as e:
            errors += 1
            print(f"  Sample {i+1:2d}: ERROR - {e}")
        
        time.sleep(0.2)  # Small delay between requests

    if not latencies:
        return {"error": "All requests failed"}

    results = {
        "endpoint": "authenticated",
        "samples": len(latencies),
        "errors": errors,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
    }

    print(f"\n{'─'*40}")
    print(f"  Min:    {results['min_ms']:6.1f} ms")
    print(f"  Max:    {results['max_ms']:6.1f} ms")
    print(f"  Mean:   {results['mean_ms']:6.1f} ms")
    print(f"  Median: {results['median_ms']:6.1f} ms")
    print(f"  Stdev:  {results['stdev_ms']:6.1f} ms")
    print(f"  P95:    {results['p95_ms']:6.1f} ms")
    
    return results


def measure_latency_orderbook(ticker: str, num_samples: int = 20) -> Dict[str, Any]:
    """
    Measure round-trip latency for orderbook fetches (unauthenticated).
    This is what matters for price monitoring.
    """
    print(f"\n{'='*60}")
    print("ORDERBOOK LATENCY TEST")
    print(f"{'='*60}")
    print(f"Endpoint: {BASE_URL}")
    print(f"Testing: GET /trade-api/v2/markets/{ticker}/orderbook")
    print(f"Samples: {num_samples}")
    print()

    latencies: List[float] = []
    errors = 0

    for i in range(num_samples):
        try:
            start = time.perf_counter()
            fetch_orderbook(ticker)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            print(f"  Sample {i+1:2d}: {latency_ms:6.1f} ms")
        except Exception as e:
            errors += 1
            print(f"  Sample {i+1:2d}: ERROR - {e}")
        
        time.sleep(0.2)

    if not latencies:
        return {"error": "All requests failed"}

    results = {
        "endpoint": "orderbook",
        "ticker": ticker,
        "samples": len(latencies),
        "errors": errors,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
    }

    print(f"\n{'─'*40}")
    print(f"  Min:    {results['min_ms']:6.1f} ms")
    print(f"  Max:    {results['max_ms']:6.1f} ms")
    print(f"  Mean:   {results['mean_ms']:6.1f} ms")
    print(f"  Median: {results['median_ms']:6.1f} ms")
    print(f"  Stdev:  {results['stdev_ms']:6.1f} ms")
    print(f"  P95:    {results['p95_ms']:6.1f} ms")

    return results


def interpret_latency(results: Dict[str, Any]) -> str:
    """
    Interpret latency results and provide recommendations.
    """
    if "error" in results:
        return "❌ Could not measure latency - check API credentials"
    
    median = results["median_ms"]
    
    interpretations = []
    
    if median < 50:
        interpretations.append("🟢 EXCELLENT: Sub-50ms latency")
        interpretations.append("   You can compete with most automated traders")
        interpretations.append("   Market-making and scalping strategies are viable")
    elif median < 100:
        interpretations.append("🟡 GOOD: 50-100ms latency")
        interpretations.append("   Adequate for most strategies")
        interpretations.append("   May lose some races to faster traders")
    elif median < 200:
        interpretations.append("🟠 MODERATE: 100-200ms latency")
        interpretations.append("   Focus on slower strategies (multi-minute holds)")
        interpretations.append("   Avoid scalping or rapid mean-reversion")
    else:
        interpretations.append("🔴 HIGH: >200ms latency")
        interpretations.append("   Consider running from a cloud server (AWS/GCP)")
        interpretations.append("   Focus on longer-duration positions only")
    
    # Check consistency
    if results.get("stdev_ms", 0) > results.get("mean_ms", 100) * 0.5:
        interpretations.append("")
        interpretations.append("⚠️  HIGH VARIANCE: Latency is inconsistent")
        interpretations.append("   This can cause unexpected slippage")
        interpretations.append("   Check your network connection")
    
    return "\n".join(interpretations)


def find_active_ticker(private_key) -> str:
    """Find an active NCAAM market ticker for testing."""
    try:
        resp = _get(private_key, "/trade-api/v2/markets", params={
            "status": "open",
            "series_ticker": "KXNCAAMBGAME",
            "limit": 1
        })
        markets = resp.get("markets", [])
        if markets:
            return markets[0]["ticker"]
    except:
        pass
    
    # Fallback - return a placeholder
    return "KXNCAAMBGAME-26FEB05TEST-TEST"


def run_full_diagnostics():
    """Run complete diagnostics suite."""
    print("\n" + "="*60)
    print("KALSHI TRADING SYSTEM DIAGNOSTICS")
    print("="*60)
    print(f"\nEnvironment: {ENV}")
    print(f"Base URL: {BASE_URL}")
    print(f"API Key ID: {API_KEY_ID[:8]}..." if API_KEY_ID else "API Key: NOT SET")
    
    if not API_KEY_ID or not PRIVATE_KEY_PATH:
        print("\n❌ ERROR: KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set")
        return
    
    try:
        private_key = _load_private_key(PRIVATE_KEY_PATH)
        print(f"Private Key: Loaded successfully")
    except Exception as e:
        print(f"\n❌ ERROR loading private key: {e}")
        return
    
    # Test authentication
    print("\n" + "-"*40)
    print("Testing API authentication...")
    try:
        balance = _get(private_key, "/trade-api/v2/portfolio/balance")
        balance_dollars = int(balance.get("balance", 0)) / 100
        print(f"✓ Authentication successful")
        print(f"  Account balance: ${balance_dollars:.2f}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return
    
    # Measure authenticated latency
    auth_results = measure_latency_authenticated(private_key, num_samples=20)
    print("\n" + interpret_latency(auth_results))
    
    # Find active ticker and measure orderbook latency
    print("\n" + "-"*40)
    print("Finding active market for orderbook test...")
    ticker = find_active_ticker(private_key)
    print(f"Using ticker: {ticker}")
    
    try:
        ob_results = measure_latency_orderbook(ticker, num_samples=20)
        print("\n" + interpret_latency(ob_results))
    except Exception as e:
        print(f"Orderbook test failed: {e}")
        ob_results = {"error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    if "error" not in auth_results:
        auth_median = auth_results["median_ms"]
        print(f"\nAuthenticated API median latency: {auth_median:.0f}ms")
        
        if auth_median < 100:
            print("→ Your latency is competitive for live trading")
            print("→ Strategies requiring quick execution are viable")
        elif auth_median < 200:
            print("→ Your latency is adequate but not optimal")
            print("→ Consider using maker orders more to avoid races")
        else:
            print("→ Your latency puts you at a disadvantage")
            print("→ Consider:")
            print("  • Running from a cloud VM (AWS us-east-1 or GCP)")
            print("  • Using longer holding periods")
            print("  • Avoiding time-sensitive strategies")
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "environment": ENV,
        "authenticated_latency": auth_results,
        "orderbook_latency": ob_results if "error" not in ob_results else None,
    }
    
    results_path = "latency_diagnostics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_full_diagnostics()