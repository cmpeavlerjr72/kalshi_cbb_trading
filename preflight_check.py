# preflight_check.py
# Run this BEFORE tonight_runner.py to validate your setup (env, auth, markets, ESPN mappings)

import os
import sys
import re
import time
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import requests

# -----------------------------------------------------------------------------
# Existing checks (env/auth/deps/files/markets/dirs)
# -----------------------------------------------------------------------------

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

        # heuristic: 2 games × $6 (kept from your original file)
        required = 12.0
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
        "python_dotenv",  # import name differs
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"   ✓ {pkg.replace('_', '-')}")
        except ImportError:
            print(f"   ✗ {pkg.replace('_', '-') } not installed")
            missing.append(pkg.replace("_", "-"))

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

    return not bool(missing)


def check_markets():
    """Check if tonight's markets are available"""
    print("\n7. Checking market availability...")

    try:
        from combo_vnext import _load_private_key, get_markets_in_series, SERIES_TICKER

        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        private_key = _load_private_key(key_path)

        markets = get_markets_in_series(private_key, SERIES_TICKER)
        print(f"   ✓ Found {len(markets)} NCAAM markets")

        # pull team_name values from tonight_runner if possible (better than hardcoding)
        teams = _load_team_names_from_runner_fallback()
        found = {team: False for team in teams}

        for market in markets:
            ticker = (market.get("ticker", "") or "").upper()
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
    print("\n8. Checking output directories...")

    dirs = ["logs", "data"]

    for dirname in dirs:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print(f"   ✓ Created {dirname}/")
        else:
            print(f"   ✓ {dirname}/ exists")

    return True


# -----------------------------------------------------------------------------
# NEW: ESPN mapping validation + best-effort suggestions
# -----------------------------------------------------------------------------

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _collect_events(scoreboard: Dict[str, Any]) -> List[Dict[str, Any]]:
    direct = scoreboard.get("events") if isinstance(scoreboard.get("events"), list) else []
    leagues = scoreboard.get("leagues") if isinstance(scoreboard.get("leagues"), list) else []
    league_lists = []
    for L in leagues:
        evs = L.get("events") if isinstance(L.get("events"), list) else []
        league_lists.extend(evs)

    raw = league_lists if len(league_lists) >= len(direct) else direct

    seen = set()
    out = []
    for ev in raw:
        eid = str(ev.get("id") or "")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        out.append(ev)
    return out


def _event_competitors(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    comp = (ev.get("competitions") or [None])[0] or {}
    return comp.get("competitors") or []


def _event_team_abbrs(ev: Dict[str, Any]) -> List[str]:
    abbrs = []
    for c in _event_competitors(ev):
        team = c.get("team") or {}
        abbr = (team.get("abbreviation") or "").upper()
        if abbr:
            abbrs.append(abbr)
    return abbrs


def _event_team_names(ev: Dict[str, Any]) -> List[str]:
    names = []
    for c in _event_competitors(ev):
        team = c.get("team") or {}
        # try several common ESPN fields
        for key in ("displayName", "shortDisplayName", "name", "location"):
            val = team.get(key)
            if isinstance(val, str) and val.strip():
                names.append(val.strip())
                break
    return names


def _event_pretty(ev: Dict[str, Any]) -> str:
    abbrs = _event_team_abbrs(ev)
    names = _event_team_names(ev)
    return f"{' vs '.join(abbrs) or '??'} — {' vs '.join(names) or 'Unknown teams'}"


def _fetch_scoreboard(date_yyyymmdd: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(
            ESPN_SCOREBOARD,
            params={"dates": date_yyyymmdd, "groups": "50", "limit": 500},
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0", "cache-control": "no-cache"},
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _best_candidates_for_game(
    events: List[Dict[str, Any]],
    label: str,
    team_code: Optional[str],
    opp_code: Optional[str],
    limit: int = 3
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Returns top candidates (score, event) where score higher is better.
    Scoring uses:
      - opponent match boost
      - team_code match boost
      - label string overlap with team names
    """
    label_n = _norm(label)
    team_code_u = (team_code or "").upper()
    opp_code_u = (opp_code or "").upper()

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ev in events:
        abbrs = _event_team_abbrs(ev)
        names = _event_team_names(ev)
        name_n = _norm(" ".join(names))

        score = 0.0
        if opp_code_u and opp_code_u in abbrs:
            score += 5.0
        if team_code_u and team_code_u in abbrs:
            score += 5.0

        # label/name overlap
        if label_n and name_n:
            label_words = set(label_n.split())
            name_words = set(name_n.split())
            overlap = len(label_words & name_words)
            score += overlap * 0.5

        # slight preference if exactly 2 teams known
        if len(abbrs) == 2:
            score += 0.25

        if score > 0:
            scored.append((score, ev))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


def _suggest_fix(label: str, candidates: List[Tuple[float, Dict[str, Any]]], opp_code: Optional[str]) -> str:
    if not candidates:
        return "No good candidate events found on ESPN for that date."

    best_score, best_ev = candidates[0]
    abbrs = _event_team_abbrs(best_ev)
    names = _event_team_names(best_ev)

    # If opponent code given and present, suggest the other abbr as espn_team
    opp_u = (opp_code or "").upper()
    suggestion_team = None
    suggestion_opp = None

    if opp_u and opp_u in abbrs and len(abbrs) == 2:
        suggestion_opp = opp_u
        suggestion_team = abbrs[0] if abbrs[1] == opp_u else abbrs[1]
    elif len(abbrs) == 2:
        # if no opponent, suggest both as a pair
        suggestion_team = abbrs[0]
        suggestion_opp = abbrs[1]

    lines = []
    lines.append(f"Best match: {_event_pretty(best_ev)} (score={best_score:.2f})")

    if suggestion_team and suggestion_opp:
        lines.append(f"Suggested config: espn_team='{suggestion_team}', espn_opponent='{suggestion_opp}'")
    elif suggestion_team:
        lines.append(f"Suggested config: espn_team='{suggestion_team}'")
    else:
        lines.append("Suggested config: (could not infer team/opponent abbreviations reliably)")

    # show additional candidates
    if len(candidates) > 1:
        lines.append("Other candidates:")
        for sc, ev in candidates[1:]:
            lines.append(f"  - {_event_pretty(ev)} (score={sc:.2f})")

    return "\n      ".join(lines)


def _load_games_from_runner() -> List[Dict[str, Any]]:
    # Import the runner module (safe since it guards main)
    import importlib
    mod = importlib.import_module("tonight_runner")
    games = getattr(mod, "GAMES", None)
    if not isinstance(games, list):
        raise RuntimeError("tonight_runner.GAMES not found or not a list")
    return games


def _load_team_names_from_runner_fallback() -> List[str]:
    try:
        games = _load_games_from_runner()
        teams = []
        for g in games:
            tn = (g.get("team_name") or "").upper().strip()
            if tn:
                teams.append(tn)
        return sorted(set(teams)) if teams else ["DAV","YALE","VCU","BRAD","CONN","ILST","VALP","MURR","BEL"]
    except Exception:
        return ["DAV","YALE","VCU","BRAD","CONN","ILST","VALP","MURR","BEL"]


def check_espn_mappings():
    """
    Validates that each game in tonight_runner.GAMES can be matched on ESPN using
    (espn_date, espn_team, espn_opponent). If not, provides suggested fixes.
    """
    print("\n6. Checking ESPN mappings (from tonight_runner.py)...")

    try:
        from espn_game_clock import EspnGameClock
    except Exception as e:
        print(f"   ✗ Could not import EspnGameClock: {e}")
        return False

    try:
        games = _load_games_from_runner()
    except Exception as e:
        print(f"   ✗ Failed to load GAMES from tonight_runner.py: {e}")
        return False

    # group by date to minimize API calls
    dates = sorted({(g.get("espn_date") or "").strip() for g in games if (g.get("espn_date") or "").strip()})
    if not dates:
        print("   ✗ No espn_date values found in GAMES")
        return False

    scoreboard_by_date: Dict[str, Optional[Dict[str, Any]]] = {}
    for d in dates:
        js = _fetch_scoreboard(d)
        scoreboard_by_date[d] = js
        if js is None:
            print(f"   ✗ ESPN scoreboard unreachable for date={d}")
        else:
            ev_count = len(_collect_events(js))
            print(f"   ✓ ESPN scoreboard fetched for {d} ({ev_count} events)")

    if any(scoreboard_by_date[d] is None for d in dates):
        print("   ✗ ESPN unreachable for at least one date — cannot validate mappings.")
        return False

    any_fail = False

    for g in games:
        label = g.get("label", "UNKNOWN")
        d = (g.get("espn_date") or "").strip()
        team = (g.get("espn_team") or "").strip().upper()
        opp = (g.get("espn_opponent") or "").strip().upper() if g.get("espn_opponent") else None

        if not d or not team:
            print(f"   ✗ [{label}] Missing espn_date or espn_team")
            any_fail = True
            continue

        clock = EspnGameClock(yyyymmdd=d, team_code=team, opponent_code=opp, cache_ttl_secs=1)
        ctx = clock.get_live_context()
        why = ctx.get("why")

        if ctx.get("state") is None and why in ("event_not_found", "no_team_code", "espn_unreachable"):
            print(f"   ✗ [{label}] ESPN match FAILED (team={team} opp={opp or '-'}) why={why}")
            # Suggest fix using scoreboard content
            js = scoreboard_by_date[d]
            events = _collect_events(js) if js else []
            cands = _best_candidates_for_game(events, label=label, team_code=team, opp_code=opp, limit=3)
            suggestion = _suggest_fix(label, cands, opp)
            print(f"      {suggestion}")
            any_fail = True
        else:
            state = ctx.get("state")
            secs_to_tip = ctx.get("secs_to_tip")
            print(f"   ✓ [{label}] ESPN match OK (team={team} opp={opp or '-'}) state={state} why={why}"
                  + (f" secs_to_tip={secs_to_tip}" if secs_to_tip is not None else ""))

    if any_fail:
        print("\n   ✗ ESPN mapping check FAILED — fix the suggested espn_team/espn_opponent codes above.")
        print("     (This matters because PregameAnchoredStrategy needs secs_to_tip.)")
        return False

    print("\n   ✓ All ESPN mappings verified.")
    return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  KALSHI MULTI-STRATEGY - PRE-FLIGHT CHECK")
    print("=" * 70)

    checks = [
        ("Environment Variables", check_env_vars),
        ("Private Key", check_private_key),
        ("API Connection", check_api_connection),
        ("Dependencies", check_dependencies),
        ("Code Files", check_files),
        ("ESPN Mappings", check_espn_mappings),   # NEW
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
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70 + "\n")

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:10s} {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("  ✓ ALL CHECKS PASSED - READY TO RUN")
        print("=" * 70 + "\n")
        print("Run: python tonight_runner.py\n")
        return 0
    else:
        print("  ✗ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
