#!/usr/bin/env python3
"""
Kalshi Live Trading Dashboard
Run: python dashboard.py
Open: http://localhost:8050

Reads CSV logs from Cloudflare R2 (synced by tonight_runner_cloud.py every 20s)
and serves a live-updating dark-themed dashboard.
"""

import os
import io
import csv
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime, timezone

import re
import requests as _requests

from dotenv import load_dotenv

load_dotenv()

# Kalshi API for position reconciliation
try:
    from combo_vnext import _load_private_key
    from fetch_kalshi_ledger import fetch_positions as kalshi_fetch_positions
    _kalshi_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    _kalshi_private_key = _load_private_key(_kalshi_key_path) if _kalshi_key_path else None
    if _kalshi_private_key:
        print("[dashboard] Kalshi API key loaded for reconciliation")
    else:
        print("[dashboard] No KALSHI_PRIVATE_KEY_PATH — reconciliation disabled")
except Exception as _e:
    _kalshi_private_key = None
    print(f"[dashboard] Kalshi key load failed (reconciliation disabled): {_e}")

# =============================================================================
# ESPN LOOKUP (from games.json)
# =============================================================================

GAMES_JSON = Path(__file__).parent / "games.json"


def _safe_name(s: str) -> str:
    """Must match tonight_runner_cloud.py safe_name() exactly."""
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)


def _load_espn_lookup():
    """Map game labels (safe_name form matching R2 folders) to ESPN config."""
    try:
        games = json.loads(GAMES_JSON.read_text())
        lookup = {}
        for g in games:
            key = _safe_name(g["label"])
            lookup[key] = {
                "espn_date": g["espn_date"],
                "espn_team": g["espn_team"].upper(),
                "espn_opponent": (g.get("espn_opponent") or "").upper(),
                "team_name": g.get("team_name", ""),
            }
        return lookup
    except Exception:
        return {}


ESPN_LOOKUP = _load_espn_lookup()

# =============================================================================
# R2 / S3 CONFIG
# =============================================================================

R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET = os.getenv("R2_BUCKET", "").strip()
R2_PREFIX = (os.getenv("R2_PREFIX") or "kalshi/logs").strip().strip("/")

_s3 = None


def get_s3_client():
    global _s3
    if _s3 is None:
        import boto3
        from botocore.config import Config

        _s3 = boto3.client(
            "s3",
            region_name="auto",
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )
    return _s3


# =============================================================================
# R2 CACHE (15-second TTL)
# =============================================================================

class R2Cache:
    def __init__(self, ttl_secs: int = 15):
        self.ttl = ttl_secs
        self._store = {}  # key -> (timestamp, data)
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry and (time.time() - entry[0]) < self.ttl:
                return entry[1]
        return None

    def put(self, key: str, data):
        with self._lock:
            self._store[key] = (time.time(), data)


_cache = R2Cache(ttl_secs=15)


# =============================================================================
# ESPN SCORE FETCHING
# =============================================================================

# Separate longer-lived cache for ESPN scores (survives across 20s dashboard refreshes)
_espn_cache = R2Cache(ttl_secs=30)


def _fetch_espn_scores(espn_date):
    """
    Fetch ESPN scoreboard for a date, return {TEAM_ABBR: score_info}.
    Cached for 30s. Blocking but fast (3s timeout).
    """
    cache_key = f"espn_{espn_date}"
    cached = _espn_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        resp = _requests.get(url, params={"dates": espn_date, "groups": "50", "limit": "500"},
                             timeout=3, headers={"User-Agent": "Mozilla/5.0"})
        data = resp.json()
    except Exception as e:
        print(f"[dashboard] ESPN fetch error for {espn_date}: {e}")
        return {}

    scores = {}
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        competitors = comp.get("competitors") or []

        status_obj = comp.get("status") or ev.get("status") or {}
        typ = status_obj.get("type") or {}
        state = (typ.get("state") or "").lower()
        completed = bool(typ.get("completed"))
        period = 0
        clock_str = ""
        try:
            period = int(status_obj.get("period") or 0)
        except (ValueError, TypeError):
            pass
        clock_str = status_obj.get("displayClock") or ""

        if completed or state == "post":
            clock_display = "Final" + (" (OT)" if period > 2 else "")
        elif state == "pre":
            clock_display = "Pre-game"
        elif state == "in":
            if period == 1:
                clock_display = f"1H {clock_str}"
            elif period == 2:
                clock_display = f"2H {clock_str}"
            elif period > 2:
                clock_display = f"OT{period-2} {clock_str}"
            else:
                clock_display = clock_str
        else:
            clock_display = "--"

        teams = {}
        for c in competitors:
            abbr = (c.get("team", {}).get("abbreviation") or "").upper()
            score_val = c.get("score")
            try:
                teams[abbr] = int(score_val) if score_val is not None else None
            except (ValueError, TypeError):
                teams[abbr] = None

        info = {"teams": teams, "clock_display": clock_display, "state": state, "period": period}
        for abbr in teams:
            scores[abbr] = info

    print(f"[dashboard] ESPN fetched {len(scores)} team entries for {espn_date}")
    _espn_cache.put(cache_key, scores)
    return scores


def _get_game_score(game_key):
    """
    Get score info for a game. Returns dict with team_score, opp_score,
    score_display, clock_display, or empty dict if unavailable.
    """
    cfg = ESPN_LOOKUP.get(game_key)
    if not cfg:
        print(f"[dashboard] No ESPN_LOOKUP entry for game_key={game_key!r}  (keys: {list(ESPN_LOOKUP.keys())[:3]}...)")
        return {}

    all_scores = _fetch_espn_scores(cfg["espn_date"])
    if not all_scores:
        return {}

    info = all_scores.get(cfg["espn_team"])
    if not info:
        print(f"[dashboard] ESPN team {cfg['espn_team']!r} not found in scoreboard (available: {list(all_scores.keys())[:10]}...)")
        return {}

    teams = info["teams"]
    team_score = teams.get(cfg["espn_team"])
    opp_score = teams.get(cfg["espn_opponent"])

    if team_score is not None and opp_score is not None:
        score_display = f"{cfg['espn_team']} {team_score} - {cfg['espn_opponent']} {opp_score}"
    elif team_score is not None:
        score_display = f"{cfg['espn_team']} {team_score}"
    else:
        score_display = ""

    return {
        "team_score": team_score,
        "opp_score": opp_score,
        "score_display": score_display,
        "clock_display": info["clock_display"],
        "game_state": info["state"],
    }


# =============================================================================
# MR SIGNAL COMPUTATION
# =============================================================================

def compute_mr_signal(snapshots, sport="cbb"):
    """
    Compute MR proximity from snapshot mid values.
    Uses sport-specific params matching the live strategy.
    """
    # Sport-specific params
    if sport == "tennis":
        lookback = 80
        low_vol_std_mult = 3.0
        high_vol_std_mult = 2.0
    else:
        lookback = 120
        low_vol_std_mult = 2.5
        high_vol_std_mult = 1.5

    mids = []
    for s in snapshots:
        v = s.get("mid")
        if v:
            try:
                f = float(v)
                mids.append(f)
            except (ValueError, TypeError):
                pass

    if len(mids) < 10:
        return None

    window = mids[-lookback:]
    n = len(window)
    mean = sum(window) / n
    variance = sum((x - mean) ** 2 for x in window) / n
    std = variance ** 0.5

    if std < 0.1:
        return {"mr_mean": round(mean, 2), "mr_std": 0, "mr_threshold": 0,
                "mr_deviation": 0, "mr_pct": 0, "status": "dead"}

    std_mult = low_vol_std_mult if std < 5.0 else high_vol_std_mult
    threshold = std_mult * std
    deviation = mids[-1] - mean

    pct = abs(deviation) / threshold * 100 if threshold > 0 else 0

    # Trigger prices: the mid price that would fire an MR entry
    yes_trigger = round(mean - threshold, 1)  # mid drops here → buy YES
    no_trigger = round(mean + threshold, 1)   # mid rises here → buy NO

    return {
        "mr_mean": round(mean, 2),
        "mr_std": round(std, 2),
        "mr_threshold": round(threshold, 2),
        "mr_deviation": round(deviation, 2),
        "mr_pct": round(min(pct, 999), 1),
        "yes_trigger": yes_trigger,
        "no_trigger": no_trigger,
        "current_mid": round(mids[-1], 1),
    }


def compute_mr_series(snapshots, sport="cbb"):
    """
    Walk all valid snapshots and compute rolling MR bands at each point.
    Uses sport-specific params matching the live strategy.
    Returns a downsampled list of {time, mid, mean, upper, lower} dicts (~120 pts).
    """
    # Sport-specific params matching production_strategies.py / tennis_runner.py
    if sport == "tennis":
        lookback = 80
        low_vol_std_mult = 3.0
        high_vol_std_mult = 2.0
    else:
        lookback = 120
        low_vol_std_mult = 2.5
        high_vol_std_mult = 1.5
    low_vol_cutoff = 5.0

    # Extract (time, mid) pairs
    points = []
    for s in snapshots:
        v = s.get("mid")
        if not v:
            continue
        try:
            mid = float(v)
        except (ValueError, TypeError):
            continue
        points.append((s.get("timestamp", ""), mid))

    if len(points) < lookback:
        return []

    # Compute rolling stats at every point
    mids = [p[1] for p in points]
    full = []
    for i in range(len(points)):
        window = mids[max(0, i - lookback + 1):i + 1]
        if len(window) < lookback:
            continue  # skip warmup
        n = len(window)
        mean = sum(window) / n
        variance = sum((x - mean) ** 2 for x in window) / n
        std = variance ** 0.5
        if std < 0.1:
            continue
        std_mult = low_vol_std_mult if std < low_vol_cutoff else high_vol_std_mult
        upper = mean + std_mult * std
        lower = mean - std_mult * std
        full.append({
            "time": points[i][0],
            "mid": round(mids[i], 2),
            "mean": round(mean, 2),
            "upper": round(upper, 2),
            "lower": round(lower, 2),
        })

    # Downsample to ~120 display points
    if len(full) > 120:
        step = len(full) / 120
        sampled = []
        for j in range(120):
            sampled.append(full[int(j * step)])
        # Always include the last point
        if sampled[-1] is not full[-1]:
            sampled[-1] = full[-1]
        return sampled
    return full


# =============================================================================
# R2 DISCOVERY
# =============================================================================

def list_prefixes(prefix: str):
    """List 'subdirectories' under a prefix using delimiter='/'."""
    s3 = get_s3_client()
    if not prefix.endswith("/"):
        prefix += "/"
    result = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            p = cp["Prefix"].rstrip("/")
            result.append(p.rsplit("/", 1)[-1])
    return result


def _list_flat_files():
    """
    List all flat files directly under R2_PREFIX/ (the old log format).
    Returns list of (key, filename) tuples.
    Cached for the TTL period.
    """
    cache_key = "__flat_files__"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result = []
    try:
        s3 = get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=R2_BUCKET, Prefix=f"{R2_PREFIX}/", Delimiter="/"
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                name = key.rsplit("/", 1)[-1]
                result.append((key, name))
    except Exception as e:
        print(f"[dashboard] _list_flat_files error: {e}")

    _cache.put(cache_key, result)
    return result


def _parse_flat_filename(name: str):
    """
    Parse old-format flat filenames like:
      multistrat_Belmont at Bradley_20260210_010528_snapshots.csv
    Returns (game_label, date_str, file_type) or None.
    file_type is one of: snapshots, trades, positions, events
    """
    import re
    # Pattern: multistrat_{game}_{YYYYMMDD}_{HHMMSS}_{type}.csv
    # Also: summary_{game}_{date}_{time}.json (skip these)
    m = re.match(
        r"multistrat_(.+?)_(\d{8})_(\d{6})_(snapshots|trades|positions|events)\.csv$",
        name,
    )
    if not m:
        return None
    game_label = m.group(1)
    raw_date = m.group(2)
    date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
    file_type = m.group(4)
    return game_label, date_str, file_type


def discover_dates():
    """
    Walk R2 prefix hierarchy to find date folders (new format)
    AND scan flat filenames (old format).
    Returns list of date strings like '2026-02-23'.
    """
    cache_key = "__dates__"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    dates = set()

    # New format: nested folders
    try:
        base = f"{R2_PREFIX}/kalshi"
        sports = list_prefixes(base)
        for sport in sports:
            events = list_prefixes(f"{base}/{sport}")
            for event in events:
                date_dirs = list_prefixes(f"{base}/{sport}/{event}")
                for d in date_dirs:
                    if len(d) == 10 and d[4] == "-":
                        dates.add(d)
    except Exception as e:
        print(f"[dashboard] discover_dates (nested) error: {e}")

    # Old format: flat files
    try:
        for key, name in _list_flat_files():
            parsed = _parse_flat_filename(name)
            if parsed:
                dates.add(parsed[1])
    except Exception as e:
        print(f"[dashboard] discover_dates (flat) error: {e}")

    result = sorted(dates, reverse=True)
    _cache.put(cache_key, result)
    return result


def _has_csv_files(prefix: str) -> bool:
    """Check if there are CSV files directly under a prefix (not in subfolders)."""
    try:
        s3 = get_s3_client()
        resp = s3.list_objects_v2(
            Bucket=R2_BUCKET, Prefix=f"{prefix}/snapshots.csv", MaxKeys=1
        )
        return resp.get("KeyCount", 0) > 0
    except Exception:
        return False


def discover_games(date_str: str):
    """
    Find games for a given date. Handles three R2 layouts:
      1. NEW nested:  .../game_label/TICKER/snapshots.csv  (multi-ticker)
      2. OLD nested:  .../game_label/snapshots.csv          (single-ticker in folder)
      3. OLD flat:    kalshi/logs/multistrat_{game}_{date}_{time}_snapshots.csv
    Returns: [{"game": "...", "tickers": [...], "prefix": "...", ...}, ...]
    """
    cache_key = f"__games__{date_str}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    games = []
    seen_games = set()  # avoid duplicates between nested and flat

    # --- 1 & 2: Nested folder format ---
    try:
        base = f"{R2_PREFIX}/kalshi"
        sports = list_prefixes(base)
        for sport in sports:
            events = list_prefixes(f"{base}/{sport}")
            for event in events:
                date_dirs = list_prefixes(f"{base}/{sport}/{event}")
                if date_str not in date_dirs:
                    continue
                game_dirs = list_prefixes(f"{base}/{sport}/{event}/{date_str}")
                for game in game_dirs:
                    game_prefix = f"{base}/{sport}/{event}/{date_str}/{game}"
                    tickers = list_prefixes(game_prefix)

                    if tickers:
                        # New format: ticker subfolders exist
                        games.append({
                            "game": game,
                            "tickers": sorted(tickers),
                            "prefix": game_prefix,
                            "sport": sport,
                            "event": event,
                        })
                        seen_games.add(game)
                    elif _has_csv_files(game_prefix):
                        # Old nested: CSVs directly in game folder
                        games.append({
                            "game": game,
                            "tickers": ["_root"],
                            "prefix_override": game_prefix,
                            "sport": sport,
                            "event": event,
                        })
                        seen_games.add(game)
    except Exception as e:
        print(f"[dashboard] discover_games (nested) error: {e}")

    # --- 3: Flat file format ---
    try:
        # Group flat files by (game_label, date) -> {type: r2_key}
        flat_games = {}  # game_label -> {snapshots: key, trades: key, ...}
        for key, name in _list_flat_files():
            parsed = _parse_flat_filename(name)
            if not parsed:
                continue
            game_label, file_date, file_type = parsed
            if file_date != date_str:
                continue
            if game_label not in flat_games:
                flat_games[game_label] = {}
            flat_games[game_label][file_type] = key

        for game_label, files in flat_games.items():
            # Skip if already found in nested format
            game_key = game_label.replace(" ", "_")
            if game_key in seen_games:
                continue
            if "snapshots" not in files:
                continue
            games.append({
                "game": game_key,
                "tickers": ["_flat"],
                "flat_files": files,
            })
    except Exception as e:
        print(f"[dashboard] discover_games (flat) error: {e}")

    _cache.put(cache_key, games)
    return games


# =============================================================================
# CSV PARSING FROM R2
# =============================================================================

def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def safe_int(v, default=0):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def fetch_csv_from_r2(key: str):
    """Fetch a CSV file from R2, return list of dicts. Uses cache."""
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        s3 = get_s3_client()
        resp = s3.get_object(Bucket=R2_BUCKET, Key=key)
        body = resp["Body"].read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(body))
        rows = list(reader)
        _cache.put(key, rows)
        return rows
    except s3.exceptions.NoSuchKey:
        return []
    except Exception as e:
        if "NoSuchKey" in str(e) or "404" in str(e):
            return []
        print(f"[dashboard] fetch_csv error for {key}: {e}")
        return []


# =============================================================================
# FEE CALC (mirrors production_strategies.py:49)
# =============================================================================

def calc_taker_fee(price_cents: int, qty: int = 1) -> float:
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    p = price_cents / 100.0
    fee_per = min(0.07 * p * (1 - p), 0.02)
    return fee_per * qty * 100


# =============================================================================
# DATA PROCESSING
# =============================================================================

def compute_capital_risked(trades):
    """Total capital deployed in cents (sum of fill_price * qty for all entries)."""
    return sum(
        safe_float(t.get("fill_price")) * safe_int(t.get("qty"))
        for t in trades if t.get("action", "").startswith("entry_fill")
    )


def compute_open_positions(trades):
    """
    Walk trades.csv chronologically. entry_fill adds to FIFO queue,
    exit_* removes qty. Returns list of open position dicts.
    """
    queues = {}  # (strategy, side) -> list of {price, qty, time, reason}

    for t in trades:
        action = t.get("action", "")
        strategy = t.get("strategy", "")
        side = t.get("side", "")
        key = (strategy, side)

        if action.startswith("entry_fill"):
            if key not in queues:
                queues[key] = []
            queues[key].append({
                "strategy": strategy,
                "ticker": t.get("ticker", ""),
                "side": side,
                "entry_price": safe_float(t.get("fill_price")),
                "qty": safe_int(t.get("qty")),
                "entry_time": t.get("timestamp", ""),
                "reason": t.get("reason", ""),
                "fee": safe_float(t.get("fee_cents")),
            })
        elif action.startswith("exit_") and action != "exit_error":
            exit_qty = safe_int(t.get("qty"))
            if key in queues:
                while exit_qty > 0 and queues[key]:
                    front = queues[key][0]
                    if front["qty"] <= exit_qty:
                        exit_qty -= front["qty"]
                        queues[key].pop(0)
                    else:
                        front["qty"] -= exit_qty
                        exit_qty = 0

    open_pos = []
    for key, q in queues.items():
        for pos in q:
            if pos["qty"] > 0:
                open_pos.append(pos)
    return open_pos


def mark_to_market(open_positions, latest_snapshot):
    """Value open positions at latest bid minus estimated exit fee."""
    if not latest_snapshot:
        return open_positions

    yes_bid = safe_float(latest_snapshot.get("yes_bid"))
    no_bid = safe_float(latest_snapshot.get("no_bid"))

    for pos in open_positions:
        if pos["side"] == "yes" and yes_bid > 0:
            gross = (yes_bid - pos["entry_price"]) * pos["qty"]
            fee_est = calc_taker_fee(int(yes_bid), pos["qty"])
            pos["unrealized_pnl"] = gross - fee_est - pos.get("fee", 0)
            pos["mark_price"] = yes_bid
        elif pos["side"] == "no" and no_bid > 0:
            gross = (no_bid - pos["entry_price"]) * pos["qty"]
            fee_est = calc_taker_fee(int(no_bid), pos["qty"])
            pos["unrealized_pnl"] = gross - fee_est - pos.get("fee", 0)
            pos["mark_price"] = no_bid
        else:
            pos["unrealized_pnl"] = 0
            pos["mark_price"] = 0

        # Age in seconds
        try:
            entry_dt = datetime.fromisoformat(pos["entry_time"].replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - entry_dt).total_seconds()
            pos["age_secs"] = int(age)
        except Exception:
            pos["age_secs"] = 0

    return open_positions


def aggregate_closed_positions(positions_rows):
    """Aggregate from positions.csv — trades, W/L, locks/stops, net P&L, avg hold."""
    by_strategy = {}

    for row in positions_rows:
        strat = row.get("strategy", "unknown")
        if strat not in by_strategy:
            by_strategy[strat] = {
                "strategy": strat,
                "trades": 0, "wins": 0, "losses": 0,
                "locks": 0, "stops": 0,
                "net_pnl": 0.0, "gross_pnl": 0.0,
                "total_hold_secs": 0,
            }

        s = by_strategy[strat]
        s["trades"] += 1
        net = safe_float(row.get("net_pnl"))
        s["net_pnl"] += net
        s["gross_pnl"] += safe_float(row.get("gross_pnl"))
        s["total_hold_secs"] += safe_int(row.get("hold_secs"))

        if net > 0:
            s["wins"] += 1
        else:
            s["losses"] += 1

        exit_type = row.get("exit_type", "")
        if "lock" in exit_type:
            s["locks"] += 1
        elif "stop" in exit_type:
            s["stops"] += 1

    for s in by_strategy.values():
        if s["trades"] > 0:
            s["avg_hold_secs"] = s["total_hold_secs"] / s["trades"]
            s["win_rate"] = s["wins"] / s["trades"]
        else:
            s["avg_hold_secs"] = 0
            s["win_rate"] = 0
        del s["total_hold_secs"]

    return list(by_strategy.values())


def ticker_type(ticker_label: str) -> str:
    """Label ends with digit = Spread, otherwise ML."""
    if ticker_label and ticker_label[-1].isdigit():
        return "Spread"
    return "ML"


def build_chart_data(snapshots, positions_rows, ticker_label, sport="cbb"):
    """
    Build chart series for one ticker.
    - mid_series: downsampled mid values from snapshots.csv (~60 points)
    - pnl_series: cumulative P&L from positions.csv (one point per closed trade)
    - mr_series: rolling MR bands with sport-specific params
    """
    # Mid price series (downsample to ~60 points)
    mid_series = []
    valid_snaps = [s for s in snapshots if s.get("mid")]
    step = max(1, len(valid_snaps) // 60)
    for i in range(0, len(valid_snaps), step):
        s = valid_snaps[i]
        mid_series.append({
            "time": s.get("timestamp", ""),
            "secs_to_close": safe_int(s.get("secs_to_close")),
            "mid": safe_float(s.get("mid")),
        })

    # Cumulative P&L series
    pnl_series = []
    cum_pnl = 0.0
    for row in positions_rows:
        net = safe_float(row.get("net_pnl"))
        cum_pnl += net
        pnl_series.append({
            "time": row.get("exit_time", ""),
            "net_pnl": net,
            "cum_pnl": cum_pnl,
        })

    # MR band series
    mr_series = compute_mr_series(snapshots, sport=sport)

    return {"mid_series": mid_series, "pnl_series": pnl_series, "mr_series": mr_series}


def compute_order_activity(trades, events):
    """
    Summarize order activity from trades.csv + events.csv.
    Events has entry_attempt_*, entry_skip_* (orders that never hit exchange).
    Trades has entry_fill_*, entry_nofill_* (orders that were placed).
    """
    attempts = 0
    fills = 0
    nofills = 0
    skips = 0
    last_event = None

    # Count attempts and skips from events.csv
    for e in events:
        ev = e.get("event", "")
        if ev.startswith("entry_attempt"):
            attempts += 1
            detail = e.get("detail", "")
            # Parse "yes 3x@38c mr_deviation" → side, price
            parts = detail.split()
            side = parts[0] if parts else ""
            price = ""
            if len(parts) > 1 and "@" in parts[1]:
                price = parts[1].split("@")[1].rstrip("c")
            last_event = {"action": "attempt", "side": side, "price": price, "time": e.get("timestamp", "")}
        elif ev.startswith("entry_skip"):
            skips += 1
            detail = e.get("detail", "")
            parts = detail.split("@")
            side = parts[0].strip() if parts else ""
            price = parts[1].split("c")[0] if len(parts) > 1 else ""
            # Friendly skip reason
            reason = ev.replace("entry_skip_", "").replace("_", " ")
            last_event = {"action": "skip", "side": side, "price": price, "time": e.get("timestamp", ""), "reason": reason}

    # Count fills and nofills from trades.csv
    for t in trades:
        action = t.get("action", "")
        if "entry_fill" in action:
            fills += 1
            last_event = {"action": "fill", "side": t.get("side", ""), "price": t.get("intended_price", ""), "time": t.get("timestamp", "")}
        elif "entry_nofill" in action:
            nofills += 1
            last_event = {"action": "nofill", "side": t.get("side", ""), "price": t.get("intended_price", ""), "time": t.get("timestamp", "")}

    return {
        "attempts": attempts,
        "fills": fills,
        "nofills": nofills,
        "skips": skips,
        "last": last_event,
    }


def build_trade_markers(trades, positions_rows):
    """Build entry/exit markers for chart overlay."""
    markers = []
    for t in trades:
        action = t.get("action", "")
        if action.startswith("entry_fill"):
            markers.append({
                "type": "entry",
                "time": t.get("timestamp", ""),
                "price": safe_float(t.get("fill_price")),
                "side": t.get("side", ""),
            })
        elif action.startswith("exit_") and "nofill" not in action and "error" not in action:
            markers.append({
                "type": "exit",
                "time": t.get("timestamp", ""),
                "price": safe_float(t.get("fill_price")),
                "side": t.get("side", ""),
            })
    return markers


# =============================================================================
# ORCHESTRATOR: build full dashboard payload for /api/data
# =============================================================================

def build_dashboard_data(date_str: str):
    games = discover_games(date_str)
    if not games:
        return {"date": date_str, "games": [], "error": "No data found for this date"}

    result = {
        "date": date_str,
        "games": [],
        "portfolio": {
            "total_pnl": 0.0,
            "realized": 0.0,
            "unrealized": 0.0,
            "capital_risked": 0.0,
            "open_count": 0,
            "wins": 0,
            "losses": 0,
        },
        "ml_vs_spread": {
            "ml_realized": 0.0,
            "ml_unrealized": 0.0,
            "ml_total": 0.0,
            "ml_risked": 0.0,
            "spread_realized": 0.0,
            "spread_unrealized": 0.0,
            "spread_total": 0.0,
            "spread_risked": 0.0,
        },
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }

    for game_info in games:
        game_key = game_info["game"]
        game_sport = game_info.get("sport", "cbb")
        game_data = {
            "game": game_key.replace("_", " "),
            "sport": game_sport,
            "tickers": [],
            "events": [],
            "game_progress": None,
            "espn_wp": None,
        }

        # ESPN live scores (CBB only — tennis has no ESPN clock)
        if game_sport != "tennis":
            try:
                score_info = _get_game_score(game_key)
                if score_info:
                    game_data["team_score"] = score_info.get("team_score")
                    game_data["opp_score"] = score_info.get("opp_score")
                    game_data["score_display"] = score_info.get("score_display", "")
                    game_data["clock_display"] = score_info.get("clock_display", "")
                    game_data["game_state"] = score_info.get("game_state", "")
            except Exception as e:
                print(f"[dashboard] ESPN score fetch error for {game_key}: {e}")

        all_open_positions = []
        all_closed_stats = []

        for ticker_label in game_info["tickers"]:
            # Determine how to fetch CSVs based on format
            if ticker_label == "_flat" and "flat_files" in game_info:
                # Old flat format: individual R2 keys per file type
                ff = game_info["flat_files"]
                snapshots = fetch_csv_from_r2(ff["snapshots"]) if "snapshots" in ff else []
                trades = fetch_csv_from_r2(ff["trades"]) if "trades" in ff else []
                positions = fetch_csv_from_r2(ff["positions"]) if "positions" in ff else []
                events = fetch_csv_from_r2(ff["events"]) if "events" in ff else []
            elif ticker_label == "_root" and "prefix_override" in game_info:
                # Old nested format: CSVs directly in game folder
                csv_prefix = game_info["prefix_override"]
                snapshots = fetch_csv_from_r2(f"{csv_prefix}/snapshots.csv")
                trades = fetch_csv_from_r2(f"{csv_prefix}/trades.csv")
                positions = fetch_csv_from_r2(f"{csv_prefix}/positions.csv")
                events = fetch_csv_from_r2(f"{csv_prefix}/events.csv")
            else:
                # New format: ticker subfolder
                csv_prefix = f"{game_info['prefix']}/{ticker_label}"
                snapshots = fetch_csv_from_r2(f"{csv_prefix}/snapshots.csv")
                trades = fetch_csv_from_r2(f"{csv_prefix}/trades.csv")
                positions = fetch_csv_from_r2(f"{csv_prefix}/positions.csv")
                events = fetch_csv_from_r2(f"{csv_prefix}/events.csv")

            # For old formats, derive a display label from the ticker column in CSV
            if ticker_label in ("_root", "_flat") and snapshots:
                raw_ticker = snapshots[0].get("ticker", "")
                # e.g. "KXNCAAMBGAME-26FEB10SIUINST-SIU" -> "SIU"
                ticker_label = raw_ticker.rsplit("-", 1)[-1] if "-" in raw_ticker else game_info["game"]
            elif ticker_label in ("_root", "_flat"):
                # No snapshot data at all — skip this empty ticker
                if not trades and not positions and not events:
                    continue
                ticker_label = game_info["game"]

            # Latest snapshot for this ticker
            latest_snap = snapshots[-1] if snapshots else {}

            # Game-level metadata from latest snapshot
            if latest_snap:
                gp = latest_snap.get("game_progress")
                if gp:
                    game_data["game_progress"] = safe_float(gp)
                wp = latest_snap.get("espn_live_win_pct")
                if wp:
                    game_data["espn_wp"] = safe_float(wp)

            # Open positions from trades
            open_pos = compute_open_positions(trades)
            # Tag each with display ticker, preserve raw Kalshi ticker for reconciliation
            for p in open_pos:
                p["raw_ticker"] = p.get("ticker", "")
                p["ticker"] = ticker_label
            open_pos = mark_to_market(open_pos, latest_snap)

            # Closed position stats
            closed_stats = aggregate_closed_positions(positions)
            for s in closed_stats:
                s["ticker"] = ticker_label

            # Ticker-level realized / unrealized / capital risked
            realized = sum(safe_float(r.get("net_pnl")) for r in positions)
            unrealized = sum(p.get("unrealized_pnl", 0) for p in open_pos)
            capital_risked = compute_capital_risked(trades)

            # Bid/ask from latest snapshot
            yes_bid = safe_float(latest_snap.get("yes_bid")) if latest_snap else 0
            yes_ask = safe_float(latest_snap.get("yes_ask")) if latest_snap else 0
            no_bid = safe_float(latest_snap.get("no_bid")) if latest_snap else 0
            no_ask = safe_float(latest_snap.get("no_ask")) if latest_snap else 0
            mid = safe_float(latest_snap.get("mid")) if latest_snap else 0
            spread = safe_float(latest_snap.get("spread")) if latest_snap else 0

            # Charts
            chart_data = build_chart_data(snapshots, positions, ticker_label, sport=game_sport)
            trade_markers = build_trade_markers(trades, positions)
            order_activity = compute_order_activity(trades, events)

            # MR signal proximity
            try:
                mr_signal = compute_mr_signal(snapshots, sport=game_sport)
            except Exception:
                mr_signal = None

            ticker_data = {
                "label": ticker_label,
                "type": ticker_type(ticker_label),
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_bid": no_bid,
                "no_ask": no_ask,
                "mid": mid,
                "spread": spread,
                "open_count": len(open_pos),
                "realized": realized,
                "unrealized": unrealized,
                "capital_risked": capital_risked,
                "chart": chart_data,
                "markers": trade_markers,
                "mr_signal": mr_signal,
                "orders": order_activity,
            }
            game_data["tickers"].append(ticker_data)

            all_open_positions.extend(open_pos)
            all_closed_stats.extend(closed_stats)

            # Portfolio totals
            result["portfolio"]["realized"] += realized
            result["portfolio"]["unrealized"] += unrealized
            result["portfolio"]["capital_risked"] += capital_risked
            result["portfolio"]["open_count"] += len(open_pos)

            for s in closed_stats:
                result["portfolio"]["wins"] += s.get("wins", 0)
                result["portfolio"]["losses"] += s.get("losses", 0)

            # Events (last 20)
            for e in events[-20:]:
                game_data["events"].append({
                    "time": e.get("timestamp", ""),
                    "strategy": e.get("strategy", ""),
                    "event": e.get("event", ""),
                    "detail": e.get("detail", ""),
                })

        game_data["open_positions"] = all_open_positions
        game_data["closed_stats"] = all_closed_stats
        game_data["events"] = game_data["events"][-20:]

        # Per-game ML vs Spread aggregation
        ml_real = sum(t["realized"] for t in game_data["tickers"] if t["type"] == "ML")
        ml_unreal = sum(t["unrealized"] for t in game_data["tickers"] if t["type"] == "ML")
        ml_risked = sum(t["capital_risked"] for t in game_data["tickers"] if t["type"] == "ML")
        sp_real = sum(t["realized"] for t in game_data["tickers"] if t["type"] == "Spread")
        sp_unreal = sum(t["unrealized"] for t in game_data["tickers"] if t["type"] == "Spread")
        sp_risked = sum(t["capital_risked"] for t in game_data["tickers"] if t["type"] == "Spread")
        game_data["ml_pnl"] = {"realized": ml_real, "unrealized": ml_unreal, "total": ml_real + ml_unreal, "risked": ml_risked}
        game_data["spread_pnl"] = {"realized": sp_real, "unrealized": sp_unreal, "total": sp_real + sp_unreal, "risked": sp_risked}

        # Accumulate into global ML vs Spread
        result["ml_vs_spread"]["ml_realized"] += ml_real
        result["ml_vs_spread"]["ml_unrealized"] += ml_unreal
        result["ml_vs_spread"]["ml_risked"] += ml_risked
        result["ml_vs_spread"]["spread_realized"] += sp_real
        result["ml_vs_spread"]["spread_unrealized"] += sp_unreal
        result["ml_vs_spread"]["spread_risked"] += sp_risked
        # Skip games with no meaningful data (empty CSVs / failed runs)
        has_snapshots = any(t.get("mid", 0) != 0 for t in game_data["tickers"])
        has_trades = any(
            t.get("realized", 0) != 0 or t.get("open_count", 0) > 0 or t.get("capital_risked", 0) > 0
            for t in game_data["tickers"]
        )
        has_charts = any(t.get("chart", {}).get("mid_series") for t in game_data["tickers"])
        if has_snapshots or has_trades or has_charts:
            result["games"].append(game_data)

    result["portfolio"]["total_pnl"] = (
        result["portfolio"]["realized"] + result["portfolio"]["unrealized"]
    )
    total = result["portfolio"]["wins"] + result["portfolio"]["losses"]
    result["portfolio"]["win_rate"] = (
        result["portfolio"]["wins"] / total if total > 0 else 0
    )

    port = result["portfolio"]
    cap = port["capital_risked"]
    port["roi_pct"] = (port["total_pnl"] / cap * 100) if cap else 0

    mvs = result["ml_vs_spread"]
    mvs["ml_total"] = mvs["ml_realized"] + mvs["ml_unrealized"]
    mvs["spread_total"] = mvs["spread_realized"] + mvs["spread_unrealized"]

    # --- Position reconciliation against Kalshi API ---
    result["reconciliation"] = []
    result["has_mismatch"] = False
    if _kalshi_private_key:
        try:
            kalshi_positions = kalshi_fetch_positions(_kalshi_private_key)

            # Build Kalshi position map: {ticker -> {yes: qty, no: qty}}
            kalshi_map = {}
            for kp in kalshi_positions:
                kt = kp.get("ticker", "")
                ks = kp.get("side", "")
                kq = int(kp.get("count", 0))
                if kq <= 0:
                    continue
                if kt not in kalshi_map:
                    kalshi_map[kt] = {"yes": 0, "no": 0}
                if ks in ("yes", "no"):
                    kalshi_map[kt][ks] += kq

            # Build bot position map from all open positions across all games
            bot_map = {}
            for game_data in result["games"]:
                for pos in game_data.get("open_positions", []):
                    rt = pos.get("raw_ticker", "")
                    if not rt:
                        continue
                    ps = pos.get("side", "")
                    pq = pos.get("qty", 0)
                    if pq <= 0:
                        continue
                    if rt not in bot_map:
                        bot_map[rt] = {"yes": 0, "no": 0}
                    if ps in ("yes", "no"):
                        bot_map[rt][ps] += pq

            # Compare: union of all tickers from both sources
            all_tickers = set(kalshi_map.keys()) | set(bot_map.keys())
            for ticker in sorted(all_tickers):
                kalshi_qty = kalshi_map.get(ticker, {"yes": 0, "no": 0})
                bot_qty = bot_map.get(ticker, {"yes": 0, "no": 0})
                # Short display label from ticker
                display = ticker.rsplit("-", 1)[-1] if "-" in ticker else ticker
                for side in ("yes", "no"):
                    kq = kalshi_qty[side]
                    bq = bot_qty[side]
                    if kq == 0 and bq == 0:
                        continue
                    status = "OK" if kq == bq else "MISMATCH"
                    if status == "MISMATCH":
                        result["has_mismatch"] = True
                    result["reconciliation"].append({
                        "ticker": display,
                        "raw_ticker": ticker,
                        "side": side,
                        "bot_qty": bq,
                        "kalshi_qty": kq,
                        "status": status,
                    })
        except Exception as e:
            print(f"[dashboard] Reconciliation error: {e}")

    return result


# =============================================================================
# BACKGROUND DATA POLLER
# =============================================================================

_bg_data = {}       # date_str -> full dashboard payload
_bg_data_lock = threading.Lock()
_bg_active_date = None  # date the poller is currently refreshing
_bg_wake = threading.Event()  # signal poller to fetch immediately
_bg_status = {"state": "init", "last_error": None, "last_ok": None, "cycles": 0}


def _bg_poller():
    """Background thread: rebuilds dashboard data, then waits 15s or until woken."""
    global _bg_status
    while True:
        date_str = _bg_active_date
        if date_str:
            _bg_status["state"] = f"fetching {date_str}"
            try:
                t0 = time.time()
                payload = build_dashboard_data(date_str)
                elapsed = time.time() - t0
                with _bg_data_lock:
                    _bg_data[date_str] = payload
                ng = len(payload.get("games", []))
                _bg_status["state"] = "idle"
                _bg_status["last_ok"] = f"{ng} games in {elapsed:.1f}s"
                _bg_status["last_error"] = None
                _bg_status["cycles"] += 1
                print(f"[poller] Refreshed {date_str}: {ng} games in {elapsed:.1f}s", flush=True)
            except Exception as e:
                import traceback
                _bg_status["state"] = "error"
                _bg_status["last_error"] = str(e)
                _bg_status["cycles"] += 1
                print(f"[poller] Error building data for {date_str}: {e}", flush=True)
                traceback.print_exc()
        else:
            _bg_status["state"] = "waiting for date"
        _bg_wake.wait(timeout=15)
        _bg_wake.clear()


def get_cached_data(date_str):
    """Return cached data instantly. Never blocks — poller fills data in background."""
    global _bg_active_date

    # Tell poller which date to fetch; wake it immediately on date change
    if _bg_active_date != date_str:
        _bg_active_date = date_str
        _bg_wake.set()  # wake poller now

    with _bg_data_lock:
        cached = _bg_data.get(date_str)

    if cached:
        cached["_poller"] = _bg_status.copy()
        return cached

    stub = {
        "date": date_str, "games": [],
        "fetch_time": datetime.now(timezone.utc).isoformat(),
        "portfolio": {"total_pnl": 0, "realized": 0, "unrealized": 0,
                      "capital_risked": 0, "roi_pct": 0,
                      "open_count": 0, "wins": 0, "losses": 0, "win_rate": 0},
        "ml_vs_spread": {"ml_realized": 0, "ml_unrealized": 0, "ml_total": 0, "ml_risked": 0,
                         "spread_realized": 0, "spread_unrealized": 0, "spread_total": 0, "spread_risked": 0},
        "_poller": _bg_status.copy(),
    }
    return stub


# =============================================================================
# HTTP HANDLER
# =============================================================================

class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress default logging

    def _send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path == "/":
                self._send_html(DASHBOARD_HTML)

            elif path == "/api/dates":
                dates = discover_dates()
                self._send_json({"dates": dates})

            elif path == "/api/data":
                date_str = params.get("date", [None])[0]
                if not date_str:
                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    date_str = today
                data = get_cached_data(date_str)
                self._send_json(data)

            else:
                self.send_response(404)
                self.end_headers()
        except BrokenPipeError:
            pass
        except Exception as e:
            print(f"[dashboard] do_GET error on {self.path}: {e}")
            import traceback; traceback.print_exc()
            try:
                self._send_json({"error": str(e)}, status=500)
            except Exception:
                pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    global _bg_active_date
    port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "8050")))

    # Seed poller with today's date so it starts fetching immediately
    _bg_active_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Start background data poller
    poller = threading.Thread(target=_bg_poller, daemon=True)
    poller.start()
    print(f"[dashboard] Background poller started for {_bg_active_date}")

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard running on port {port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


# =============================================================================
# DASHBOARD HTML (dark theme, SVG charts, auto-refresh)
# =============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Kalshi Live Dashboard</title>
<style>
:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
  --border: #30363d; --text: #c9d1d9; --text2: #8b949e;
  --green: #3fb950; --red: #f85149; --blue: #58a6ff;
  --orange: #d29922; --purple: #bc8cff; --white: #e6edf3;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:'Cascadia Code','Fira Code',monospace; font-size:13px; padding:12px; }
h1 { font-size:18px; color:var(--white); }
h2 { font-size:15px; color:var(--white); margin:8px 0 6px; border-bottom:1px solid var(--border); padding-bottom:4px; }
h3 { font-size:13px; color:var(--text2); margin:6px 0 4px; }
.header { display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:var(--bg2); border:1px solid var(--border); border-radius:6px; margin-bottom:10px; }
.header-right { display:flex; align-items:center; gap:12px; }
.sync { color:var(--text2); font-size:12px; }
.sync.stale { color:var(--orange); }
select { background:var(--bg3); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:3px 8px; font-family:inherit; font-size:12px; }
.portfolio { display:grid; grid-template-columns:repeat(6,1fr); gap:8px; margin-bottom:10px; }
.stat-card { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:10px 12px; text-align:center; }
.stat-card .label { font-size:11px; color:var(--text2); text-transform:uppercase; letter-spacing:0.5px; }
.stat-card .value { font-size:20px; font-weight:bold; margin-top:2px; }
.positive { color:var(--green); }
.negative { color:var(--red); }
.neutral { color:var(--text2); }
.charts { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px; }
.chart-box { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:10px; min-height:220px; }
.chart-box h3 { margin-bottom:6px; }
.game-section { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:10px 12px; margin-bottom:10px; }
.game-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.game-header .title { font-size:14px; color:var(--white); font-weight:bold; }
.game-header .meta { color:var(--text2); font-size:12px; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th { text-align:left; color:var(--text2); font-size:11px; text-transform:uppercase; padding:4px 8px; border-bottom:1px solid var(--border); }
td { padding:4px 8px; border-bottom:1px solid var(--bg3); }
tr:hover { background:var(--bg3); }
.tag { display:inline-block; padding:1px 6px; border-radius:3px; font-size:11px; }
.tag-ml { background:#1f3a5f; color:var(--blue); }
.tag-spread { background:#3b2e1a; color:var(--orange); }
.tag-yes { background:#1a3a1a; color:var(--green); }
.tag-no { background:#3a1a1a; color:var(--red); }
.events-box { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:10px 12px; margin-bottom:10px; max-height:250px; overflow-y:auto; }
.event-row { font-size:11px; padding:2px 0; border-bottom:1px solid var(--bg3); display:flex; gap:8px; }
.event-time { color:var(--text2); min-width:80px; }
.event-strat { color:var(--blue); min-width:100px; }
.event-type { color:var(--orange); min-width:120px; }
.event-detail { color:var(--text); flex:1; word-break:break-all; }
.waiting { text-align:center; padding:40px; color:var(--text2); font-size:14px; }
svg text { font-family:inherit; }
.legend { display:flex; gap:14px; margin-bottom:4px; flex-wrap:wrap; }
.legend-item { font-size:11px; display:flex; align-items:center; gap:4px; }
.legend-dot { width:10px; height:10px; border-radius:2px; display:inline-block; }
.recon-banner { background:#3a1a1a; border:2px solid var(--red); border-radius:6px; padding:10px 14px; margin-bottom:10px; display:none; }
.recon-banner.visible { display:block; }
.recon-banner h3 { color:var(--red); margin-bottom:6px; font-size:14px; }
.recon-banner table { width:100%; border-collapse:collapse; font-size:12px; }
.recon-banner th { text-align:left; color:var(--text2); padding:3px 8px; border-bottom:1px solid var(--border); }
.recon-banner td { padding:3px 8px; border-bottom:1px solid var(--bg3); }
.recon-banner .mismatch { color:var(--red); font-weight:bold; }
.recon-banner .ok { color:var(--green); }
.recon-ok-banner { background:#1a3a1a; border:1px solid var(--green); border-radius:6px; padding:6px 14px; margin-bottom:10px; display:none; font-size:12px; color:var(--green); }
.recon-ok-banner.visible { display:block; }
</style>
</head>
<body>

<div class="header">
  <h1>KALSHI LIVE DASHBOARD</h1>
  <div class="header-right">
    <span class="sync" id="syncLabel">Connecting...</span>
    <label>Sport: <select id="sportSelect"><option value="all">All</option><option value="cbb">CBB</option><option value="tennis">Tennis</option></select></label>
    <label>Date: <select id="dateSelect"></select></label>
    <label><input type="checkbox" id="hideInactive" checked> Hide Inactive</label>
  </div>
</div>

<div class="portfolio" id="portfolio">
  <div class="stat-card"><div class="label">Total P&amp;L</div><div class="value" id="totalPnl">--</div></div>
  <div class="stat-card"><div class="label">Realized</div><div class="value" id="realized">--</div></div>
  <div class="stat-card"><div class="label">Unrealized</div><div class="value" id="unrealized">--</div></div>
  <div class="stat-card"><div class="label">Capital Risked</div><div class="value neutral" id="capitalRisked">--</div></div>
  <div class="stat-card"><div class="label">Open Positions</div><div class="value neutral" id="openCount">--</div></div>
  <div class="stat-card"><div class="label">Win Rate</div><div class="value neutral" id="winRate">--</div></div>
</div>

<div class="portfolio" id="mlVsSpread" style="grid-template-columns:repeat(4,1fr);">
  <div class="stat-card" style="border-left:3px solid var(--blue);">
    <div class="label"><span class="tag tag-ml">ML</span> Total P&amp;L</div><div class="value" id="mlTotal">--</div>
  </div>
  <div class="stat-card" style="border-left:3px solid var(--blue);">
    <div class="label"><span class="tag tag-ml">ML</span> Realized / Unreal</div><div class="value" id="mlDetail">--</div>
  </div>
  <div class="stat-card" style="border-left:3px solid var(--orange);">
    <div class="label"><span class="tag tag-spread">SPREAD</span> Total P&amp;L</div><div class="value" id="spreadTotal">--</div>
  </div>
  <div class="stat-card" style="border-left:3px solid var(--orange);">
    <div class="label"><span class="tag tag-spread">SPREAD</span> Realized / Unreal</div><div class="value" id="spreadDetail">--</div>
  </div>
</div>

<div class="charts">
  <div class="chart-box">
    <h3>Cumulative P&amp;L Over Time</h3>
    <div class="legend" id="pnlLegend"></div>
    <div id="pnlChart"></div>
  </div>
  <div class="chart-box">
    <h3>Price (Mid) by Ticker</h3>
    <div class="legend" id="priceLegend"></div>
    <div id="priceChart"></div>
  </div>
</div>

<div class="recon-ok-banner" id="reconOk">Positions reconciled — all OK</div>
<div class="recon-banner" id="reconBanner">
  <h3>POSITION MISMATCH — Kalshi vs Bot</h3>
  <table>
    <thead><tr><th>Ticker</th><th>Side</th><th>Bot Qty</th><th>Kalshi Qty</th><th>Status</th></tr></thead>
    <tbody id="reconBody"></tbody>
  </table>
</div>

<div id="gamesContainer"></div>

<script>
// ─── CONFIG ───
const REFRESH_MS = 20000;
const COLORS = {LOU:'#58a6ff',LOU2:'#3fb950',LOU5:'#d29922',UNC1:'#bc8cff'};
const DEFAULT_COLORS = ['#58a6ff','#3fb950','#d29922','#bc8cff','#f778ba','#79c0ff'];
let lastFetchTime = null;
let lastData = null;

function colorFor(label, idx) {
  return COLORS[label] || DEFAULT_COLORS[idx % DEFAULT_COLORS.length];
}

function fmtCents(v) {
  if (v == null || isNaN(v)) return '--';
  const sign = v >= 0 ? '+' : '';
  return sign + v.toFixed(1) + 'c';
}
function fmtDollars(v) {
  if (v == null || isNaN(v)) return '';
  return ' ($' + (v/100).toFixed(2) + ')';
}
function fmtPct(pnl, risked) {
  if (!risked || risked === 0) return '--';
  const pct = (pnl / risked * 100).toFixed(1);
  const sign = pct >= 0 ? '+' : '';
  return sign + pct + '%';
}
function fmtRisked(v) {
  if (v == null || isNaN(v) || v === 0) return '--';
  return '$' + (v/100).toFixed(2);
}
function pnlClass(v) { return v > 0.05 ? 'positive' : v < -0.05 ? 'negative' : 'neutral'; }

function fmtTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  } catch(e) { return iso.slice(11,19); }
}
function fmtAge(secs) {
  if (!secs || secs <= 0) return '--';
  if (secs < 60) return secs + 's';
  if (secs < 3600) return Math.floor(secs/60) + 'm ' + (secs%60) + 's';
  return Math.floor(secs/3600) + 'h ' + Math.floor((secs%3600)/60) + 'm';
}

function fmtOrders(ord) {
  if (!ord || ord.attempts === 0) return '<span style="color:var(--text2)">--</span>';
  const f = ord.fills, nf = ord.nofills, sk = ord.skips||0;
  let parts = [];
  if (f > 0) parts.push('<span style="color:var(--green)">'+f+' fill'+(f>1?'s':'')+'</span>');
  if (nf > 0) parts.push('<span style="color:var(--orange)">'+nf+' nofill'+(nf>1?'s':'')+'</span>');
  if (sk > 0) parts.push('<span style="color:var(--text2)">'+sk+' skip'+(sk>1?'s':'')+'</span>');
  let html = parts.join(' / ');
  if (ord.last) {
    const ago = ord.last.time ? fmtTime(ord.last.time) : '';
    const side = (ord.last.side||'').toUpperCase();
    const sideColor = ord.last.side === 'yes' ? 'var(--green)' : 'var(--red)';
    const icons = {fill:'&#10003;', nofill:'&#10007;', skip:'&#8856;', attempt:'&#9654;'};
    const colors = {fill:'var(--green)', nofill:'var(--orange)', skip:'var(--text2)', attempt:'var(--blue)'};
    const act = ord.last.action||'attempt';
    const reason = ord.last.reason ? ' ('+ord.last.reason+')' : '';
    html += '<br><span style="font-size:10px;color:var(--text2)">Last: <span style="color:'+(colors[act]||'var(--text2)')+'">'+( icons[act]||'?')+'</span> <span style="color:'+sideColor+'">'+side+'</span>'+(ord.last.price ? ' @'+ord.last.price+'c' : '')+reason+' '+ago+'</span>';
  }
  return html;
}

function fmtSignal(sig) {
  // MR signal: show the trigger price needed for next entry
  // Below mean → would buy YES at yes_trigger price
  // Above mean → would buy NO at no_trigger price
  if (!sig) return '<span style="color:var(--text2)">--</span>';
  if (sig.status === 'dead') return '<span style="color:var(--text2)" title="Price not moving enough for MR">FLAT</span>';
  const dev = sig.mr_deviation;
  const pct = sig.mr_pct;
  const mid = sig.current_mid;
  // Which side is closer to triggering
  const side = dev < 0 ? 'YES' : 'NO';
  const sideColor = dev < 0 ? '#3fb950' : '#f85149';
  const triggerPrice = dev < 0 ? sig.yes_trigger : sig.no_trigger;

  if (pct >= 100) {
    return '<span style="color:'+sideColor+';font-weight:bold;" title="Mid '+mid+'c hit trigger '+triggerPrice+'c">BUY '+side+'</span>';
  }
  // Color ramp: far from trigger → close to trigger
  let color = '#8b949e';
  if (pct >= 95) color = '#f85149';
  else if (pct >= 80) color = '#d29922';
  else if (pct >= 50) color = '#e3b341';

  // Mini progress bar
  const barW = Math.min(pct, 100);
  const bar = '<span style="display:inline-block;width:40px;height:6px;background:var(--bg3);border-radius:3px;vertical-align:middle;margin-left:4px;">'
    + '<span style="display:block;width:'+barW+'%;height:100%;background:'+color+';border-radius:3px;"></span></span>';
  return '<span style="color:'+color+'" title="Mid: '+mid+'c | Mean: '+sig.mr_mean+'c | '+side+' trigger: '+triggerPrice+'c">'+side+' @ '+triggerPrice+'c'+bar+'</span>';
}

// ─── SVG CHART HELPER ───
function buildSVG(series, opts) {
  // series: [{label, color, points:[{x,y}]}]
  // opts: {width, height, xLabel, yLabel, showFill, fillColor, fillNegColor}
  const W = opts.width || 500, H = opts.height || 180;
  const pad = {top:10, right:15, bottom:28, left:50};
  const cw = W - pad.left - pad.right, ch = H - pad.top - pad.bottom;

  // Compute bounds
  let allX = [], allY = [];
  series.forEach(s => s.points.forEach(p => { allX.push(p.x); allY.push(p.y); }));
  if (allX.length === 0) return '<svg width="'+W+'" height="'+H+'"><text x="'+W/2+'" y="'+H/2+'" fill="#8b949e" text-anchor="middle" font-size="12">No data yet</text></svg>';

  let xMin = Math.min(...allX), xMax = Math.max(...allX);
  let yMin = Math.min(...allY), yMax = Math.max(...allY);
  if (xMax === xMin) xMax = xMin + 1;
  if (yMax === yMin) { yMin -= 1; yMax += 1; }
  // Add Y padding
  const yPad = (yMax - yMin) * 0.1;
  yMin -= yPad; yMax += yPad;
  // Include 0 in Y range for P&L charts
  if (opts.showFill) { yMin = Math.min(yMin, 0); yMax = Math.max(yMax, 0); }

  function sx(v) { return pad.left + (v - xMin)/(xMax - xMin) * cw; }
  function sy(v) { return pad.top + (1 - (v - yMin)/(yMax - yMin)) * ch; }

  let svg = '<svg width="'+W+'" height="'+H+'" xmlns="http://www.w3.org/2000/svg">';

  // Gridlines
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + (yMax-yMin)*i/yTicks;
    const y = sy(v);
    svg += '<line x1="'+pad.left+'" y1="'+y+'" x2="'+(W-pad.right)+'" y2="'+y+'" stroke="#21262d" stroke-width="1"/>';
    svg += '<text x="'+(pad.left-4)+'" y="'+(y+3)+'" fill="#8b949e" font-size="10" text-anchor="end">'+v.toFixed(1)+'</text>';
  }
  // X-axis ticks
  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const v = xMin + (xMax-xMin)*i/xTicks;
    const x = sx(v);
    svg += '<text x="'+x+'" y="'+(H-4)+'" fill="#8b949e" font-size="10" text-anchor="middle">'+v.toFixed(0)+'m</text>';
  }

  // Zero line for P&L
  if (opts.showFill && yMin < 0 && yMax > 0) {
    const y0 = sy(0);
    svg += '<line x1="'+pad.left+'" y1="'+y0+'" x2="'+(W-pad.right)+'" y2="'+y0+'" stroke="#30363d" stroke-width="1" stroke-dasharray="4,3"/>';
  }

  // Fill area for total line (first series with showFill)
  if (opts.showFill && series.length > 0) {
    const total = series.find(s => s.isTotal) || series[0];
    if (total.points.length > 1) {
      const y0 = sy(0);
      // Positive fill
      let posPath = 'M'+sx(total.points[0].x)+','+y0;
      total.points.forEach(p => { posPath += ' L'+sx(p.x)+','+Math.min(sy(Math.max(p.y,0)),y0); });
      posPath += ' L'+sx(total.points[total.points.length-1].x)+','+y0+' Z';
      svg += '<path d="'+posPath+'" fill="rgba(63,185,80,0.1)"/>';
      // Negative fill
      let negPath = 'M'+sx(total.points[0].x)+','+y0;
      total.points.forEach(p => { negPath += ' L'+sx(p.x)+','+Math.max(sy(Math.min(p.y,0)),y0); });
      negPath += ' L'+sx(total.points[total.points.length-1].x)+','+y0+' Z';
      svg += '<path d="'+negPath+'" fill="rgba(248,81,73,0.1)"/>';
    }
  }

  // Lines
  series.forEach(s => {
    if (s.points.length < 2) return;
    const w = s.isTotal ? 2.5 : 1.5;
    let d = 'M';
    s.points.forEach((p,i) => { d += (i?'L':'') + sx(p.x).toFixed(1)+','+sy(p.y).toFixed(1)+' '; });
    svg += '<path d="'+d+'" fill="none" stroke="'+s.color+'" stroke-width="'+w+'" stroke-linejoin="round"/>';
  });

  // Dots for trade markers
  if (opts.markers) {
    opts.markers.forEach(m => {
      const x = sx(m.x), y = sy(m.y);
      if (m.type === 'entry') {
        svg += '<polygon points="'+(x-4)+','+(y+4)+' '+x+','+(y-4)+' '+(x+4)+','+(y+4)+'" fill="'+( m.side==='yes'?'#3fb950':'#f85149')+'" opacity="0.8"/>';
      } else {
        svg += '<polygon points="'+(x-4)+','+(y-4)+' '+x+','+(y+4)+' '+(x+4)+','+(y-4)+'" fill="#d29922" opacity="0.8"/>';
      }
    });
  }

  svg += '</svg>';
  return svg;
}

function buildMRSVG(mrSeries, markers, opts) {
  const W = opts.width || 600;
  const H = opts.height || 200;
  const pad = {top:10, right:10, bottom:22, left:45};
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  if (!mrSeries || mrSeries.length < 2) {
    return '<svg width="'+W+'" height="'+H+'"><text x="'+W/2+'" y="'+H/2+'" fill="#8b949e" text-anchor="middle" font-size="12">No MR data</text></svg>';
  }

  // Compute x (minutes) from first timestamp
  const refTime = new Date(mrSeries[0].time).getTime();
  const pts = mrSeries.map(p => ({
    x: (new Date(p.time).getTime() - refTime) / 60000,
    mid: p.mid, mean: p.mean, upper: p.upper, lower: p.lower
  }));

  const allX = pts.map(p => p.x);
  let allY = [];
  pts.forEach(p => { allY.push(p.mid, p.upper, p.lower); });
  // Include marker prices in Y range
  if (markers) markers.forEach(m => { allY.push(m.price); });

  let xMin = Math.min(...allX), xMax = Math.max(...allX);
  let yMin = Math.min(...allY), yMax = Math.max(...allY);
  if (xMax === xMin) xMax = xMin + 1;
  if (yMax === yMin) { yMin -= 1; yMax += 1; }
  const yPad = (yMax - yMin) * 0.1;
  yMin -= yPad; yMax += yPad;

  function sx(v) { return pad.left + (v - xMin)/(xMax - xMin) * cw; }
  function sy(v) { return pad.top + (1 - (v - yMin)/(yMax - yMin)) * ch; }

  let svg = '<svg width="'+W+'" height="'+H+'" xmlns="http://www.w3.org/2000/svg">';

  // Gridlines
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + (yMax-yMin)*i/yTicks;
    const y = sy(v);
    svg += '<line x1="'+pad.left+'" y1="'+y+'" x2="'+(W-pad.right)+'" y2="'+y+'" stroke="#21262d" stroke-width="1"/>';
    svg += '<text x="'+(pad.left-4)+'" y="'+(y+3)+'" fill="#8b949e" font-size="10" text-anchor="end">'+v.toFixed(1)+'</text>';
  }
  // X-axis ticks
  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const v = xMin + (xMax-xMin)*i/xTicks;
    const x = sx(v);
    svg += '<text x="'+x+'" y="'+(H-4)+'" fill="#8b949e" font-size="10" text-anchor="middle">'+v.toFixed(0)+'m</text>';
  }

  // Band fill (semi-transparent polygon between upper and lower)
  let bandPath = '';
  pts.forEach((p,i) => { bandPath += (i===0?'M':'L') + sx(p.x).toFixed(1)+','+sy(p.upper).toFixed(1)+' '; });
  for (let i = pts.length-1; i >= 0; i--) { bandPath += 'L'+sx(pts[i].x).toFixed(1)+','+sy(pts[i].lower).toFixed(1)+' '; }
  bandPath += 'Z';
  svg += '<path d="'+bandPath+'" fill="rgba(188,140,255,0.08)"/>';

  // Upper band line (dashed)
  let upperD = 'M';
  pts.forEach((p,i) => { upperD += (i?'L':'') + sx(p.x).toFixed(1)+','+sy(p.upper).toFixed(1)+' '; });
  svg += '<path d="'+upperD+'" fill="none" stroke="#6e4da0" stroke-width="1" stroke-dasharray="4,3"/>';

  // Lower band line (dashed)
  let lowerD = 'M';
  pts.forEach((p,i) => { lowerD += (i?'L':'') + sx(p.x).toFixed(1)+','+sy(p.lower).toFixed(1)+' '; });
  svg += '<path d="'+lowerD+'" fill="none" stroke="#6e4da0" stroke-width="1" stroke-dasharray="4,3"/>';

  // Mean line (dashed purple)
  let meanD = 'M';
  pts.forEach((p,i) => { meanD += (i?'L':'') + sx(p.x).toFixed(1)+','+sy(p.mean).toFixed(1)+' '; });
  svg += '<path d="'+meanD+'" fill="none" stroke="#bc8cff" stroke-width="1.5" stroke-dasharray="6,3"/>';

  // Mid price line (solid blue)
  let midD = 'M';
  pts.forEach((p,i) => { midD += (i?'L':'') + sx(p.x).toFixed(1)+','+sy(p.mid).toFixed(1)+' '; });
  svg += '<path d="'+midD+'" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linejoin="round"/>';

  // Trade markers
  if (markers && markers.length > 0) {
    markers.forEach(m => {
      const mx = sx((new Date(m.time).getTime() - refTime) / 60000);
      const my = sy(m.price);
      if (mx < pad.left || mx > W - pad.right) return;
      if (m.type === 'entry') {
        svg += '<polygon points="'+(mx-5)+','+(my+5)+' '+mx+','+(my-5)+' '+(mx+5)+','+(my+5)+'" fill="'+(m.side==='yes'?'#3fb950':'#f85149')+'" opacity="0.9"/>';
      } else {
        svg += '<polygon points="'+(mx-5)+','+(my-5)+' '+mx+','+(my+5)+' '+(mx+5)+','+(my-5)+'" fill="#d29922" opacity="0.9"/>';
      }
    });
  }

  svg += '</svg>';
  return svg;
}

// ─── RENDER FUNCTIONS ───
function renderPortfolio(p) {
  const risked = p.capital_risked || 0;
  const tp = document.getElementById('totalPnl');
  tp.textContent = fmtPct(p.total_pnl, risked) + ' (' + fmtCents(p.total_pnl) + ')';
  tp.className = 'value ' + pnlClass(p.total_pnl);

  const r = document.getElementById('realized');
  r.textContent = fmtPct(p.realized, risked) + ' (' + fmtCents(p.realized) + ')';
  r.className = 'value ' + pnlClass(p.realized);

  const u = document.getElementById('unrealized');
  u.textContent = fmtPct(p.unrealized, risked) + ' (' + fmtCents(p.unrealized) + ')';
  u.className = 'value ' + pnlClass(p.unrealized);

  document.getElementById('capitalRisked').textContent = fmtRisked(risked);
  document.getElementById('openCount').textContent = p.open_count;

  const wr = document.getElementById('winRate');
  const total = p.wins + p.losses;
  wr.textContent = total > 0 ? (p.win_rate*100).toFixed(0)+'% ('+p.wins+'W-'+p.losses+'L)' : '--';
}

function renderMlVsSpread(mvs) {
  if (!mvs) return;
  const mt = document.getElementById('mlTotal');
  mt.textContent = fmtPct(mvs.ml_total, mvs.ml_risked) + ' (' + fmtCents(mvs.ml_total) + ')';
  mt.className = 'value ' + pnlClass(mvs.ml_total);

  const md = document.getElementById('mlDetail');
  md.textContent = fmtCents(mvs.ml_realized) + ' / ' + fmtCents(mvs.ml_unrealized) + ' | Risked: ' + fmtRisked(mvs.ml_risked);
  md.className = 'value ' + pnlClass(mvs.ml_realized);

  const st = document.getElementById('spreadTotal');
  st.textContent = fmtPct(mvs.spread_total, mvs.spread_risked) + ' (' + fmtCents(mvs.spread_total) + ')';
  st.className = 'value ' + pnlClass(mvs.spread_total);

  const sd = document.getElementById('spreadDetail');
  sd.textContent = fmtCents(mvs.spread_realized) + ' / ' + fmtCents(mvs.spread_unrealized) + ' | Risked: ' + fmtRisked(mvs.spread_risked);
  sd.className = 'value ' + pnlClass(mvs.spread_realized);
}

function isoToMinutes(iso, refTime) {
  if (!iso) return 0;
  try {
    const t = new Date(iso).getTime();
    return (t - refTime) / 60000;
  } catch(e) { return 0; }
}

function renderCharts(data) {
  const pnlEl = document.getElementById('pnlChart');
  const priceEl = document.getElementById('priceChart');
  const pnlLegend = document.getElementById('pnlLegend');
  const priceLegend = document.getElementById('priceLegend');

  const chartW = Math.min(600, (window.innerWidth - 80) / 2);

  // Collect all tickers across games
  let allTickers = [];
  (data.games||[]).forEach(g => { (g.tickers||[]).forEach((t,i) => { allTickers.push({...t, _idx:allTickers.length}); }); });

  if (allTickers.length === 0) {
    pnlEl.innerHTML = '<div class="waiting">Waiting for data...</div>';
    priceEl.innerHTML = '<div class="waiting">Waiting for data...</div>';
    return;
  }

  // Find earliest timestamp as reference
  let refTime = Infinity;
  allTickers.forEach(t => {
    (t.chart?.mid_series||[]).forEach(p => { const d=new Date(p.time).getTime(); if(d<refTime) refTime=d; });
    (t.chart?.pnl_series||[]).forEach(p => { const d=new Date(p.time).getTime(); if(d<refTime) refTime=d; });
  });
  if (refTime === Infinity) refTime = Date.now();

  // P&L chart — show ML total, Spread total, and Portfolio total lines
  let pnlSeries = [];
  pnlLegend.innerHTML = '';

  // Collect per-ticker trade events grouped by ML/Spread
  let mlTickerSeries = [];
  let spTickerSeries = [];
  allTickers.forEach(t => {
    const pts = (t.chart?.pnl_series||[]).map(p => ({x: isoToMinutes(p.time, refTime), cum: p.cum_pnl}));
    if (pts.length === 0) return;
    if (t.type === 'ML') mlTickerSeries.push(pts);
    else spTickerSeries.push(pts);
  });

  // Build a step-style cumulative line: starts at 0, flat between trades, steps on each event
  const nowMinutes = (Date.now() - refTime) / 60000;
  function buildCumLine(groups) {
    let events = [];
    groups.forEach((pts, idx) => { pts.forEach(p => events.push({x:p.x, idx:idx, cum:p.cum})); });
    if (events.length === 0) return [];
    events.sort((a,b) => a.x - b.x);
    let last = new Array(groups.length).fill(0);
    let out = [{x:0, y:0}];  // start at origin
    events.forEach(e => {
      let prevSum = 0; for (let i=0;i<last.length;i++) prevSum += last[i];
      out.push({x:e.x, y:prevSum});  // flat step to just before this event
      last[e.idx] = e.cum;
      let newSum = 0; for (let i=0;i<last.length;i++) newSum += last[i];
      out.push({x:e.x, y:newSum});   // jump to new value
    });
    // extend flat to current time
    let finalSum = 0; for (let i=0;i<last.length;i++) finalSum += last[i];
    out.push({x:nowMinutes, y:finalSum});
    return out;
  }

  const mlPoints = buildCumLine(mlTickerSeries);
  const spPoints = buildCumLine(spTickerSeries);
  const totalPoints = buildCumLine([...mlTickerSeries, ...spTickerSeries]);

  if (mlPoints.length > 0) {
    pnlSeries.push({label:'All ML', color:'#58a6ff', points:mlPoints, isTotal:false});
    pnlLegend.innerHTML += '<span class="legend-item"><span class="legend-dot" style="background:#58a6ff"></span>All ML</span>';
  }
  if (spPoints.length > 0) {
    pnlSeries.push({label:'All Spread', color:'#d29922', points:spPoints, isTotal:false});
    pnlLegend.innerHTML += '<span class="legend-item"><span class="legend-dot" style="background:#d29922"></span>All Spread</span>';
  }
  if (totalPoints.length > 0) {
    pnlSeries.push({label:'Total', color:'#e6edf3', points:totalPoints, isTotal:true});
    pnlLegend.innerHTML += '<span class="legend-item"><span class="legend-dot" style="background:#e6edf3"></span><b>Total</b></span>';
  }

  pnlEl.innerHTML = buildSVG(pnlSeries, {width:chartW, height:180, showFill:true});

  // Price chart
  let priceSeries = [];
  let priceMarkers = [];
  priceLegend.innerHTML = '';

  allTickers.forEach((t,idx) => {
    const c = colorFor(t.label, idx);
    const points = (t.chart?.mid_series||[]).map(p => ({x:isoToMinutes(p.time, refTime), y:p.mid}));
    if (points.length > 0) {
      priceSeries.push({label:t.label, color:c, points});
      priceLegend.innerHTML += '<span class="legend-item"><span class="legend-dot" style="background:'+c+'"></span>'+t.label+'</span>';
    }
    (t.markers||[]).forEach(m => {
      priceMarkers.push({x:isoToMinutes(m.time, refTime), y:m.price, type:m.type, side:m.side});
    });
  });

  priceEl.innerHTML = buildSVG(priceSeries, {width:chartW, height:180, markers:priceMarkers});
}

function renderReconciliation(data) {
  const banner = document.getElementById('reconBanner');
  const okBanner = document.getElementById('reconOk');
  const body = document.getElementById('reconBody');
  const recon = data.reconciliation || [];

  if (!recon.length) {
    banner.className = 'recon-banner';
    okBanner.className = 'recon-ok-banner';
    return;
  }

  body.innerHTML = '';
  recon.forEach(r => {
    const cls = r.status === 'MISMATCH' ? 'mismatch' : 'ok';
    body.innerHTML += '<tr><td>' + r.ticker + '</td><td>' + r.side.toUpperCase() +
      '</td><td>' + r.bot_qty + '</td><td>' + r.kalshi_qty +
      '</td><td class="' + cls + '">' + r.status + '</td></tr>';
  });

  if (data.has_mismatch) {
    banner.className = 'recon-banner visible';
    okBanner.className = 'recon-ok-banner';
  } else {
    banner.className = 'recon-banner';
    okBanner.className = 'recon-ok-banner visible';
  }
}

function renderGames(data) {
  const container = document.getElementById('gamesContainer');
  container.innerHTML = '';

  if (!data.games || data.games.length === 0) {
    container.innerHTML = '<div class="waiting">Waiting for game data...</div>';
    return;
  }

  const hideInactive = document.getElementById('hideInactive').checked;

  data.games.forEach(game => {
    // Skip inactive games: market settled/empty, no open positions, no capital risked
    if (hideInactive) {
      const hasActiveMarket = (game.tickers||[]).some(t => {
        const mid = t.mid || 0;
        return mid > 3 && mid < 97;  // not settled near 0 or 100
      });
      const hasPositions = (game.open_positions||[]).length > 0;
      const hasRisked = (game.tickers||[]).some(t => (t.capital_risked||0) > 0);
      if (!hasActiveMarket && !hasPositions && !hasRisked) return;
    }

    const sec = document.createElement('div');
    sec.className = 'game-section';

    const isTennis = (game.sport||'cbb') === 'tennis';
    const typeLabel = isTennis ? 'MATCH' : 'GAME';
    const progress = game.game_progress != null ? (game.game_progress*100).toFixed(0)+'%' : '--';
    const wp = game.espn_wp != null ? (game.espn_wp > 1 ? game.espn_wp.toFixed(0) : (game.espn_wp*100).toFixed(0))+'%' : '--';
    const scorePart = (!isTennis && game.score_display) ? '<span style="color:var(--white);font-size:14px;font-weight:bold;">'+game.score_display+'</span> <span style="color:var(--text2);font-size:12px;">'+( game.clock_display||'')+'</span> &nbsp; ' : '';
    const sportBadge = isTennis ? '<span class="tag" style="background:#1a3a1a;color:#3fb950;margin-right:6px;">TENNIS</span>' : '';

    let html = '<div class="game-header"><span class="title">'+sportBadge+typeLabel+': '+game.game+'</span>';
    const metaParts = [scorePart, 'Progress: '+progress];
    if (!isTennis) metaParts.push('ESPN WP: '+wp);
    html += '<span class="meta">'+metaParts.join(' &nbsp; ')+'</span></div>';

    // Ticker table
    html += '<h3>Tickers</h3>';
    html += '<table><tr><th>Ticker</th><th>Type</th><th>Bid/Ask</th><th>Mid</th><th>Sprd</th><th>MR Entry</th><th>Orders</th><th>Open</th><th>Risked</th><th>Realized</th><th>Unrealized</th><th>Total</th></tr>';
    (game.tickers||[]).forEach(t => {
      const typeTag = t.type==='ML'?'tag-ml':'tag-spread';
      const tickTotal = (t.realized||0) + (t.unrealized||0);
      const risked = t.capital_risked||0;
      html += '<tr>';
      html += '<td><b>'+t.label+'</b></td>';
      html += '<td><span class="tag '+typeTag+'">'+t.type+'</span></td>';
      html += '<td>'+(t.yes_bid||'--')+'/'+(t.yes_ask||'--')+'</td>';
      html += '<td>'+(t.mid?t.mid.toFixed(1):'--')+'</td>';
      html += '<td>'+(t.spread||'--')+'</td>';
      html += '<td>'+fmtSignal(t.mr_signal)+'</td>';
      html += '<td>'+fmtOrders(t.orders)+'</td>';
      html += '<td>'+t.open_count+'</td>';
      html += '<td>'+fmtRisked(risked)+'</td>';
      html += '<td class="'+pnlClass(t.realized)+'" title="'+fmtCents(t.realized)+' on '+fmtRisked(risked)+' risked">'+fmtPct(t.realized,risked)+'</td>';
      html += '<td class="'+pnlClass(t.unrealized)+'" title="'+fmtCents(t.unrealized)+' on '+fmtRisked(risked)+' risked">'+fmtPct(t.unrealized,risked)+'</td>';
      html += '<td class="'+pnlClass(tickTotal)+'" title="'+fmtCents(tickTotal)+' on '+fmtRisked(risked)+' risked"><b>'+fmtPct(tickTotal,risked)+'</b></td>';
      html += '</tr>';
    });

    // ML subtotal row
    const mlPnl = game.ml_pnl || {realized:0, unrealized:0, total:0, risked:0};
    html += '<tr style="border-top:2px solid #1f3a5f;background:#0d1a2d;">';
    html += '<td colspan="9" style="text-align:right;"><span class="tag tag-ml">ML</span> <b>Subtotal</b></td>';
    html += '<td class="'+pnlClass(mlPnl.realized)+'">'+fmtPct(mlPnl.realized,mlPnl.risked)+'</td>';
    html += '<td class="'+pnlClass(mlPnl.unrealized)+'">'+fmtPct(mlPnl.unrealized,mlPnl.risked)+'</td>';
    html += '<td class="'+pnlClass(mlPnl.total)+'"><b>'+fmtPct(mlPnl.total,mlPnl.risked)+' ('+fmtCents(mlPnl.total)+')</b></td>';
    html += '</tr>';

    // Spread subtotal row
    const spPnl = game.spread_pnl || {realized:0, unrealized:0, total:0, risked:0};
    html += '<tr style="border-top:2px solid #3b2e1a;background:#1a1508;">';
    html += '<td colspan="9" style="text-align:right;"><span class="tag tag-spread">SPREAD</span> <b>Subtotal</b></td>';
    html += '<td class="'+pnlClass(spPnl.realized)+'">'+fmtPct(spPnl.realized,spPnl.risked)+'</td>';
    html += '<td class="'+pnlClass(spPnl.unrealized)+'">'+fmtPct(spPnl.unrealized,spPnl.risked)+'</td>';
    html += '<td class="'+pnlClass(spPnl.total)+'"><b>'+fmtPct(spPnl.total,spPnl.risked)+' ('+fmtCents(spPnl.total)+')</b></td>';
    html += '</tr>';

    // Game total row
    const gameTotal = mlPnl.total + spPnl.total;
    const gameRisked = (mlPnl.risked||0) + (spPnl.risked||0);
    html += '<tr style="border-top:2px solid var(--border);background:var(--bg3);">';
    html += '<td colspan="9" style="text-align:right;"><b>Game Total</b></td>';
    html += '<td></td><td></td>';
    html += '<td class="'+pnlClass(gameTotal)+'"><b>'+fmtPct(gameTotal,gameRisked)+' ('+fmtCents(gameTotal)+')</b></td>';
    html += '</tr>';

    html += '</table>';

    // Per-ticker MR strategy charts
    (game.tickers||[]).forEach(t => {
      const mr = t.chart?.mr_series;
      if (!mr || mr.length < 2) return;
      const chartW = Math.min(800, window.innerWidth - 80);
      // Legend
      html += '<div style="margin-top:12px;margin-bottom:4px;display:flex;align-items:center;gap:16px;">';
      html += '<span style="font-size:12px;font-weight:bold;color:var(--text1);">MR: '+t.label+'</span>';
      html += '<span style="font-size:11px;color:#58a6ff;">&#9644; Mid</span>';
      html += '<span style="font-size:11px;color:#bc8cff;">&#9472;&#9472; Mean</span>';
      html += '<span style="font-size:11px;color:#6e4da0;">&#9618; Entry Bands</span>';
      html += '</div>';
      // Build markers for this ticker
      const mrMarkers = (t.markers||[]);
      html += '<div class="mr-chart">' + buildMRSVG(mr, mrMarkers, {width:chartW, height:200}) + '</div>';
    });

    // Open positions
    if (game.open_positions && game.open_positions.length > 0) {
      html += '<h3>Open Positions</h3>';
      html += '<table><tr><th>Strategy</th><th>Ticker</th><th>Side</th><th>Entry</th><th>Qty</th><th>Mark</th><th>Age</th><th>Unrealized</th></tr>';
      game.open_positions.forEach(p => {
        const sideTag = p.side==='yes'?'tag-yes':'tag-no';
        html += '<tr>';
        html += '<td>'+p.strategy+'</td>';
        html += '<td>'+p.ticker+'</td>';
        html += '<td><span class="tag '+sideTag+'">'+p.side.toUpperCase()+'</span></td>';
        html += '<td>'+p.entry_price.toFixed(1)+'c</td>';
        html += '<td>'+p.qty+'</td>';
        html += '<td>'+(p.mark_price?p.mark_price.toFixed(1)+'c':'--')+'</td>';
        html += '<td>'+fmtAge(p.age_secs)+'</td>';
        html += '<td class="'+pnlClass(p.unrealized_pnl)+'">'+fmtCents(p.unrealized_pnl)+'</td>';
        html += '</tr>';
      });
      html += '</table>';
    }

    // Closed position stats by strategy
    if (game.closed_stats && game.closed_stats.length > 0) {
      html += '<h3>Closed Positions (by Strategy)</h3>';
      html += '<table><tr><th>Strategy</th><th>Ticker</th><th>Trades</th><th>W-L</th><th>Locks</th><th>Stops</th><th>Net P&amp;L</th><th>Avg Hold</th></tr>';
      game.closed_stats.forEach(s => {
        html += '<tr>';
        html += '<td>'+s.strategy+'</td>';
        html += '<td>'+(s.ticker||'')+'</td>';
        html += '<td>'+s.trades+'</td>';
        html += '<td>'+s.wins+'W-'+s.losses+'L</td>';
        html += '<td>'+s.locks+'</td>';
        html += '<td>'+s.stops+'</td>';
        html += '<td class="'+pnlClass(s.net_pnl)+'">'+fmtCents(s.net_pnl)+fmtDollars(s.net_pnl)+'</td>';
        html += '<td>'+fmtAge(Math.round(s.avg_hold_secs))+'</td>';
        html += '</tr>';
      });
      html += '</table>';
    }

    // Events
    if (game.events && game.events.length > 0) {
      html += '<h3>Recent Events (last 20)</h3>';
      html += '<div style="max-height:180px;overflow-y:auto;">';
      game.events.slice().reverse().forEach(e => {
        html += '<div class="event-row">';
        html += '<span class="event-time">'+fmtTime(e.time)+'</span>';
        html += '<span class="event-strat">'+e.strategy+'</span>';
        html += '<span class="event-type">'+e.event+'</span>';
        html += '<span class="event-detail">'+e.detail+'</span>';
        html += '</div>';
      });
      html += '</div>';
    }

    sec.innerHTML = html;
    container.appendChild(sec);
  });
}

// ─── DATA FETCHING ───
async function fetchDates() {
  try {
    const resp = await fetch('/api/dates');
    const data = await resp.json();
    const sel = document.getElementById('dateSelect');
    sel.innerHTML = '';
    (data.dates||[]).forEach((d,i) => {
      const opt = document.createElement('option');
      opt.value = d; opt.textContent = d;
      if (i===0) opt.selected = true;  // most recent date with R2 data
      sel.appendChild(opt);
    });
    // Add today as an option if not present, but don't auto-select
    const today = new Date().toISOString().slice(0,10);
    if (!data.dates.includes(today)) {
      const opt = document.createElement('option');
      opt.value = today; opt.textContent = today + ' (today)';
      sel.prepend(opt);
    }
  } catch(e) { console.error('fetchDates error:', e); }
}

async function fetchData() {
  const date = document.getElementById('dateSelect').value;
  if (!date) return;
  const sportFilter = document.getElementById('sportSelect').value;
  try {
    const resp = await fetch('/api/data?date='+date);
    const data = await resp.json();
    lastFetchTime = Date.now();

    // Filter games by selected sport
    const allGames = data.games || [];
    if (sportFilter !== 'all') {
      data.games = allGames.filter(g => (g.sport || 'cbb') === sportFilter);
    }

    // Recompute portfolio totals for filtered games
    if (sportFilter !== 'all') {
      const port = {total_pnl:0, realized:0, unrealized:0, capital_risked:0, open_count:0, wins:0, losses:0, win_rate:0, roi_pct:0};
      const mvs = {ml_realized:0, ml_unrealized:0, ml_total:0, ml_risked:0, spread_realized:0, spread_unrealized:0, spread_total:0, spread_risked:0};
      data.games.forEach(g => {
        (g.tickers||[]).forEach(t => {
          port.realized += t.realized||0;
          port.unrealized += t.unrealized||0;
          port.capital_risked += t.capital_risked||0;
          port.open_count += t.open_count||0;
        });
        (g.closed_stats||[]).forEach(s => { port.wins += s.wins||0; port.losses += s.losses||0; });
        const ml = g.ml_pnl||{}; const sp = g.spread_pnl||{};
        mvs.ml_realized += ml.realized||0; mvs.ml_unrealized += ml.unrealized||0; mvs.ml_risked += ml.risked||0;
        mvs.spread_realized += sp.realized||0; mvs.spread_unrealized += sp.unrealized||0; mvs.spread_risked += sp.risked||0;
      });
      port.total_pnl = port.realized + port.unrealized;
      const total = port.wins + port.losses;
      port.win_rate = total > 0 ? port.wins / total : 0;
      port.roi_pct = port.capital_risked ? (port.total_pnl / port.capital_risked * 100) : 0;
      mvs.ml_total = mvs.ml_realized + mvs.ml_unrealized;
      mvs.spread_total = mvs.spread_realized + mvs.spread_unrealized;
      data.portfolio = port;
      data.ml_vs_spread = mvs;
    }

    lastData = data;
    renderPortfolio(data.portfolio || {});
    renderMlVsSpread(data.ml_vs_spread || {});
    renderCharts(data);
    renderGames(data);
    renderReconciliation(data);

    // Hide ML vs Spread row when no CBB games visible
    const hasCbb = (data.games||[]).some(g => (g.sport||'cbb') === 'cbb');
    document.getElementById('mlVsSpread').style.display = hasCbb ? '' : 'none';

    updateSync();
  } catch(e) {
    console.error('fetchData error:', e);
    document.getElementById('syncLabel').textContent = 'Error fetching data';
    document.getElementById('syncLabel').className = 'sync stale';
  }
}

function updateSync() {
  const el = document.getElementById('syncLabel');
  if (!lastFetchTime) { el.textContent = 'Connecting...'; return; }
  const ago = Math.round((Date.now() - lastFetchTime)/1000);
  el.textContent = 'Last sync: ' + ago + 's ago';
  el.className = ago > 30 ? 'sync stale' : 'sync';
}

// ─── INIT ───
fetchDates().then(() => fetchData());
setInterval(fetchData, REFRESH_MS);
setInterval(updateSync, 1000);
document.getElementById('dateSelect').addEventListener('change', fetchData);
document.getElementById('sportSelect').addEventListener('change', fetchData);
document.getElementById('hideInactive').addEventListener('change', () => { if (lastData) renderGames(lastData); });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
