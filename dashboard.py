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

import requests as _requests

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# ESPN LOOKUP (from games.json)
# =============================================================================

GAMES_JSON = Path(__file__).parent / "games.json"


def _load_espn_lookup():
    """Map game labels (underscore form) to ESPN config."""
    try:
        games = json.loads(GAMES_JSON.read_text())
        lookup = {}
        for g in games:
            key = g["label"].replace(" ", "_")
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

_espn_bg_lock = threading.Lock()
_espn_bg_pending = set()  # dates currently being fetched in background


def _fetch_all_scores_sync(espn_date):
    """Blocking ESPN fetch — called from background thread."""
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        resp = _requests.get(url, params={"dates": espn_date, "groups": "50", "limit": "500"},
                             timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        data = resp.json()
    except Exception:
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

    cache_key = f"__espn_scores__{espn_date}"
    _cache.put(cache_key, scores)
    with _espn_bg_lock:
        _espn_bg_pending.discard(espn_date)
    return scores


def _fetch_all_scores(espn_date):
    """
    Non-blocking ESPN fetch. Returns cached scores immediately if available,
    otherwise kicks off a background thread and returns {} (scores appear next refresh).
    """
    cache_key = f"__espn_scores__{espn_date}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    # Kick off background fetch if not already running
    with _espn_bg_lock:
        if espn_date not in _espn_bg_pending:
            _espn_bg_pending.add(espn_date)
            threading.Thread(target=_fetch_all_scores_sync, args=(espn_date,), daemon=True).start()

    return {}


def _get_game_score(game_key):
    """
    Get score info for a game. Returns dict with team_score, opp_score,
    score_display, clock_display, or empty dict if unavailable.
    """
    cfg = ESPN_LOOKUP.get(game_key)
    if not cfg:
        return {}

    all_scores = _fetch_all_scores(cfg["espn_date"])
    if not all_scores:
        return {}

    info = all_scores.get(cfg["espn_team"])
    if not info:
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

def compute_mr_signal(snapshots):
    """
    Compute MR proximity from snapshot mid values.
    Matches MR strategy: lookback=60, low_vol_std_mult=2.5 (std<5), high_vol_std_mult=1.5.
    """
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

    window = mids[-60:]  # last 60 samples (MR lookback)
    n = len(window)
    mean = sum(window) / n
    variance = sum((x - mean) ** 2 for x in window) / n
    std = variance ** 0.5

    if std < 0.1:
        return {"mr_mean": round(mean, 2), "mr_std": 0, "mr_threshold": 0,
                "mr_deviation": 0, "mr_pct": 0, "status": "dead"}

    std_mult = 2.5 if std < 5.0 else 1.5
    threshold = std_mult * std
    deviation = mids[-1] - mean

    pct = abs(deviation) / threshold * 100 if threshold > 0 else 0

    return {
        "mr_mean": round(mean, 2),
        "mr_std": round(std, 2),
        "mr_threshold": round(threshold, 2),
        "mr_deviation": round(deviation, 2),
        "mr_pct": round(min(pct, 999), 1),
    }


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
                        })
                        seen_games.add(game)
                    elif _has_csv_files(game_prefix):
                        # Old nested: CSVs directly in game folder
                        games.append({
                            "game": game,
                            "tickers": ["_root"],
                            "prefix_override": game_prefix,
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

        if action == "entry_fill":
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


def build_chart_data(snapshots, positions_rows, ticker_label):
    """
    Build chart series for one ticker.
    - mid_series: downsampled mid values from snapshots.csv (~60 points)
    - pnl_series: cumulative P&L from positions.csv (one point per closed trade)
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

    return {"mid_series": mid_series, "pnl_series": pnl_series}


def build_trade_markers(trades, positions_rows):
    """Build entry/exit markers for chart overlay."""
    markers = []
    for t in trades:
        action = t.get("action", "")
        if action == "entry_fill":
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
            "open_count": 0,
            "wins": 0,
            "losses": 0,
        },
        "ml_vs_spread": {
            "ml_realized": 0.0,
            "ml_unrealized": 0.0,
            "ml_total": 0.0,
            "spread_realized": 0.0,
            "spread_unrealized": 0.0,
            "spread_total": 0.0,
        },
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }

    for game_info in games:
        game_key = game_info["game"]
        game_data = {
            "game": game_key.replace("_", " "),
            "tickers": [],
            "events": [],
            "game_progress": None,
            "espn_wp": None,
        }

        # ESPN live scores
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
            # Tag each with ticker
            for p in open_pos:
                p["ticker"] = ticker_label
            open_pos = mark_to_market(open_pos, latest_snap)

            # Closed position stats
            closed_stats = aggregate_closed_positions(positions)
            for s in closed_stats:
                s["ticker"] = ticker_label

            # Ticker-level realized / unrealized
            realized = sum(safe_float(r.get("net_pnl")) for r in positions)
            unrealized = sum(p.get("unrealized_pnl", 0) for p in open_pos)

            # Bid/ask from latest snapshot
            yes_bid = safe_float(latest_snap.get("yes_bid")) if latest_snap else 0
            yes_ask = safe_float(latest_snap.get("yes_ask")) if latest_snap else 0
            no_bid = safe_float(latest_snap.get("no_bid")) if latest_snap else 0
            no_ask = safe_float(latest_snap.get("no_ask")) if latest_snap else 0
            mid = safe_float(latest_snap.get("mid")) if latest_snap else 0
            spread = safe_float(latest_snap.get("spread")) if latest_snap else 0

            # Charts
            chart_data = build_chart_data(snapshots, positions, ticker_label)
            trade_markers = build_trade_markers(trades, positions)

            # MR signal proximity
            try:
                mr_signal = compute_mr_signal(snapshots)
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
                "chart": chart_data,
                "markers": trade_markers,
                "mr_signal": mr_signal,
            }
            game_data["tickers"].append(ticker_data)

            all_open_positions.extend(open_pos)
            all_closed_stats.extend(closed_stats)

            # Portfolio totals
            result["portfolio"]["realized"] += realized
            result["portfolio"]["unrealized"] += unrealized
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
        sp_real = sum(t["realized"] for t in game_data["tickers"] if t["type"] == "Spread")
        sp_unreal = sum(t["unrealized"] for t in game_data["tickers"] if t["type"] == "Spread")
        game_data["ml_pnl"] = {"realized": ml_real, "unrealized": ml_unreal, "total": ml_real + ml_unreal}
        game_data["spread_pnl"] = {"realized": sp_real, "unrealized": sp_unreal, "total": sp_real + sp_unreal}

        # Accumulate into global ML vs Spread
        result["ml_vs_spread"]["ml_realized"] += ml_real
        result["ml_vs_spread"]["ml_unrealized"] += ml_unreal
        result["ml_vs_spread"]["spread_realized"] += sp_real
        result["ml_vs_spread"]["spread_unrealized"] += sp_unreal
        # Skip games with no meaningful data (empty CSVs / failed runs)
        has_snapshots = any(t.get("mid", 0) != 0 for t in game_data["tickers"])
        has_trades = any(t.get("realized", 0) != 0 or t.get("open_count", 0) > 0 for t in game_data["tickers"])
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

    mvs = result["ml_vs_spread"]
    mvs["ml_total"] = mvs["ml_realized"] + mvs["ml_unrealized"]
    mvs["spread_total"] = mvs["spread_realized"] + mvs["spread_unrealized"]

    return result


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
                data = build_dashboard_data(date_str)
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
    port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "8050")))
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
.portfolio { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:10px; }
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
</style>
</head>
<body>

<div class="header">
  <h1>KALSHI LIVE DASHBOARD</h1>
  <div class="header-right">
    <span class="sync" id="syncLabel">Connecting...</span>
    <label>Date: <select id="dateSelect"></select></label>
  </div>
</div>

<div class="portfolio" id="portfolio">
  <div class="stat-card"><div class="label">Total P&amp;L</div><div class="value" id="totalPnl">--</div></div>
  <div class="stat-card"><div class="label">Realized</div><div class="value" id="realized">--</div></div>
  <div class="stat-card"><div class="label">Unrealized</div><div class="value" id="unrealized">--</div></div>
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

<div id="gamesContainer"></div>

<script>
// ─── CONFIG ───
const REFRESH_MS = 20000;
const COLORS = {LOU:'#58a6ff',LOU2:'#3fb950',LOU5:'#d29922',UNC1:'#bc8cff'};
const DEFAULT_COLORS = ['#58a6ff','#3fb950','#d29922','#bc8cff','#f778ba','#79c0ff'];
let lastFetchTime = null;

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

function fmtSignal(sig) {
  if (!sig) return '<span style="color:var(--text2)">--</span>';
  if (sig.status === 'dead') return '<span style="color:var(--text2)">DEAD</span>';
  const dev = sig.mr_deviation;
  const thr = sig.mr_threshold;
  const pct = sig.mr_pct;
  const dir = dev > 0 ? 'HIGH' : 'LOW';
  const absDev = Math.abs(dev).toFixed(1);
  const dist = Math.abs(Math.abs(dev) - thr).toFixed(1);

  if (pct >= 100) {
    return '<span style="color:#f85149;font-weight:bold;">TRIGGERED (' + dir + ')</span>';
  }
  // Color: green < 50%, yellow 50-80%, orange 80-95%, red 95%+
  let color = '#3fb950';
  if (pct >= 95) color = '#f85149';
  else if (pct >= 80) color = '#d29922';
  else if (pct >= 50) color = '#e3b341';

  // Mini bar
  const barW = Math.min(pct, 100);
  const bar = '<span style="display:inline-block;width:40px;height:6px;background:var(--bg3);border-radius:3px;vertical-align:middle;margin-left:4px;">'
    + '<span style="display:block;width:'+barW+'%;height:100%;background:'+color+';border-radius:3px;"></span></span>';
  return '<span style="color:'+color+'">'+dist+'c ('+dir+')'+bar+'</span>';
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

// ─── RENDER FUNCTIONS ───
function renderPortfolio(p) {
  const tp = document.getElementById('totalPnl');
  tp.textContent = fmtCents(p.total_pnl) + fmtDollars(p.total_pnl);
  tp.className = 'value ' + pnlClass(p.total_pnl);

  const r = document.getElementById('realized');
  r.textContent = fmtCents(p.realized);
  r.className = 'value ' + pnlClass(p.realized);

  const u = document.getElementById('unrealized');
  u.textContent = fmtCents(p.unrealized);
  u.className = 'value ' + pnlClass(p.unrealized);

  document.getElementById('openCount').textContent = p.open_count;

  const wr = document.getElementById('winRate');
  const total = p.wins + p.losses;
  wr.textContent = total > 0 ? (p.win_rate*100).toFixed(0)+'% ('+p.wins+'W-'+p.losses+'L)' : '--';
}

function renderMlVsSpread(mvs) {
  if (!mvs) return;
  const mt = document.getElementById('mlTotal');
  mt.textContent = fmtCents(mvs.ml_total) + fmtDollars(mvs.ml_total);
  mt.className = 'value ' + pnlClass(mvs.ml_total);

  const md = document.getElementById('mlDetail');
  md.textContent = fmtCents(mvs.ml_realized) + ' / ' + fmtCents(mvs.ml_unrealized);
  md.className = 'value ' + pnlClass(mvs.ml_realized);

  const st = document.getElementById('spreadTotal');
  st.textContent = fmtCents(mvs.spread_total) + fmtDollars(mvs.spread_total);
  st.className = 'value ' + pnlClass(mvs.spread_total);

  const sd = document.getElementById('spreadDetail');
  sd.textContent = fmtCents(mvs.spread_realized) + ' / ' + fmtCents(mvs.spread_unrealized);
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

  let mlCumByTime = {};
  let spCumByTime = {};
  let totalCumByTime = {};

  allTickers.forEach(t => {
    (t.chart?.pnl_series||[]).forEach(p => {
      const x = isoToMinutes(p.time, refTime).toFixed(2);
      if (!totalCumByTime[x]) totalCumByTime[x] = 0;
      totalCumByTime[x] += p.cum_pnl;
      if (t.type === 'ML') {
        if (!mlCumByTime[x]) mlCumByTime[x] = 0;
        mlCumByTime[x] += p.cum_pnl;
      } else {
        if (!spCumByTime[x]) spCumByTime[x] = 0;
        spCumByTime[x] += p.cum_pnl;
      }
    });
  });

  // ML total line
  const mlSorted = Object.entries(mlCumByTime).map(([x,y])=>({x:parseFloat(x),y})).sort((a,b)=>a.x-b.x);
  if (mlSorted.length > 0) {
    pnlSeries.push({label:'All ML', color:'#58a6ff', points:mlSorted, isTotal:false});
    pnlLegend.innerHTML += '<span class="legend-item"><span class="legend-dot" style="background:#58a6ff"></span>All ML</span>';
  }

  // Spread total line
  const spSorted = Object.entries(spCumByTime).map(([x,y])=>({x:parseFloat(x),y})).sort((a,b)=>a.x-b.x);
  if (spSorted.length > 0) {
    pnlSeries.push({label:'All Spread', color:'#d29922', points:spSorted, isTotal:false});
    pnlLegend.innerHTML += '<span class="legend-item"><span class="legend-dot" style="background:#d29922"></span>All Spread</span>';
  }

  // Portfolio total line
  const totalSorted = Object.entries(totalCumByTime).map(([x,y])=>({x:parseFloat(x),y})).sort((a,b)=>a.x-b.x);
  if (totalSorted.length > 0) {
    pnlSeries.push({label:'Total', color:'#e6edf3', points:totalSorted, isTotal:true});
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

function renderGames(data) {
  const container = document.getElementById('gamesContainer');
  container.innerHTML = '';

  if (!data.games || data.games.length === 0) {
    container.innerHTML = '<div class="waiting">Waiting for game data...</div>';
    return;
  }

  data.games.forEach(game => {
    const sec = document.createElement('div');
    sec.className = 'game-section';

    const progress = game.game_progress != null ? (game.game_progress*100).toFixed(0)+'%' : '--';
    const wp = game.espn_wp != null ? (game.espn_wp > 1 ? game.espn_wp.toFixed(0) : (game.espn_wp*100).toFixed(0))+'%' : '--';
    const scorePart = game.score_display ? '<span style="color:var(--white);font-size:14px;font-weight:bold;">'+game.score_display+'</span> <span style="color:var(--text2);font-size:12px;">'+( game.clock_display||'')+'</span> &nbsp; ' : '';

    let html = '<div class="game-header"><span class="title">GAME: '+game.game+'</span>';
    html += '<span class="meta">'+scorePart+'Progress: '+progress+' &nbsp; ESPN WP: '+wp+'</span></div>';

    // Ticker table
    html += '<h3>Tickers</h3>';
    html += '<table><tr><th>Ticker</th><th>Type</th><th>Bid/Ask</th><th>Mid</th><th>Sprd</th><th>Signal</th><th>Open</th><th>Realized</th><th>Unrealized</th><th>Total</th></tr>';
    (game.tickers||[]).forEach(t => {
      const typeTag = t.type==='ML'?'tag-ml':'tag-spread';
      const tickTotal = (t.realized||0) + (t.unrealized||0);
      html += '<tr>';
      html += '<td><b>'+t.label+'</b></td>';
      html += '<td><span class="tag '+typeTag+'">'+t.type+'</span></td>';
      html += '<td>'+(t.yes_bid||'--')+'/'+(t.yes_ask||'--')+'</td>';
      html += '<td>'+(t.mid?t.mid.toFixed(1):'--')+'</td>';
      html += '<td>'+(t.spread||'--')+'</td>';
      html += '<td>'+fmtSignal(t.mr_signal)+'</td>';
      html += '<td>'+t.open_count+'</td>';
      html += '<td class="'+pnlClass(t.realized)+'">'+fmtCents(t.realized)+'</td>';
      html += '<td class="'+pnlClass(t.unrealized)+'">'+fmtCents(t.unrealized)+'</td>';
      html += '<td class="'+pnlClass(tickTotal)+'"><b>'+fmtCents(tickTotal)+'</b></td>';
      html += '</tr>';
    });

    // ML subtotal row
    const mlPnl = game.ml_pnl || {realized:0, unrealized:0, total:0};
    html += '<tr style="border-top:2px solid #1f3a5f;background:#0d1a2d;">';
    html += '<td colspan="7" style="text-align:right;"><span class="tag tag-ml">ML</span> <b>Subtotal</b></td>';
    html += '<td class="'+pnlClass(mlPnl.realized)+'">'+fmtCents(mlPnl.realized)+'</td>';
    html += '<td class="'+pnlClass(mlPnl.unrealized)+'">'+fmtCents(mlPnl.unrealized)+'</td>';
    html += '<td class="'+pnlClass(mlPnl.total)+'"><b>'+fmtCents(mlPnl.total)+fmtDollars(mlPnl.total)+'</b></td>';
    html += '</tr>';

    // Spread subtotal row
    const spPnl = game.spread_pnl || {realized:0, unrealized:0, total:0};
    html += '<tr style="border-top:2px solid #3b2e1a;background:#1a1508;">';
    html += '<td colspan="7" style="text-align:right;"><span class="tag tag-spread">SPREAD</span> <b>Subtotal</b></td>';
    html += '<td class="'+pnlClass(spPnl.realized)+'">'+fmtCents(spPnl.realized)+'</td>';
    html += '<td class="'+pnlClass(spPnl.unrealized)+'">'+fmtCents(spPnl.unrealized)+'</td>';
    html += '<td class="'+pnlClass(spPnl.total)+'"><b>'+fmtCents(spPnl.total)+fmtDollars(spPnl.total)+'</b></td>';
    html += '</tr>';

    // Game total row
    const gameTotal = mlPnl.total + spPnl.total;
    html += '<tr style="border-top:2px solid var(--border);background:var(--bg3);">';
    html += '<td colspan="7" style="text-align:right;"><b>Game Total</b></td>';
    html += '<td></td><td></td>';
    html += '<td class="'+pnlClass(gameTotal)+'"><b>'+fmtCents(gameTotal)+fmtDollars(gameTotal)+'</b></td>';
    html += '</tr>';

    html += '</table>';

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
      if (i===0) opt.selected = true;
      sel.appendChild(opt);
    });
    // Also add today if not present
    const today = new Date().toISOString().slice(0,10);
    if (!data.dates.includes(today)) {
      const opt = document.createElement('option');
      opt.value = today; opt.textContent = today + ' (today)';
      opt.selected = true;
      sel.prepend(opt);
    }
  } catch(e) { console.error('fetchDates error:', e); }
}

async function fetchData() {
  const date = document.getElementById('dateSelect').value;
  if (!date) return;
  try {
    const resp = await fetch('/api/data?date='+date);
    const data = await resp.json();
    lastFetchTime = Date.now();
    renderPortfolio(data.portfolio || {});
    renderMlVsSpread(data.ml_vs_spread || {});
    renderCharts(data);
    renderGames(data);
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
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
