# tonight_runner_cloud.py
# Render background-worker version of tonight_runner.py
#
# Key differences vs local runner:
# - No interactive "press ENTER"
# - Uses a writable WORKDIR (Render ephemeral disk) and syncs ./logs to Cloudflare R2
# - Loads games from JSON (recommended) or falls back to an in-file GAMES list
# - Periodic uploader thread keeps R2 up to date during the run

import os
import sys
import json
import time
import threading
import signal
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import datetime as dt
import time

# --------------------------------------------------------------------------------------
# Ensure imports work even if we chdir into an ephemeral workdir
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from combo_vnext import (
    _load_private_key,
    _get,
    get_markets_in_series,
    fetch_market,
    parse_iso,
    utc_now,
    print_status,
    SERIES_TICKER,
)

from espn_game_clock import EspnGameClock
from production_strategies import (
    MeanReversionStrategy,
    ExposureTracker,
    GameRunner,
)


# --------------------------------------------------------------------------------------
# Optional: Cloudflare R2 uploader (S3-compatible via boto3)
# --------------------------------------------------------------------------------------
from pathlib import Path
import re
from datetime import datetime

LOG_ROOT = Path(os.getenv("KALSHI_LOG_ROOT", "kalshi-logs"))

def safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)

def build_game_log_dir(
    *,
    sport: str,
    event: str,
    game_label: str,
    game_date: str,   # YYYYMMDD or YYYY-MM-DD
) -> Path:
    if len(game_date) == 8:  # YYYYMMDD
        game_date = f"{game_date[:4]}-{game_date[4:6]}-{game_date[6:]}"
    return (
        LOG_ROOT
        / "kalshi"
        / safe_name(sport)
        / safe_name(event)
        / game_date
        / safe_name(game_label)
    )


def _safe_import_boto3():
    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore
        return boto3, Config
    except Exception:
        return None, None
    

class R2Uploader:
    """
    Minimal "sync this local logs folder to R2" uploader.

    Requires env vars:
      R2_ENDPOINT_URL      e.g. https://<accountid>.r2.cloudflarestorage.com
      R2_ACCESS_KEY_ID
      R2_SECRET_ACCESS_KEY
      R2_BUCKET
    Optional:
      R2_PREFIX            e.g. "kalshi/logs"  (default: "kalshi/logs")
      R2_REGION            default: "auto"
      R2_PUBLIC_BASE       optional base URL used only for printing links
    """

    def __init__(self, local_logs_dir: Path):
        self.local_logs_dir = local_logs_dir

        self.endpoint_url = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key_id = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET", "").strip()

        self.prefix = (os.getenv("R2_PREFIX") or "kalshi/logs").strip().strip("/")
        self.region = (os.getenv("R2_REGION") or "auto").strip()
        self.public_base = os.getenv("R2_PUBLIC_BASE", "").strip().rstrip("/")

        self._enabled = all([self.endpoint_url, self.access_key_id, self.secret_access_key, self.bucket])
        self._client = None

        self._manifest_path = self.local_logs_dir / ".r2_manifest.json"
        self._manifest: Dict[str, Any] = self._load_manifest()

        if self._enabled:
            boto3, Config = _safe_import_boto3()
            if boto3 is None:
                print_status("[R2] boto3 not installed. Add boto3 to requirements to enable uploads.")
                self._enabled = False
            else:
                # Cloudflare R2 is S3-compatible; region "auto" is commonly used.
                self._client = boto3.client(
                    "s3",
                    region_name=self.region,
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    config=Config(signature_version="s3v4"),
                )
                print_status(f"[R2] Enabled. bucket={self.bucket} prefix={self.prefix}")
        else:
            print_status("[R2] Not enabled (missing env vars). Logs will remain local-only.")

    def enabled(self) -> bool:
        return bool(self._enabled and self._client)

    def _load_manifest(self) -> Dict[str, Any]:
        try:
            if self._manifest_path.exists():
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {"files": {}}  # key -> {"size":..., "mtime":..., "sha1":...}

    def _save_manifest(self) -> None:
        try:
            self._manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _sha1_file(p: Path) -> str:
        h = hashlib.sha1()
        with p.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _key_for(self, rel_path: str) -> str:
        rel_path = rel_path.lstrip("/").replace("\\", "/")
        return f"{self.prefix}/{rel_path}"

    def upload_changed(self, max_files: int = 500) -> int:
        """
        Upload any file whose (size, mtime) changed vs manifest.
        We compute sha1 only on changed candidates to avoid excess CPU.
        """
        if not self.enabled():
            return 0

        uploaded = 0
        files_db: Dict[str, Any] = self._manifest.get("files", {})

        # Walk local logs
        all_files: List[Path] = []
        for p in self.local_logs_dir.rglob("*"):
            if p.is_file():
                # skip the manifest itself
                if p.name == ".r2_manifest.json":
                    continue
                all_files.append(p)

        # Sort older first (nice for long runs)
        all_files.sort(key=lambda x: x.stat().st_mtime)

        for p in all_files:
            if uploaded >= max_files:
                break

            rel = str(p.relative_to(self.local_logs_dir)).replace("\\", "/")
            st = p.stat()
            size = int(st.st_size)
            mtime = int(st.st_mtime)

            prev = files_db.get(rel) or {}
            if prev.get("size") == size and prev.get("mtime") == mtime:
                continue

            # changed candidate → confirm with sha1 (optional but safer)
            sha1 = self._sha1_file(p)
            if prev.get("sha1") == sha1:
                # mtime changed but content same
                files_db[rel] = {"size": size, "mtime": mtime, "sha1": sha1}
                continue

            key = self._key_for(rel)
            try:
                extra_args = {}
                # Light content-type help
                if p.suffix.lower() == ".json":
                    extra_args["ContentType"] = "application/json"
                elif p.suffix.lower() == ".csv":
                    extra_args["ContentType"] = "text/csv"
                elif p.suffix.lower() == ".txt":
                    extra_args["ContentType"] = "text/plain"

                self._client.upload_file(str(p), self.bucket, key, ExtraArgs=extra_args or None)
                files_db[rel] = {"size": size, "mtime": mtime, "sha1": sha1}
                uploaded += 1
            except Exception as e:
                print_status(f"[R2] Upload failed for {rel}: {e}")

        self._manifest["files"] = files_db
        self._save_manifest()

        if uploaded:
            print_status(f"[R2] Uploaded {uploaded} file(s)")
        return uploaded

    CONFIG_R2_KEY = "kalshi/config/strategy_config.json"
    MATCHES_R2_KEY = "kalshi/config/tennis_matches.json"

    def download_config(self, local_path: Path) -> bool:
        """Download strategy_config.json from R2 if content changed. Returns True if file was updated."""
        return self._download_r2_file(self.CONFIG_R2_KEY, local_path, "Config")

    def download_matches(self, local_path: Path) -> bool:
        """Download tennis_matches.json from R2 if content changed. Returns True if file was updated."""
        return self._download_r2_file(self.MATCHES_R2_KEY, local_path, "Matches")

    def _download_r2_file(self, r2_key: str, local_path: Path, label: str) -> bool:
        """Generic R2 file download with change detection."""
        if not self.enabled():
            return False
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=r2_key)
            data = resp["Body"].read()
        except Exception as e:
            if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e):
                return False
            print_status(f"[R2] download_{label.lower()} error: {e}")
            return False

        new_hash = hashlib.sha256(data).hexdigest()
        if local_path.exists():
            existing_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()
            if existing_hash == new_hash:
                return False

        local_path.write_bytes(data)
        print_status(f"[R2] {label} updated: {local_path}")
        return True

    def write_index(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Writes a small index.json into the logs folder (and then uploader sync will ship it).
        Useful for a future dashboard to discover latest files.
        """
        idx = {
            "updated_utc": utc_now().isoformat(),
            "bucket": self.bucket,
            "prefix": self.prefix,
        }
        if extra:
            idx.update(extra)

        try:
            (self.local_logs_dir / "index.json").write_text(json.dumps(idx, indent=2), encoding="utf-8")
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Cloud/worker logging directory setup
# --------------------------------------------------------------------------------------
def setup_workdir() -> Tuple[Path, Path]:
    """
    Use a writable directory on Render (ephemeral disk).
    We chdir into it so production_strategies keeps writing to ./logs automatically.
    """
    workdir = Path(os.getenv("WORKDIR") or "/tmp/kalshi_worker").resolve()
    logs_dir = workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Change working directory so all existing code that writes "logs/..." just works.
    os.chdir(str(workdir))
    print_status(f"[WORKDIR] cwd={workdir} logs={logs_dir}")
    return workdir, logs_dir


# --------------------------------------------------------------------------------------
# MARKET RESOLUTION (same as tonight_runner.py)
# --------------------------------------------------------------------------------------
def find_market_for_team(private_key, team_name: str) -> Tuple[str, Dict[str, Any]]:
    """Find the ML market for a given team (ticker suffix match, then title fallback)."""
    markets = get_markets_in_series(private_key, SERIES_TICKER)
    team_code = team_name.upper()

    suffix = f"-{team_code}"
    candidates = [m for m in markets if (m.get("ticker") or "").upper().endswith(suffix)]

    if not candidates:
        team_l = team_name.lower()
        for m in markets:
            title = (m.get("title") or "").lower()
            if team_l in title:
                candidates.append(m)

    if not candidates:
        raise RuntimeError(f"No markets found for team '{team_name}'")

    now = utc_now()
    future = [m for m in candidates if parse_iso(m["close_time"]) > now]
    if future:
        candidates = future

    candidates.sort(key=lambda m: parse_iso(m["close_time"]))
    chosen = candidates[0]
    return chosen["ticker"], chosen


# --------------------------------------------------------------------------------------
# ESPN CLOCK
# --------------------------------------------------------------------------------------
def setup_espn_clock(game_config: Dict[str, Any]):
    try:
        clock = EspnGameClock(
            yyyymmdd=game_config["espn_date"],
            team_code=game_config["espn_team"],
            opponent_code=game_config.get("espn_opponent"),
            cache_ttl_secs=10,
        )
        print_status(
            f"[{game_config['label']}] ESPN clock: "
            f"{game_config['espn_team']} vs {game_config.get('espn_opponent', '?')}"
        )
        return clock
    except Exception as e:
        print_status(f"[{game_config['label']}] ESPN clock failed: {e}")
        return None


# --------------------------------------------------------------------------------------
# GAMES: recommended to load from JSON for cloud worker
# --------------------------------------------------------------------------------------
DEFAULT_GAMES: List[Dict[str, Any]] = [
    # Keep as example/fallback; in production use GAMES_JSON_PATH or GAMES_JSON env.
    {
        "label": "USC at Penn St.",
        "team_name": "USC",
        "model_p_win": 0.55,
        "partner_p_win": 0.55,
        "segment": "Home Fav",
        "espn_date": "20260208",
        "espn_team": "USC",
        "espn_opponent": "PSU",
    }
]


def load_games() -> List[Dict[str, Any]]:
    """
    Priority:
      1) env GAMES_JSON (inline JSON)
      2) file path env GAMES_JSON_PATH
      3) fallback DEFAULT_GAMES
    """
    inline = os.getenv("GAMES_JSON", "").strip()
    if inline:
        try:
            data = json.loads(inline)
            if isinstance(data, list):
                return data
        except Exception as e:
            print_status(f"[GAMES] Invalid GAMES_JSON: {e}")

    path = os.getenv("GAMES_JSON_PATH", "").strip()
    if path:
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception as e:
                print_status(f"[GAMES] Invalid JSON file {path}: {e}")
        else:
            print_status(f"[GAMES] GAMES_JSON_PATH not found: {path}")

    return list(DEFAULT_GAMES)


# --------------------------------------------------------------------------------------
# Private key loading helpers (Render-friendly)
# --------------------------------------------------------------------------------------
def load_private_key_for_worker() -> Any:
    """
    Supports either:
      - KALSHI_PRIVATE_KEY_PATH: path to PEM file (Render secret file / mounted disk)
    """
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not key_path:
        raise RuntimeError("Set KALSHI_PRIVATE_KEY_PATH in environment.")
    return _load_private_key(key_path)


# --------------------------------------------------------------------------------------
# Per-game worker
# --------------------------------------------------------------------------------------
def wait_for_espn_live(label: str, espn_clock: Optional[EspnGameClock]) -> bool:
    if espn_clock is None:
        print_status(f"[{label}] No ESPN clock configured — starting immediately.")
        return True

    poll = int(os.getenv("ESPN_LIVE_POLL_SECS", "15"))
    grace_after_tip = int(os.getenv("ESPN_LIVE_GRACE_AFTER_TIP_SECS", "1800"))  # 30 min

    last_print = 0.0
    tip_seen = False
    tip_epoch = None

    while True:
        ctx = espn_clock.get_live_context() or {}
        state = (ctx.get("state") or "").lower()
        secs_to_tip = ctx.get("secs_to_tip")
        secs_to_end = ctx.get("secs_to_game_end")

        # LIVE condition
        if state == "in" and secs_to_end is not None:
            print_status(f"[{label}] ESPN is LIVE — starting trading/logging now.")
            return True

        now = time.time()

        # Track when tip *should* have happened
        if secs_to_tip is not None:
            if not tip_seen:
                tip_epoch = now + secs_to_tip
                tip_seen = True
            else:
                tip_epoch = min(tip_epoch, now + secs_to_tip)

        # Print status at most once per minute
        if now - last_print > 60:
            if state == "pre" and secs_to_tip is not None:
                print_status(f"[{label}] Waiting for ESPN live... tip in ~{int(secs_to_tip)}s")
            else:
                print_status(f"[{label}] Waiting for ESPN live... state={state or 'unknown'} why={ctx.get('why')}")
            last_print = now

        # Skip condition: tip passed + grace window exceeded
        if tip_seen and tip_epoch is not None:
            if now > tip_epoch + grace_after_tip:
                print_status(
                    f"[{label}] ESPN never went live "
                    f"{int((now - tip_epoch) / 60)} min after scheduled tip — skipping."
                )
                return False

        time.sleep(max(5, poll))


def _aggregate_sub_results(sub_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge per-ticker runner summaries into one game result.

    Each ticker's strategy names are namespaced (e.g. paired_mr/LOU, mean_reversion/LOU2)
    so the final report shows per-ticker P&L breakdown.
    """
    merged: Dict[str, Any] = {"strategies": {}}
    for ticker_label, summary in sub_results.items():
        if "error" in summary:
            merged["strategies"][f"ERROR/{ticker_label}"] = summary
            continue
        for strat_name, stats in summary.get("strategies", {}).items():
            merged["strategies"][f"{strat_name}/{ticker_label}"] = stats
    return merged


def _run_sub_ticker(
    *,
    label: str,
    ticker_label: str,
    ticker: str,
    market: Dict[str, Any],
    strategies: list,
    private_key,
    espn_clock,
    log_dir: Path,
    sub_results: Dict[str, Any],
    maker_entries: bool = False,
    maker_exits: bool = False,
    min_entry_price: int = 0,
):
    """Run a single GameRunner for one ticker. Called in its own thread."""
    try:
        sub_log_dir = log_dir / safe_name(ticker_label)
        sub_log_dir.mkdir(parents=True, exist_ok=True)

        runner = GameRunner(
            game_label=f"{label}/{ticker_label}",
            ticker=ticker,
            market=market,
            strategies=strategies,
            private_key=private_key,
            espn_clock=espn_clock,
            log_dir=str(sub_log_dir),
            maker_entries=maker_entries,
            maker_exits=maker_exits,
            min_entry_price=min_entry_price,
        )

        summary = runner.run()
        sub_results[ticker_label] = summary
        print_status(f"[{label}/{ticker_label}] ✓ Complete")

    except Exception as e:
        print_status(f"[{label}/{ticker_label}] ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sub_results[ticker_label] = {"error": str(e)}


def run_game(game_config: Dict[str, Any], private_key, results: Dict[str, Any]):
    label = game_config["label"]
    try:
        print_status(f"\n[{label}] Initializing...")

        # --- Discover ML ticker ---
        ml_ticker, ml_market = find_market_for_team(private_key, game_config["team_name"])
        print_status(f"[{label}] ML Market: {ml_ticker} — {ml_market.get('title', 'N/A')}")

        # --- Shared ESPN clock (one instance, cache-safe for multi-thread reads) ---
        espn_clock = setup_espn_clock(game_config)

        preferred_side = "yes" if float(game_config["model_p_win"]) >= 0.50 else "no"

        # --- Spread tickers (optional — backward compatible) ---
        spread_tickers = game_config.get("spread_tickers") or []

        # --- Build ticker list: ML + spreads ---
        ticker_configs = []  # list of (ticker_label, ticker, market, is_spread)

        # Extract short label from ML ticker (e.g. "LOU" from "KXNCAAMBGAME-...-LOU")
        ml_label = ml_ticker.rsplit("-", 1)[-1] if "-" in ml_ticker else game_config["team_name"]
        ticker_configs.append((ml_label, ml_ticker, ml_market, False))

        for sp_ticker in spread_tickers:
            try:
                sp_market = fetch_market(private_key, sp_ticker)
                sp_label = sp_ticker.rsplit("-", 1)[-1]  # e.g. "LOU2", "UNC1"
                ticker_configs.append((sp_label, sp_ticker, sp_market, True))
                print_status(f"[{label}] Spread Market: {sp_ticker} — {sp_market.get('title', 'N/A')}")
            except Exception as e:
                print_status(f"[{label}] ⚠ Failed to fetch spread ticker {sp_ticker}: {e}")

        n_tickers = len(ticker_configs)
        total_alloc = float(game_config["allocation"])

        # 50/50 ML vs spread split — ML gets half, spreads share the other half
        n_spread_tickers = sum(1 for _, _, _, is_sp in ticker_configs if is_sp)
        ml_alloc = total_alloc * 0.50
        spread_alloc_each = (total_alloc * 0.50 / max(1, n_spread_tickers)) if n_spread_tickers else 0.0

        # --- Shared ExposureTracker (thread-safe, caps total game exposure) ---
        exposure = ExposureTracker(max_exposure_dollars=total_alloc)

        # --- Game log dir (shared parent) ---
        log_dir = build_game_log_dir(
            sport=str(game_config.get("sport") or "cbb"),
            event=str(game_config.get("event") or "game"),
            game_label=label,
            game_date=str(game_config.get("espn_date") or utc_now().strftime("%Y%m%d")),
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        print_status(
            f"[{label}] {n_tickers} ticker(s) | "
            f"TotalAlloc:${total_alloc:.2f} | ML:${ml_alloc:.2f} Spread×{n_spread_tickers}:${spread_alloc_each:.2f}ea | "
            f"Preferred:{preferred_side.upper()}"
        )

        # --- Load strategy config overrides ---
        strategy_config_overrides = {}
        try:
            cfg_path = REPO_ROOT / "strategy_config.json"
            if cfg_path.exists():
                strategy_config_overrides = json.loads(cfg_path.read_text(encoding="utf-8"))
                print_status(f"[{label}] Loaded strategy config from {cfg_path}")
        except Exception as e:
            print_status(f"[{label}] ⚠ Failed to load strategy_config.json: {e}")

        # --- Build strategies + spawn sub-threads per ticker ---
        sub_results: Dict[str, Any] = {}
        sub_threads: List[threading.Thread] = []

        for ticker_label, ticker, market, is_spread in ticker_configs:
            # --- Per-ticker private key (avoids thread-safety surprises) ---
            key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
            ticker_pk = _load_private_key(key_path)

            # --- preferred_side for spread tickers ---
            # LOU* suffix → same as ML preferred_side
            # UNC* (opponent) suffix → flipped
            if is_spread:
                team_code = game_config["team_name"].upper()
                if ticker_label.upper().startswith(team_code):
                    ticker_preferred = preferred_side
                else:
                    ticker_preferred = "no" if preferred_side == "yes" else "yes"
            else:
                ticker_preferred = preferred_side

            # --- Strategy: MR-only with maker entries + min price filter ---
            ticker_alloc = spread_alloc_each if is_spread else ml_alloc
            strategies = [
                MeanReversionStrategy(
                    max_capital=ticker_alloc,
                    preferred_side=ticker_preferred,
                    exposure_tracker=exposure,
                ),
            ]
            # Apply config overrides from strategy_config.json
            for strat in strategies:
                overrides = strategy_config_overrides.get(strat.name, {})
                if overrides:
                    strat.update_params(overrides)

            ticker_type = "SPREAD" if is_spread else "ML"
            print_status(
                f"[{label}/{ticker_label}] {ticker_type} | "
                f"MR:${ticker_alloc:.2f} [maker+mp20] | "
                f"Preferred:{ticker_preferred.upper()}"
            )

            t = threading.Thread(
                target=_run_sub_ticker,
                kwargs=dict(
                    label=label,
                    ticker_label=ticker_label,
                    ticker=ticker,
                    market=market,
                    strategies=strategies,
                    private_key=ticker_pk,
                    espn_clock=espn_clock,
                    log_dir=log_dir,
                    sub_results=sub_results,
                    maker_entries=True,
                    maker_exits=True,
                    min_entry_price=20,
                ),
                name=f"{label}/{ticker_label}",
                daemon=False,
            )
            t.start()
            sub_threads.append(t)
            print_status(f"[{label}] Started sub-runner: {ticker_label}")

        # --- Wait for all sub-threads ---
        for t in sub_threads:
            t.join()

        # --- Aggregate results ---
        results[label] = _aggregate_sub_results(sub_results)
        print_status(f"[{label}] ✓ All {n_tickers} ticker(s) complete")

    except Exception as e:
        print_status(f"[{label}] ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[label] = {"error": str(e)}


# --------------------------------------------------------------------------------------
# Uploader loop
# --------------------------------------------------------------------------------------
class StopFlag:
    def __init__(self):
        self._stop = False
        self._lock = threading.Lock()

    def set(self):
        with self._lock:
            self._stop = True

    def is_set(self) -> bool:
        with self._lock:
            return self._stop


def uploader_loop(stop: StopFlag, uploader: R2Uploader, interval_secs: int = 20):
    """
    Periodically sync logs to R2 so you can watch results live.
    """
    print_status(f"[R2] Uploader thread started (interval={interval_secs}s)")
    while not stop.is_set():
        try:
            uploader.download_config(Path("strategy_config.json"))
            uploader.download_matches(Path("tennis_matches.json"))
            uploader.write_index()
            uploader.upload_changed()
        except Exception as e:
            print_status(f"[R2] Uploader error: {e}")
        time.sleep(max(5, int(interval_secs)))

    # final flush
    try:
        uploader.write_index({"final_flush": True})
        uploader.upload_changed(max_files=5000)
    except Exception:
        pass
    print_status("[R2] Uploader thread stopped")


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main() -> int:
    setup_workdir()

    print("\n" + "=" * 80)
    print("  PRODUCTION WORKER RUNNER (CLOUD) — MR-ONLY + MAKER ENTRIES + MP20")
    print("  Output: ./logs locally + optional R2 sync")
    print("=" * 80)

    api_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    if not api_key:
        print_status("✗ Set KALSHI_API_KEY_ID in environment.")
        return 1

    try:
        private_key_master = load_private_key_for_worker()
        print_status("✓ Private key loaded")
    except Exception as e:
        print_status(f"✗ Key error: {e}")
        return 1

    games = load_games()
    if not games:
        print_status("✗ No games configured (GAMES_JSON / GAMES_JSON_PATH empty).")
        return 1

    # Balance → per-game allocation (same idea as tonight_runner)
    try:
        resp = _get(private_key_master, "/trade-api/v2/portfolio/balance")
        balance_cents = int(resp.get("balance", 0))
        balance = balance_cents / 100.0
        print_status(f"✓ Balance: ${balance:.2f}")

        per_game_allocation = balance / max(1, len(games))
        for g in games:
            g["allocation"] = float(g.get("allocation") or per_game_allocation)

        print_status(f"✓ Default per-game allocation: ${per_game_allocation:.2f} (unless overridden per game)")
    except Exception as e:
        print_status(f"✗ Balance check failed: {e}")
        return 1

    print_status("GAMES:")
    for i, g in enumerate(games, 1):
        print_status(f"  {i}. {g.get('label')} | team={g.get('team_name')} | p={g.get('model_p_win')}")

    # R2 uploader (optional) — sync the whole kalshi-logs tree
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    uploader = R2Uploader(local_logs_dir=LOG_ROOT)


    stop = StopFlag()

    def _handle_sig(_sig, _frame):
        print_status("⚠ Received shutdown signal. Stopping...")
        stop.set()

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    uploader_thread = threading.Thread(
        target=uploader_loop,
        args=(stop, uploader, int(os.getenv("R2_SYNC_INTERVAL_SECS") or "20")),
        name="r2_uploader",
        daemon=True,
    )
    uploader_thread.start()

    results: Dict[str, Any] = {}
    threads: List[threading.Thread] = []

    # Start each game in its own thread (same as tonight_runner.py)
    for g in games:
        # Load separate key object per thread like the local runner does
        # (matches your pattern; avoids any thread-safety surprises)
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
        game_pk = _load_private_key(key_path)

        t = threading.Thread(
            target=run_game,
            args=(g, game_pk, results),
            name=g.get("label", f"game_{len(threads)+1}"),
            daemon=False,
        )
        t.start()
        threads.append(t)
        print_status(f"Started: {g.get('label')}")

    # Wait for games to finish, but allow graceful stop
    while not stop.is_set():
        alive = any(t.is_alive() for t in threads)
        if not alive:
            break
        time.sleep(2)

    # If stopping early, we don't kill threads forcibly (Render will SIGTERM again if needed)
    if stop.is_set():
        print_status("⚠ Stop flag set. Waiting briefly for threads to wrap up...")
        # best-effort join
        for t in threads:
            t.join(timeout=5)

    # Final report (prints + writes JSON summary file into logs/)
    print("\n" + "=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80 + "\n")

    total_net = 0.0
    for label, result in results.items():
        if "error" in result:
            print(f"{label}: ERROR — {result['error']}")
            continue

        print(f"{label}:")
        strategies = result.get("strategies", {})
        game_net = 0.0

        for strat_name, stats in strategies.items():
            net = float(stats.get("net_pnl", 0))
            game_net += net
            print(
                f"  {strat_name}: "
                f"{stats.get('trades', 0)} trades "
                f"({stats.get('wins', 0)}W-{stats.get('losses', 0)}L) | "
                f"Locks:{stats.get('locks', 0)} Stops:{stats.get('stops', 0)} | "
                f"Net:{net:.1f}¢ (${net/100:.2f}) | "
                f"Fees:{float(stats.get('fees', 0)):.1f}¢"
            )

        print(f"  GAME TOTAL: {game_net:.1f}¢ (${game_net/100:.2f})")
        total_net += game_net

    print(f"\n{'=' * 80}")
    print(f"PORTFOLIO NET: {total_net:.1f}¢ (${total_net/100:.2f})")
    print(f"{'=' * 80}\n")

    # Save one aggregate summary (useful for dashboards)
    try:
        agg = {
            "updated_utc": utc_now().isoformat(),
            "portfolio_net_cents": total_net,
            "results": results,
        }
        ts = utc_now().strftime("%Y%m%d_%H%M%S")
        out = LOG_ROOT / f"cloud_aggregate_summary_{ts}.json"

        out.write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
        print_status(f"[AGG] Wrote {out}")
    except Exception as e:
        print_status(f"[AGG] Failed to write aggregate summary: {e}")

    # Stop uploader + final flush
    stop.set()
    try:
        uploader_thread.join(timeout=20)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
