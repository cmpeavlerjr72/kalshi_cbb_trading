# espn_game_clock.py
import datetime as dt
import time
from typing import Any, Dict, Optional, Tuple, List
import requests

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _iso_to_dt(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _collect_events(scoreboard: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Mirrors your TS helper: sometimes ESPN uses events, sometimes leagues[].events
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

def _event_team_abbrs(ev: Dict[str, Any]) -> List[str]:
    comp = (ev.get("competitions") or [None])[0] or {}
    competitors = comp.get("competitors") or []
    abbrs = []
    for c in competitors:
        team = c.get("team") or {}
        abbr = (team.get("abbreviation") or "").upper()
        if abbr:
            abbrs.append(abbr)
    return abbrs

def _event_state(ev: Dict[str, Any]) -> str:
    st = (ev.get("status") or {}).get("type") or {}
    return (st.get("state") or "").lower()  # "pre" / "in" / "post"

class EspnGameClock:
    """
    Uses ONE-team find first; uses opponent only to disambiguate when needed.
    """
    def __init__(
        self,
        yyyymmdd: str,
        team_code: str,
        opponent_code: Optional[str] = None,
        cache_ttl_secs: int = 10,
        groups: str = "50",
        limit: int = 500,
    ):
        self.yyyymmdd = yyyymmdd
        self.team_code = (team_code or "").upper()
        self.opponent_code = (opponent_code or "").upper() if opponent_code else None
        self.cache_ttl_secs = cache_ttl_secs
        self.groups = groups
        self.limit = limit

        self._cache_at = 0.0
        self._cache_json: Optional[Dict[str, Any]] = None

    def _fetch_scoreboard(self) -> Optional[Dict[str, Any]]:
        now = time.time()
        if self._cache_json is not None and (now - self._cache_at) < self.cache_ttl_secs:
            return self._cache_json

        try:
            resp = requests.get(
                ESPN_SCOREBOARD,
                params={"dates": self.yyyymmdd, "groups": self.groups, "limit": self.limit},
                timeout=6,
                headers={"User-Agent": "Mozilla/5.0", "cache-control": "no-cache"},
            )
            if resp.status_code != 200:
                return None
            js = resp.json()
            self._cache_json = js
            self._cache_at = now
            return js
        except Exception:
            return None

    def _find_event(self, js: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        events = _collect_events(js)
        if not self.team_code:
            return None, "no_team_code"

        # 1) one-team filter
        cands = []
        for ev in events:
            abbrs = _event_team_abbrs(ev)
            if self.team_code in abbrs:
                cands.append(ev)

        if not cands:
            return None, "event_not_found"

        if len(cands) == 1:
            return cands[0], "matched_single_team_unique"

        # 2) opponent disambiguation
        if self.opponent_code:
            for ev in cands:
                abbrs = _event_team_abbrs(ev)
                if self.opponent_code in abbrs:
                    return ev, "matched_single_team_plus_opponent"

        # 3) heuristic preference: live > pre > post, then closest start time
        def score(ev: Dict[str, Any]) -> Tuple[int, float]:
            state = _event_state(ev)
            state_rank = {"in": 0, "pre": 1, "post": 2}.get(state, 3)

            comp = (ev.get("competitions") or [None])[0] or {}
            start_iso = comp.get("date") or ev.get("date")
            start_dt = _iso_to_dt(start_iso)
            if start_dt is None:
                # push unknown start times later
                return (state_rank, float("inf"))
            # closeness to now (UTC)
            now_dt = dt.datetime.now(dt.timezone.utc)
            return (state_rank, abs((start_dt - now_dt).total_seconds()))

        cands.sort(key=score)
        return cands[0], "matched_single_team_heuristic"

    def get_secs_to_game_end(self) -> Tuple[Optional[int], str]:
        js = self._fetch_scoreboard()
        if not js:
            return None, "espn_unreachable"

        ev, why = self._find_event(js)
        if not ev:
            return None, why

        status = ev.get("status") or {}
        typ = status.get("type") or {}
        state = (typ.get("state") or "").lower()
        completed = bool(typ.get("completed"))
        period = _safe_int(status.get("period"), 0)
        clock_str = status.get("displayClock") or ""

        if completed or state == "post":
            return 0, f"{why}:final"

        if state == "pre":
            # IMPORTANT: returning None here prevents “pretend time-left” pregame
            return None, f"{why}:pregame"

        # parse MM:SS
        if ":" not in clock_str:
            return None, f"{why}:clock_parse_fail"
        try:
            mm, ss = clock_str.split(":")
            clock_secs = int(mm) * 60 + int(ss)
        except Exception:
            return None, f"{why}:clock_parse_fail"

        # NCAAM: 2x20m, OT 5m (clock_secs is time remaining in current period)
        if period == 1:
            return (20 * 60) + clock_secs, f"{why}:1H"
        if period == 2:
            return clock_secs, f"{why}:2H"
        if period >= 3:
            return clock_secs, f"{why}:OT{period-2}"

        return None, f"{why}:unknown_state"
