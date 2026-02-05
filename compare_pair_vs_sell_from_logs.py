# /mnt/data/compare_pair_vs_sell_from_logs.py
"""
Compare "pair-lock" strategy (as actually executed) vs a counterfactual
"sell instead of pair" strategy using your saved per-tick CSV logs.

What this does
--------------
For each game CSV:
  1) Plots your ACTUAL TotalPnL (from total_pnl_c in the log).
  2) Replays a COUNTERFACTUAL strategy where any time the real strategy would
     have "locked a pair" (pairs_count increases), we instead:
        - SELL the currently-open leg at (best_bid - EXIT_CROSS_CENTS)
        - Realize PnL on that leg
        - Do NOT create locked pairs
     Everything else (entries / stop-loss / take-profit / other exits) is taken
     from the log transitions, and we assume the same fills as recorded for VWAP
     on the open leg (open_vwap_c) and the same open_qty.

Important caveat
----------------
We don't have true execution prices in the logs for exits/locks (only best bids),
so the counterfactual uses a simple fill model:
    sell_px = best_bid - EXIT_CROSS_CENTS (floored at 1)
You can tune EXIT_CROSS_CENTS below.

Usage
-----
Option A (hardcoded files):
    python compare_pair_vs_sell_from_logs.py

Option B (pass files):
    python compare_pair_vs_sell_from_logs.py "logs/*.csv"
    python compare_pair_vs_sell_from_logs.py "/path/to/game1.csv" "/path/to/game2.csv"

"""

import sys
import glob
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# TUNABLE ASSUMPTIONS
# -----------------------
EXIT_CROSS_CENTS = 2  # mimic combo_vnext EXIT_CROSS_CENTS default


@dataclass
class Position:
    side: Optional[str] = None  # "yes" / "no" / None
    qty: int = 0
    vwap: Optional[float] = None  # cents


def _sell_price_from_row(row: pd.Series, side: str) -> Optional[int]:
    """Counterfactual sell fill model."""
    if side == "yes":
        bid = row.get("yes_best_bid", None)
    else:
        bid = row.get("no_best_bid", None)

    if pd.isna(bid):
        return None
    bid = int(bid)
    return max(1, bid - EXIT_CROSS_CENTS)


def simulate_sell_instead_of_pair(df: pd.DataFrame) -> pd.Series:
    """
    Build a counterfactual cumulative PnL series (in cents) from the log.

    Logic:
      - Track position based on open_side/open_qty/open_vwap_c in the log.
      - When pairs_count increases (i.e., real strategy locked a pair),
        we replace it with a SELL exit of the currently-open leg.
      - When the log shows an open position disappearing WITHOUT pairs_count increasing,
        we treat that as an "actual exit" and realize PnL similarly (sell at bid-cross).
        (This keeps the counterfactual aligned to reality for stops/TP/etc.)
    """
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Ensure numeric
    for c in ["yes_best_bid", "no_best_bid", "open_qty", "open_vwap_c", "pairs_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pos = Position()
    realized = 0.0  # cents
    out = []

    prev_open_side = None
    prev_open_qty = 0
    prev_open_vwap = None
    prev_pairs = 0

    for i, row in df.iterrows():
        open_side = row.get("open_side", None)
        if isinstance(open_side, float) and pd.isna(open_side):
            open_side = None
        if isinstance(open_side, str) and open_side.strip() == "":
            open_side = None

        open_qty = row.get("open_qty", 0)
        open_qty = 0 if pd.isna(open_qty) else int(open_qty)

        open_vwap = row.get("open_vwap_c", None)
        open_vwap = None if pd.isna(open_vwap) else float(open_vwap)

        pairs = row.get("pairs_count", 0)
        pairs = 0 if pd.isna(pairs) else int(pairs)

        # Detect transitions
        opened_now = (prev_open_side is None) and (open_side in ("yes", "no")) and open_qty > 0
        closed_now = (prev_open_side in ("yes", "no")) and (open_side is None or open_qty == 0)

        pair_locked_now = pairs > prev_pairs  # real strategy created at least one new pair this tick

        # Keep our simulated position synced to the log's current open leg VWAP/QTY
        # (This means if the real strategy partially filled, we respect that.)
        if opened_now:
            pos.side = open_side
            pos.qty = open_qty
            pos.vwap = open_vwap

        # Counterfactual: when a pair lock happens, SELL instead.
        # This should coincide with the open leg being cleared in the log.
        if pair_locked_now:
            # We expect the open leg that existed just before lock to be prev_open_side/qty/vwap.
            if prev_open_side in ("yes", "no") and prev_open_qty > 0 and prev_open_vwap is not None:
                sell_px = _sell_price_from_row(row, prev_open_side)
                if sell_px is not None:
                    realized += (sell_px - float(prev_open_vwap)) * int(prev_open_qty)

            # After "selling instead", we are flat in the counterfactual.
            pos = Position()

        # If the position was closed WITHOUT a pair lock increment, treat as an exit event too.
        # (Stops, take-profit, dead-cap exits, etc.)
        if closed_now and (not pair_locked_now):
            if prev_open_side in ("yes", "no") and prev_open_qty > 0 and prev_open_vwap is not None:
                sell_px = _sell_price_from_row(row, prev_open_side)
                if sell_px is not None:
                    realized += (sell_px - float(prev_open_vwap)) * int(prev_open_qty)
            pos = Position()

        # If the log still shows an open position, keep our pos aligned to it
        # unless we just forced a counterfactual sell on lock.
        if open_side in ("yes", "no") and open_qty > 0 and (not pair_locked_now):
            pos.side = open_side
            pos.qty = open_qty
            pos.vwap = open_vwap

        # Mark-to-market (same idea as combo_vnext: use best bid - vwap on open leg)
        mtm = 0.0
        if pos.side in ("yes", "no") and pos.qty > 0 and pos.vwap is not None:
            bid = row.get("yes_best_bid") if pos.side == "yes" else row.get("no_best_bid")
            if not pd.isna(bid):
                mtm = (float(bid) - float(pos.vwap)) * int(pos.qty)

        out.append(realized + mtm)

        prev_open_side = open_side
        prev_open_qty = open_qty
        prev_open_vwap = open_vwap
        prev_pairs = pairs

    return pd.Series(out, index=df.index, name="sell_instead_total_pnl_c")


def plot_game(df: pd.DataFrame, title: str) -> None:
    df = df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Actual
    df["actual_total_pnl_c"] = pd.to_numeric(df["total_pnl_c"], errors="coerce")

    # Counterfactual
    df["sell_instead_total_pnl_c"] = simulate_sell_instead_of_pair(df)

    # -------- PnL comparison
    plt.figure()
    plt.plot(df["ts_utc"], df["actual_total_pnl_c"], label="Actual (pair-lock) TotalPnL")
    plt.plot(df["ts_utc"], df["sell_instead_total_pnl_c"], label="Counterfactual (sell instead of pair) TotalPnL")
    plt.xlabel("Time (UTC)")
    plt.ylabel("PnL (cents)")
    plt.title(f"PnL Comparison — {title}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -------- Prices (YES/NO best bids)
    plt.figure()
    plt.plot(df["ts_utc"], pd.to_numeric(df["yes_best_bid"], errors="coerce"), label="YES best bid")
    plt.plot(df["ts_utc"], pd.to_numeric(df["no_best_bid"], errors="coerce"), label="NO best bid")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Price (cents)")
    plt.title(f"Market Prices — {title}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -------- Spread between strategies over time
    plt.figure()
    spread = df["sell_instead_total_pnl_c"] - df["actual_total_pnl_c"]
    plt.plot(df["ts_utc"], spread)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Counterfactual - Actual (cents)")
    plt.title(f"Strategy Difference Over Time — {title}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print quick summary
    a_final = df["actual_total_pnl_c"].dropna().iloc[-1] if df["actual_total_pnl_c"].dropna().shape[0] else float("nan")
    s_final = df["sell_instead_total_pnl_c"].dropna().iloc[-1] if df["sell_instead_total_pnl_c"].dropna().shape[0] else float("nan")
    print(f"\n=== {title} ===")
    print(f"Actual final TotalPnL:        {a_final:.1f} c")
    print(f"Sell-instead final TotalPnL: {s_final:.1f} c")
    print(f"Delta (sell - actual):       {s_final - a_final:+.1f} c")
    print(f"Assumed EXIT_CROSS_CENTS:    {EXIT_CROSS_CENTS} c\n")


def resolve_inputs(argv: List[str]) -> List[str]:
    if len(argv) <= 1:
        # Default: look for your three files in the current folder or /mnt/data style paths
        # You can edit this list to point exactly at your files.
        defaults = [
            "Iowa at Washington_KXNCAAMBGAME-26FEB04IOWAWASH-IOWA_20260205_040959.csv",
            "Utah St. at New Mexico_KXNCAAMBGAME-26FEB04USUUNM-USU_20260205_040959.csv",
            "Washington St. at Oregon St._KXNCAAMBGAME-26FEB04WSUORST-WSU_20260205_040959.csv",
        ]
        found = []
        for p in defaults:
            found.extend(glob.glob(p))
            found.extend(glob.glob(f"/mnt/data/{p}"))
        return sorted(set(found))

    # If any arg contains wildcard, expand
    out = []
    for a in argv[1:]:
        if any(ch in a for ch in ["*", "?", "[", "]"]):
            out.extend(glob.glob(a))
        else:
            out.append(a)
    return out


def main() -> None:
    files = resolve_inputs(sys.argv)
    if not files:
        raise SystemExit("No CSV files found. Pass paths or a glob pattern.")

    for path in files:
        df = pd.read_csv(path)
        title = path.split("/")[-1].replace(".csv", "")
        plot_game(df, title=title)


if __name__ == "__main__":
    main()
