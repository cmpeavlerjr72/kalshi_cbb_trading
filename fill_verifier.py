# fill_verifier.py — Real-time fill verification against Kalshi API
#
# Registers order_ids after trades, verifies them in batches every ~30s,
# classifies each as confirmed/mismatch/missing/partial, persists to
# verified_fills.json (auto-picked up by R2 uploader).

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from combo_vnext import fetch_fills_for_order, fills_vwap_cents

MAX_CHECKS = 5
VERIFY_INTERVAL_SECS = 30.0

STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"
STATUS_MISMATCH = "mismatch"
STATUS_MISSING = "missing"
STATUS_PARTIAL = "partial"


class FillVerifier:
    """Verifies bot-logged trades against Kalshi's fill API."""

    def __init__(self, private_key, log_dir: str, ticker: str):
        self.private_key = private_key
        self.ticker = ticker
        self.log_dir = Path(log_dir)
        self.json_path = self.log_dir / "verified_fills.json"
        self._orders: Dict[str, Dict[str, Any]] = {}  # order_id -> record
        self._last_verify_time = 0.0
        self._load()

    def _load(self):
        """Load previously verified orders from disk."""
        try:
            if self.json_path.exists():
                data = json.loads(self.json_path.read_text(encoding="utf-8"))
                for rec in data.get("orders", []):
                    oid = rec.get("order_id")
                    if oid:
                        self._orders[oid] = rec
        except Exception:
            pass

    def _save(self):
        """Persist current state to verified_fills.json."""
        try:
            payload = {
                "ticker": self.ticker,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "orders": list(self._orders.values()),
                "summary": self._build_summary(),
            }
            self.json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    def _build_summary(self) -> Dict[str, int]:
        counts = {"confirmed": 0, "mismatch": 0, "missing": 0, "pending": 0, "partial": 0}
        for rec in self._orders.values():
            s = rec.get("status", STATUS_PENDING)
            if s in counts:
                counts[s] += 1
        counts["total"] = len(self._orders)
        return counts

    def register_order(
        self,
        order_id: Optional[str],
        side: str,
        action: str,
        qty: int,
        price: float,
        fee: float,
        strategy: str = "",
    ):
        """Register an order for verification. Call after _log_trade."""
        if not order_id:
            return
        if order_id in self._orders and self._orders[order_id]["status"] != STATUS_PENDING:
            return  # already verified

        self._orders[order_id] = {
            "order_id": order_id,
            "status": STATUS_PENDING,
            "ticker": self.ticker,
            "side": side,
            "action": action,
            "strategy": strategy,
            "bot_qty": qty,
            "bot_price": price,
            "bot_fee": fee,
            "kalshi_qty": None,
            "kalshi_vwap": None,
            "kalshi_fee": None,
            "kalshi_fills": None,
            "check_count": 0,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "checked_at": None,
            "mismatch_details": None,
        }
        self._save()

    def tick(self) -> List[Dict[str, Any]]:
        """Called every loop iteration. Returns list of newly-verified orders with issues."""
        now = time.time()
        if now - self._last_verify_time < VERIFY_INTERVAL_SECS:
            return []

        self._last_verify_time = now
        pending = [
            oid for oid, rec in self._orders.items()
            if rec["status"] in (STATUS_PENDING, STATUS_PARTIAL)
        ]

        if not pending:
            return []

        results = []
        for oid in pending:
            result = self._verify_one(oid)
            if result:
                results.append(result)

        if results:
            self._save()

        return results

    def _verify_one(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Verify a single order against Kalshi. Returns record if status changed."""
        rec = self._orders[order_id]
        rec["check_count"] += 1
        rec["checked_at"] = datetime.now(timezone.utc).isoformat()

        try:
            fills = fetch_fills_for_order(self.private_key, order_id)
        except Exception as e:
            rec["mismatch_details"] = f"API error: {e}"
            if rec["check_count"] >= MAX_CHECKS:
                rec["status"] = STATUS_MISSING
                return rec
            rec["status"] = STATUS_PARTIAL
            return None

        kalshi_qty = sum(int(f.get("count", 0)) for f in fills) if fills else 0

        # Compute VWAP
        kalshi_vwap = None
        if fills and kalshi_qty > 0:
            kalshi_vwap = fills_vwap_cents(fills, rec["side"])

        # Compute total fee from fills
        kalshi_fee = None
        if fills:
            try:
                kalshi_fee = sum(
                    abs(float(f.get("fee", 0))) for f in fills
                ) * 100  # convert dollars to cents
            except (TypeError, ValueError):
                pass

        rec["kalshi_qty"] = kalshi_qty
        rec["kalshi_vwap"] = kalshi_vwap
        rec["kalshi_fee"] = kalshi_fee
        rec["kalshi_fills"] = fills if fills else []

        bot_qty = rec["bot_qty"]

        # Classify
        if bot_qty == 0 and kalshi_qty == 0:
            # Nofill confirmed as nofill
            rec["status"] = STATUS_CONFIRMED
            rec["mismatch_details"] = None
            return rec

        if bot_qty == 0 and kalshi_qty > 0:
            # Bot thinks nofill but Kalshi has fills — the Brunclik scenario
            rec["status"] = STATUS_MISMATCH
            rec["mismatch_details"] = f"bot_qty=0 but kalshi_qty={kalshi_qty}"
            return rec

        if bot_qty > 0 and kalshi_qty == 0:
            # Bot thinks fill but Kalshi has nothing
            if rec["check_count"] >= MAX_CHECKS:
                rec["status"] = STATUS_MISSING
                rec["mismatch_details"] = f"bot_qty={bot_qty} but no Kalshi fills after {MAX_CHECKS} checks"
                return rec
            rec["status"] = STATUS_PARTIAL
            return None

        # Both have fills — compare qty
        if kalshi_qty != bot_qty:
            rec["status"] = STATUS_MISMATCH
            rec["mismatch_details"] = f"qty: bot={bot_qty} kalshi={kalshi_qty}"
            return rec

        # Qty matches — compare price (allow 0.5c tolerance for VWAP rounding)
        bot_price = rec["bot_price"]
        if kalshi_vwap is not None and bot_price > 0:
            price_diff = abs(kalshi_vwap - bot_price)
            if price_diff > 0.5:
                rec["status"] = STATUS_MISMATCH
                rec["mismatch_details"] = f"price: bot={bot_price:.1f}c kalshi={kalshi_vwap:.1f}c (diff={price_diff:.1f}c)"
                return rec

        # Everything matches
        rec["status"] = STATUS_CONFIRMED
        rec["mismatch_details"] = None
        return rec

    def flush(self) -> Dict[str, Any]:
        """Force-verify all remaining pending orders. Called at end-of-run."""
        pending = [
            oid for oid, rec in self._orders.items()
            if rec["status"] in (STATUS_PENDING, STATUS_PARTIAL)
        ]

        for oid in pending:
            # Give each pending order up to MAX_CHECKS attempts
            rec = self._orders[oid]
            while rec["status"] in (STATUS_PENDING, STATUS_PARTIAL) and rec["check_count"] < MAX_CHECKS:
                self._verify_one(oid)

            # If still pending after max checks, mark missing
            if rec["status"] in (STATUS_PENDING, STATUS_PARTIAL):
                rec["status"] = STATUS_MISSING
                rec["mismatch_details"] = f"still pending after {rec['check_count']} checks"

        self._save()
        return self._build_summary()

    def get_summary(self) -> Dict[str, int]:
        return self._build_summary()

    def get_mismatches(self) -> List[Dict[str, Any]]:
        """Return all orders with mismatch or missing status."""
        return [
            rec for rec in self._orders.values()
            if rec["status"] in (STATUS_MISMATCH, STATUS_MISSING)
        ]
