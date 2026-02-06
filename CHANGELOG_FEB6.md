# Feb 6 Fixes — Changelog

## What broke on Feb 5

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| ME death spiral (stop → re-enter → stop) | No per-side cooldown after stop-out | **P1c:** 120s cooldown per side after stop |
| ME late entries (NO 8x with 2min left) | No time-based lockout | **P1b:** No new positions with <300s to close |
| ME buying NO at 93¢ (35¢ divergence) | No divergence guard | **P1a:** Block entries when \|market - model\| > 20¢ |
| ME entering "decided" markets | No guard at extremes | **P1d:** Block when market >88¢ or <12¢ |
| MR disaster in Tarleton (dead market) | 1.5σ triggers on noise when σ=2¢ | **P2a:** Use 2.5σ when vol <5¢, **P2c:** dead market kill |
| MR entering too early in dead markets | No warmup/range check | **P2b:** Require 60 samples + range >5¢ |
| Both strategies in illiquid Tarleton | No market quality filter | **P3:** MarketQualityMonitor skips dead markets |
| Bot reported +$1.56 but Kalshi showed +$0.47 | Unrealized positions expired | **P4:** Settlement risk warnings near close |
| Fees = 331% of losses | Edge not fee-aware | **P5:** `fee_aware_min_edge()` + lock attempt cap (2) |

## File changes

### `production_strategies.py` (new/rewritten)
- `MarketQualityMonitor` class — tracks std, range, spread over warmup window
- `fee_aware_min_edge()` — computes breakeven edge accounting for round-trip fees
- `BaseStrategy` — added `can_enter(secs_to_close)` with P1b lockout, `_side_on_cooldown()` for P1c
- `ModelEdgeStrategy` — P1a divergence filter, P1d market-decided, P5 fee-aware edge
- `MeanReversionStrategy` — P2a dynamic σ mult, P2b warmup, P2c dead market detection
- `SpreadCaptureStrategy` — P5 lock attempt cap
- `GameRunner` — P3 market quality gating on entries, P4 settlement warnings, richer logging (4 CSV files + JSON summary)

### `tonight_runner.py` (updated)
- GAMES config emptied (fill in tonight's games)
- Allocations rebalanced: MR $2.50 (was $2), ME $1.50 (was $2), SC $1.00
- Prints active fixes at startup
- ESPN clock wiring unchanged

## Key behavioral changes

1. **Entries are much harder to trigger** — every strategy now has multiple gates (warmup, quality, cooldown, divergence, fees)
2. **Dead markets get zero capital** — MarketQualityMonitor kills entries if std <3¢ or range <5¢
3. **MR in low-vol needs 2.5σ** instead of 1.5σ — would have prevented most Tarleton losses
4. **ME won't chase decided markets** — the 88¢/12¢ cap + 20¢ divergence filter prevents the worst trades
5. **Exits always work** — even in killed markets, stops and locks still execute

## Before running tonight

1. Fill in GAMES list in `tonight_runner.py` with team codes + model probs
2. Verify ESPN team codes are correct (check ESPN scoreboard)
3. Consider running diagnostics first: `python kalshi_diagnostics.py`
