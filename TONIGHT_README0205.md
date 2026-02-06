# Production Multi-Strategy Testing - Feb 5, 2026

## What's New: Critical Bug Fixes

This is a production-ready version of the multi-strategy framework with **all critical bugs fixed** based on the code review.

### Fixed Bugs

1. **SpreadCaptureStrategy Lock Math (CRITICAL)**
   ```python
   # BEFORE (WRONG):
   potential_lock_cost = our_bid + (100 - no_bid)
   
   # AFTER (CORRECT):
   imp_no_ask = 100 - yes_bid  # Implied NO ask
   potential_lock_cost = our_bid + imp_no_ask
   ```
   - Was using wrong side's bid to calculate lock profit
   - Now correctly uses implied ask from opposite book

2. **Missing Liquidity Checks (CRITICAL)**
   - Added `has_fill_liquidity_for_implied_buy()` before ALL entries
   - Checks opposite book for actual fill capacity
   - Prevents "no fill" orders that waste time and fees

3. **Fee Calculation Rounding**
   ```python
   # BEFORE: int(price)  # truncates 47.8 → 47
   # AFTER:  round(price) # rounds 47.8 → 48
   ```
   - Fees are now calculated on properly rounded prices
   - Eliminates systematic underestimation

4. **ESPN Clock Integration**
   - All strategies now receive `game_secs_remaining` from real game clock
   - `ModelEdgeStrategy` uses actual game progress, not Kalshi close time
   - Fallback to Kalshi close time if ESPN unavailable

5. **Position Sizing Based on Capital**
   - Strategies now size positions based on remaining capital
   - Prevents over-allocation
   - Edge-based sizing: bigger edge = bigger position (up to limits)

6. **MeanReversion Early-Game Filter**
   - Now skips first 3+ minutes of game (too volatile)
   - Increased lookback window from 20 to 60 samples
   - Adds fee awareness to entry threshold

## File Structure

```
production_strategies.py  # All strategy code (combined from parts 1 & 2)
tonight_runner.py         # Driver for tonight's games
combo_vnext.py            # Execution helpers (existing)
espn_game_clock.py        # Live game clock (existing)
```

## Tonight's Games

1. **Tarlton**
   - Model: Tarlton 49% (Home Dog)
   - Team code: TARL

2. **Fairfield**
   - Model: Fairfield 58% (Road Favorite)
   - Team code: FAIR



## Capital Allocation

Per game: $6.00 total
- Model Edge: $2.00
- Mean Reversion: $2.00
- Spread Capture: $2.00

**Total capital at risk: $12.00** (2 games × $6)

## How to Run

### Prerequisites

1. Set environment variables:
   ```bash
   export KALSHI_API_KEY_ID="your_key_id"
   export KALSHI_PRIVATE_KEY_PATH="/path/to/private_key.pem"
   export KALSHI_ENV="PROD"  # or DEMO for testing
   ```

2. Verify balance:
   ```bash
   python -c "from combo_vnext import *; pk = _load_private_key(os.getenv('KALSHI_PRIVATE_KEY_PATH')); print(_get(pk, '/trade-api/v2/portfolio/balance'))"
   ```

### Run Tonight's Tests

```bash
python tonight_runner.py
```

The script will:
1. Show you the game configuration
2. Check your balance
3. Wait for ENTER to start
4. Run all 3 games in parallel threads
5. Print periodic status updates
6. Generate comprehensive logs

### During Execution

You'll see updates like:
```
[Iowa_at_Washington] YES:45/47 NO:52/54 | Open:2 | NetPnL:12.3¢ | Clocks:K:1234|G:1189 | Src:espn:2H
[UtahSt_at_NewMexico] YES:51/53 NO:46/48 | Open:1 | NetPnL:-5.2¢ | Clocks:K:1456|G:1402 | Src:espn:1H
```

This shows:
- Market prices (YES bid/ask, NO bid/ask)
- Open positions count
- Running net P&L
- Time remaining (Kalshi vs ESPN game clock)
- Clock source

### Stopping Early

- Press Ctrl+C to interrupt
- Games will continue running in background
- They auto-stop 5 minutes before market close

## Data Output

All data saved to `./logs/` directory.

### Per-Game Files

Each game creates 5 files:

1. **`snapshots_<game>_<timestamp>.csv`**
   - Every market tick (every 3 seconds)
   - Columns: timestamp, yes_bid, yes_ask, no_bid, no_ask, spread, depth, clocks
   - Use for: Price analysis, spread distribution, market microstructure

2. **`trades_<game>_<timestamp>.csv`**
   - Every order attempt and fill
   - Columns: strategy, action (entry_attempt/fill, exit_attempt/fill), side, price, qty, filled_qty, vwap, fee, liquidity_ok
   - Use for: Execution quality analysis, fill rates, slippage

3. **`positions_<game>_<timestamp>.csv`**
   - Complete lifecycle of each trade
   - Columns: position_id, strategy, entry/exit prices, fees, gross/net PnL, hold time, lock details
   - Use for: Strategy P&L attribution, win/loss analysis

4. **`events_<game>_<timestamp>.csv`**
   - Strategy decision points
   - Columns: strategy, event_type (entry_eval/exit_eval), decision (yes/no/hold), reason
   - Use for: Understanding why strategies did/didn't trade

5. **`summary_<game>_<timestamp>.json`**
   - Final statistics and all log paths
   - Use for: Quick results overview

## Tomorrow's Analysis

### Key Questions to Answer

1. **Which strategy performed best?**
   - Compare net P&L across strategies
   - Look at win rates and lock rates

2. **What was the lock fill rate?**
   ```bash
   # From positions CSV
   grep "lock" positions_*.csv | wc -l  # lock attempts
   # vs entries
   ```

3. **Did liquidity checks prevent bad trades?**
   ```bash
   # From trades CSV
   grep "entry_attempt" trades_*.csv | grep "False" | wc -l
   ```

4. **How accurate was the ESPN clock?**
   ```bash
   # From snapshots CSV - compare clock_source values
   grep "espn" snapshots_*.csv | wc -l  # ESPN available
   grep "kalshi_only" snapshots_*.csv | wc -l  # ESPN failed
   ```

5. **Fee impact analysis**
   ```bash
   # From positions CSV
   awk -F, 'NR>1 {fees+=$(column for total_fees); gross+=$(column for gross_pnl)} END {print "Fee ratio:", fees/gross}' positions_*.csv
   ```

### Sample Analysis Queries

```python
import pandas as pd

# Load position data
positions = pd.read_csv('logs/positions_Iowa_at_Washington_20260205_*.csv')

# Strategy comparison
strategy_stats = positions.groupby('strategy').agg({
    'net_pnl': ['count', 'sum', 'mean'],
    'hold_secs': 'mean',
    'exit_type': lambda x: (x == 'lock').sum()
})

# Lock success rate
lock_attempts = positions[positions['exit_type'] == 'lock'].shape[0]
total_entries = positions.shape[0]
lock_rate = lock_attempts / total_entries if total_entries > 0 else 0

# Fee efficiency
positions['fee_ratio'] = (positions['entry_fee'] + positions['exit_fee']) / positions['gross_pnl'].abs()
```

## What Changed from Previous Version

| Issue | Before | After |
|-------|--------|-------|
| SpreadCapture lock math | Used wrong book | Fixed to use implied ask |
| Liquidity checks | Missing | Now in OrderExecutor |
| Fee rounding | `int(price)` truncates | `round(price)` rounds |
| Position sizing | Fixed 1-3 contracts | Based on capital + edge |
| Game clock | Used Kalshi close time | ESPN real-time clock |
| MeanReversion warmup | 10 samples (30s) | 20 samples + early-game skip |
| Fee awareness | None | Entry thresholds account for fees |
| Logging | Basic CSV | 5 comprehensive files |
| Execution | Inline | Centralized OrderExecutor |

## Troubleshooting

### "No markets found"
- Check team codes match Kalshi tickers
- Markets might not be open yet
- Try `KALSHI_TICKER_OVERRIDE` env var

### "Insufficient liquidity" messages
- This is GOOD - it's preventing bad trades
- If it happens a lot, markets are thin
- Consider lowering position sizes

### ESPN clock fails
- Will fallback to Kalshi close time
- Not critical but less accurate
- Check ESPN API accessibility

### High stop-loss rate
- May need to increase `stop_loss` param
- Or widen entry edge threshold
- Check if market is trending (MeanReversion struggles)

## Post-Game Checklist

Tomorrow morning:

- [ ] Collect all logs/\*.csv and logs/\*.json files
- [ ] Check summary JSON for high-level results
- [ ] Analyze which strategy had best net P&L
- [ ] Calculate lock fill rate vs attempts
- [ ] Review trades.csv for execution quality
- [ ] Check events.csv for decision patterns
- [ ] Calculate total fees paid
- [ ] Identify any anomalies or bugs

## Safety Features

1. **Capital limits** - Per-strategy and per-game caps
2. **Entry throttling** - Min 30-60s between entries per strategy
3. **Auto-stop** - Stops 5 min before market close
4. **Liquidity checks** - Won't place orders without opposite-book depth
5. **Fee awareness** - Entry thresholds account for round-trip fees
6. **Stop losses** - Hard stops prevent runaway losses
7. **Thread safety** - Each game gets its own private key

## Expected Runtime

- Each game runs until ~5 min before close
- Typical NCAAM game: 2 hours
- With 3-second polling: ~2400 ticks per game
- Total log size: ~5-10 MB per game

Good luck tonight! 🏀
