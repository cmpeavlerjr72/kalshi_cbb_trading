# QUICK START GUIDE - Tonight's Multi-Strategy Test

## 🎯 What This Does

Runs **3 different trading strategies** in parallel on **3 NCAAM games** tonight:

**Strategies:**
1. **Model Edge** - Your current approach (uses model probabilities)
2. **Mean Reversion** - Pure price action (no model needed)
3. **Spread Capture** - Market making / liquidity provision

**Games (Feb 5, 2026):**
1. Iowa at Washington (Iowa 58%)
2. Utah St. at New Mexico (Utah St. 54%)
3. Washington St. at Oregon St. (Washington St. 60%)

**Capital:** $5 per game = **$15 total** ($2 model, $1.50 MR, $1.50 SC per game)

## ✅ Pre-Flight Steps (5 minutes)

### 1. Set Environment Variables

```bash
export KALSHI_API_KEY_ID="your_key_id"
export KALSHI_PRIVATE_KEY_PATH="/path/to/private_key.pem"
export KALSHI_ENV="PROD"
```

### 2. Run Pre-Flight Check

```bash
python preflight_check.py
```

This validates:
- ✓ API credentials work
- ✓ Account has sufficient balance
- ✓ All required files present
- ✓ Markets are available
- ✓ Dependencies installed

**If all checks pass**, proceed. **If any fail**, fix them first.

### 3. Start the Runner

```bash
python tonight_runner.py
```

**It will:**
- Show you the configuration
- Check your balance
- Wait for you to press ENTER
- Start all 3 games in parallel

### 4. Monitor Execution

You'll see updates like:
```
[Iowa_at_Washington] YES:45/47 NO:52/54 | Open:2 | NetPnL:12.3¢ | Clocks:K:1234|G:1189
```

Let it run! Games auto-stop 5 minutes before market close.

### 5. Tomorrow Morning - Analyze Results

```bash
python analyze_results.py
```

This shows:
- Strategy-by-strategy performance
- Lock success rates
- Execution quality
- Fee analysis
- Best/worst trades

## 📊 Key Fixes from Review

| Bug | Status |
|-----|--------|
| SpreadCapture wrong lock math | ✅ Fixed |
| Missing liquidity checks | ✅ Fixed |
| Fee rounding errors | ✅ Fixed |
| No ESPN clock | ✅ Fixed |
| Position sizing issues | ✅ Fixed |
| Early-game volatility | ✅ Fixed |

## 📁 Output Files

After running, check `logs/` directory:

**Per Game (×3):**
- `snapshots_*.csv` - Market ticks every 3 seconds
- `trades_*.csv` - Every order attempt/fill
- `positions_*.csv` - Complete trade lifecycle
- `events_*.csv` - Strategy decisions
- `summary_*.json` - Final statistics

## 🎓 What to Learn Tomorrow

**Primary Goals:**
1. Which strategy performed best overall?
2. What was the lock fill rate vs expectations?
3. Did liquidity checks prevent bad trades?
4. How accurate was the ESPN clock?
5. What's the fee impact as % of gross P&L?

**Secondary Analysis:**
- Win rates by strategy
- Average hold times
- Stop-loss vs take-profit exits
- Execution slippage
- Time-of-game performance (early vs late)

## ⚠️ Important Notes

**Capital Management:**
- Each strategy has independent capital limit
- Positions auto-sized based on edge + capital
- Hard stop-losses prevent runaway losses

**Throttling:**
- 30-60s minimum between entries per strategy
- Max 2 open positions per strategy
- Auto-stops 5 min before market close

**Liquidity:**
- Orders only placed if opposite book has depth
- This prevents "no fill" wasted time
- You'll see "SKIP ENTRY: insufficient liquidity" - this is GOOD

**Clock Source:**
- Uses ESPN live game clock when available
- Falls back to Kalshi close time if ESPN fails
- Check `clock_source` in snapshots to see which was used

## 🆘 Troubleshooting

**"No markets found"**
→ Markets not open yet, or team codes wrong
→ Try closer to game time

**"Insufficient balance"**
→ Reduce ALLOCATIONS in tonight_runner.py
→ Or run fewer games

**ESPN clock fails**
→ Non-critical, will use Kalshi close time
→ Just less accurate for late-game blending

**Stop losses firing frequently**
→ Markets might be trending (MR struggles)
→ Or volatility higher than expected
→ Review in tomorrow's analysis

## 🎯 Success Metrics

**Good Signs:**
- Lock rate >40%
- Fill rate >80%
- Fee ratio <30% of gross P&L
- Net P&L positive (but small sample)

**Warning Signs:**
- Stop rate >50%
- Fill rate <50%
- Liquidity blocks >30% of attempts
- One strategy much worse than others

## 📝 Files Included

```
production_strategies.py  ← All strategy code (fixed bugs)
tonight_runner.py         ← Main driver (run this)
preflight_check.py        ← Validation (run first)
analyze_results.py        ← Tomorrow's analysis
TONIGHT_README.md         ← Detailed documentation
```

## ⏱️ Timeline

**Tonight:**
1. 6:45 PM - Run preflight check
2. 6:50 PM - Start tonight_runner.py
3. 7:00 PM - First game tips
4. ~9:30 PM - All games complete
5. Go to bed

**Tomorrow:**
1. Run analyze_results.py
2. Review strategy performance
3. Check execution quality
4. Identify improvements
5. Plan next iteration

---

## Ready?

```bash
# Step 1
python preflight_check.py

# Step 2 (if all checks pass)
python tonight_runner.py

# Tomorrow
python analyze_results.py
```

Good luck! 🏀
