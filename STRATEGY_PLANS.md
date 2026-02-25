# Strategy Plans

## Rolling Average (MA Crossover) Strategy

### Grid Search Results (Feb 24, 2026 — 24 games: 13 CBB, 11 Tennis)

| Strategy | Sport | Best Config | $/game | Trades/game | Win Rate |
|----------|-------|-------------|--------|-------------|----------|
| EMA/TMA | Tennis | 40/160 t=12 | +$3.70 | ~1.1 | ~56% |
| TMA/TMA | CBB | 80/200 t=10 | +$2.56 | ~1.2 | ~55% |
| TMA/TMA | ALL | 100/250 t=12 | +$2.66 | ~1.0 | ~55% |

### Key characteristics
- "Sniper" strategies: ~1 trade per game, high threshold filters noise
- spread_max=6 gating (strategy self-gates, bypasses MarketQualityMonitor)
- mq_gate still used for data quality checks
- Profit comes from catching large regime shifts, not frequent mean-reversion

### Tennis deployment (LIVE — Feb 26, 2026)
- **Strategy type**: EMA/TMA (EMA fast=40, TMA slow=160)
- **Threshold**: 12 cents
- **spread_max**: 6
- **100% allocation** — no MR, RA-only for clean data collection
- **Profit defense**: activate=8c, giveback=40%, min_keep=2c

### CBB deployment (DEFERRED)
- **Strategy type**: TMA/TMA (fast=80, slow=200)
- **Threshold**: 10 cents
- Needs more backtest data before live deployment
- Will run alongside existing MR when ready

### Signal logic
1. Compute fast MA and slow MA each tick (3s)
2. If fast > slow + threshold → signal "yes" (buy YES)
3. If fast < slow - threshold → signal "no" (buy NO)
4. Only act on signal *changes* (last_signal tracking)
5. Self-gate on spread <= spread_max
6. Directional late-game filter (last 300s)
7. Exit on signal reversal, profit defense, or near-close flatten
