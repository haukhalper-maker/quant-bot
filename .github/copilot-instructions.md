# Quant Bot Development Instructions

## Project: Institutional Options Trading Bot

**Scope:** Real-time + Backtesting quantitative analysis engine for SPY/SPX options
**Stack:** Python 3.10+, PostgreSQL, Event-Driven Architecture
**Timeline:** 4-6 months to institutional grade

## Core Philosophy

1. **Event-driven everything** - All market data is an event; systems are decoupled
2. **Data-first approach** - 80% of development is data integrity and microstructure
3. **Risk-weighted** - Every decision considers Greeks, exposure, and portfolio delta
4. **Reproducible research** - Full historical replay for any trade/analysis
5. **Institutional standards** - Multi-asset thinking, position limits, monitoring

## Code Organization

- **src/core/** - Event loop, state machine, configuration
- **src/market_data/** - Data ingestion, normalization, storage
- **src/analysis/** - All quantitative metrics (Greeks, volume, footprint, DOM)
- **src/strategy/** - Signal generation and rule engine
- **src/execution/** - Order execution and fills
- **src/risk/** - Position limits and portfolio management
- **src/utils/** - Shared utilities and helpers
- **tests/** - Unit and integration tests
- **notebooks/** - Research and analysis workbooks

## Development Standards

### Phase 1 Priority
- Create robustflow from tick data → candle generation → storage
- Build backtesting harness with realistic slippage
- Set up PostgreSQL schema for tick/trade/option data
- Skeleton all analysis modules with placeholder calculations

### Module Dependencies (Build Order)
1. core (no deps)
2. market_data (depends: core)
3. analysis (depends: core, market_data)
4. strategy (depends: core, analysis, market_data)
5. execution (depends: core, risk)
6. risk (depends: analysis)

### Testing Requirements
- All data ingestion has unit tests (mock API responses)
- All Greeks calculations tested against reference libraries
- Backtester output validated against manual calculations
- Live execution has dry-run mode before production

### Documentation
- Every module has docstrings (Google style)
- Analysis calculations include reference papers/formulas
- API placeholders clearly marked [API: Description]
- Monthly architecture review docs in docs/

## API Integration Points

The following are placeholder stubs—replace with your chosen providers:

- **Market Data Feed:** [API: Real-time options tick/quote data]
- **Historical Data:** [API: Backfill and historical tick archive]
- **Greeks Reference:** [API: Optional reference Greeks feed]
- **Execution Venue:** [API: Live order placement and fills]

## Current Status

**Phase:** Foundation (Week 1-4)
- [ ] Core event-loop and state machine
- [ ] PostgreSQL schema setup
- [ ] Mock data pipeline for testing
- [ ] OHLCV candle builder
- [ ] Basic backtest executor

## Next Steps (This Week)

1. Build core event loop with asyncio
2. Design PostgreSQL schema
3. Create mock market data connector
4. Write candle aggregation logic
5. Set up backtesting runner

---

**Last Updated:** April 2026  
**Maintained By:** Primary Developer  
**Code Review:** Before merge to main
