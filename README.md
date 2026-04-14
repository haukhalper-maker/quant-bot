# Quant Options Trading Bot - SPY/SPX

**Status:** Backtesting Ready - Tastytrade Integration Started  
**Target:** Institutional-Grade Quantitative Analysis Engine  
**Scope:** Real-time + Backtesting Framework for Options Markets

## Quick Start

```bash
# Setup environment
pip install -r requirements.txt

# Test backtest with mock data
python -m src.main backtest --data-source mock

# Test Tastytrade connection (set API keys first)
export TASTYTRADE_API_KEY="your_key"
export TASTYTRADE_API_SECRET="your_secret"
python test_tastytrade.py
```

## Overview

This repository contains a comprehensive quantitative trading bot designed to analyze and execute trades on SPY and SPX options with institutional-level rigor. The system integrates deep market microstructure analysis with sophisticated risk management.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│          STRATEGY EXECUTION & PORTFOLIO MGMT             │
├─────────────────────────────────────────────────────────┤
│  Risk Engine  │  Execution Engine  │  Order Management   │
├─────────────────────────────────────────────────────────┤
│  Greeks Analysis  │  Pattern Detection  │  Signal Gen    │
├─────────────────────────────────────────────────────────┤
│  Vol/Gamma/DOM  │  Footprint  │  Tick Analysis  │ Price  │
├─────────────────────────────────────────────────────────┤
│           MARKET DATA LAYER (Real-time + Historical)     │
├─────────────────────────────────────────────────────────┤
│  Data Ingestion  │  Normalization  │  Storage & Replay   │
└─────────────────────────────────────────────────────────┘
```

## Module Breakdown

| Module | Purpose | Status |
|--------|---------|--------|
| **core** | Event loop, state machine, core utilities | ✅ Complete |
| **market_data** | Multi-source ingestion, tick/candle generation | 🟡 Tastytrade Stub |
| **analysis** | Volume, gamma, Greeks, DOM, footprint, patterns | 📝 Skeleton |
| **strategy** | Signal generation, rule engine, alpha models | 📝 Skeleton |
| **execution** | Order execution, fills, slippage modeling | 📝 Skeleton |
| **risk** | Position limits, Greeks exposure, drawdown controls | 📝 Skeleton |

## Phase Timeline

### Phase 1 (Weeks 1-4): Foundation & Data Pipeline
- [x] Core event-driven architecture ✅
- [x] Data connector skeletons (API placeholders) ✅
- [x] Tastytrade connector stub ✅
- [x] PostgreSQL schema designed ✅
- [x] Basic backtesting engine ✅
- [ ] OHLCV candle builder
- [ ] Historical data ingestion (Tastytrade)
- [ ] Greeks calculations implementation

### Phase 2 (Weeks 5-8): Market Microstructure Analysis
- [ ] Tick/trade data processing
- [ ] Order Book (DOM) reconstruction
- [ ] Volume profile & footprint
- [ ] VWAP, TWAP, profile analysis

### Phase 3 (Weeks 9-12): Greeks & Volatility Engine
- [ ] Options pricing model (Black-Scholes/Local Vol)
- [ ] Greeks calculation (delta, gamma, vega, theta, rho)
- [ ] Implied vol surface
- [ ] Gamma scalping models

### Phase 4 (Weeks 13-16): Signal Generation & Patterns
- [ ] Volatility mean reversion detection
- [ ] Support/resistance patterns
- [ ] Gamma imbalance detection
- [ ] Flow analysis (institutional vs retail)
- [ ] Multi-timeframe confluence

### Phase 5 (Weeks 17+): Execution, Risk & Optimization
- [ ] Live order execution
- [ ] Portfolio Greeks hedging
- [ ] Risk controls & circuit breakers
- [ ] Monte Carlo drawdown analysis
- [ ] Parameter optimization

## Requirements

- Python 3.10+
- PostgreSQL 13+ (Ubuntu Linux)
- 16GB+ RAM recommended
- Real-time data feed API credentials (TBD)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup PostgreSQL
createdb quant_trading
psql quant_trading < config/schema.sql
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run in backtest mode
python -m src.backtest --config config/backtest.yml

# Run in live mode (with caution)
python -m src.live --config config/live.yml
```

## Key Design Principles

1. **Event-Driven Architecture** - Every market tick is an event; everything is stateless-friendly
2. **Separation of Concerns** - Data, Analysis, Strategy, Execution are independent
3. **Reproducibility** - Full historical replay capability for research
4. **Risk First** - Position limits and Greeks monitoring are hardcoded constraints
5. **Institutional Grade** - Multi-asset, multi-timeframe, portfolio-level thinking

## API Integration Points

- **Tastytrade API** (✅ Stub Added) - Your chosen data source
  - Real-time options tick/quote data
  - Historical tick data for backtesting
  - Options chain data
- [API: Greeks/Vol Feed] - Reference pricing (optional: calculate in-house)
- [API: Execution venue] - Live order placement (can use Tastytrade)

## Database Schema

PostgreSQL will store:
- `ticks` - High-resolution trade/quote data
- `candles` - OHLCV at various intervals
- `options_universe` - Contract specs
- `trades` - Trade history & PnL
- `risk_snapshots` - Portfolio Greeks over time

## Monitor & Logging

- Prometheus metrics (latency, error rates, PnL)
- Structured logging to ELK stack
- Real-time dashboard (Grafana)
- Backtester report generation

---

**Last Updated:** April 2026  
**Architecture Review:** Quarterly  
**Next Milestone:** Phase 1 Completion (4 weeks)
