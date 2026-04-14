# Quant Bot - Getting Started Guide

## Quick Setup (5 minutes)

### Prerequisites
- Python 3.10+
- Ubuntu Linux or WSL2
- PostgreSQL 13+ (local or Docker)
- 16GB+ RAM recommended

### Installation

```bash
# Clone repo (or navigate to existing directory)
cd ~/Desktop/Quant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup PostgreSQL (with Docker)
docker run -d \
  --name quant_postgres \
  -e POSTGRES_DB=quant_trading \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:13

# Initialize schema
createdb quant_trading
psql quant_trading < config/schema.sql
```

## Quick Start

### 1. Run First Backtest (Mock Data)

```bash
# Activate venv
source venv/bin/activate

# Run backtest with default config
python -m src.main backtest --config config/backtest.yml

# Check results
cat reports/backtest_results.html
```

### 2. Check Event Loop

```bash
python -c "
from src.core import EventLoop
import asyncio

async def test():
    loop = EventLoop('test')
    print(f'Event loop created: {loop.name}')
    print(f'State: {loop.state.value}')

asyncio.run(test())
"
```

### 3. Test Data Connector

```bash
python -c "
from src.market_data import MockDataConnector
import asyncio
from datetime import datetime

async def test():
    connector = MockDataConnector()
    connected = await connector.connect()
    print(f'Connected: {connected}')
    print(f'Connector: {connector.name}')

asyncio.run(test())
"
```

## Project Structure Reference

```
.
├── src/                    # Main source code
│   ├── core/              # Event loop, state machine
│   ├── market_data/       # Data connectors
│   ├── analysis/          # Greeks, volume, patterns
│   ├── strategy/          # Signal generation
│   ├── execution/         # Order management
│   ├── risk/              # Risk controls
│   ├── utils/             # Helpers
│   └── main.py            # Entry point
├── tests/                 # Unit and integration tests
├── config/                # Configuration files
├── data/                  # Historical data
├── docs/                  # Documentation
├── logs/                  # Log files
├── README.md              # Project overview
├── requirements.txt       # Python dependencies
└── .github/
    └── copilot-instructions.md  # AI development guidelines
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Development Workflow

### 1. Pick a Task from ROADMAP.md
See [docs/ROADMAP.md](../docs/ROADMAP.md) for phase-by-phase breakdown.

### 2. Implement Feature
```bash
# Create feature in appropriate module
# src/core/ src/market_data/ src/analysis/ etc.

# Write tests in tests/test_*.py
# Run tests:
pytest tests/ -v
```

### 3. Commit & Track Progress
```bash
git add src/
git commit -m "Implement [FEATURE] - Phase X"
```

## Configuration

### Development Config
```yaml
# config/backtest.yml
mode: backtest
data_source: mock  # No API needed
risk_limits:
  max_portfolio_delta: 5000
```

### API Placeholders to Replace

When ready to integrate real data:

1. **Market Data:**
   - [ ] Replace `MockDataConnector` with `InteractiveBrokersConnector` or `PolygonConnector`
   - Docs: https://github.com/InteractiveBrokers/tws-api

2. **Greeks Reference:**
   - [ ] Implement full Black-Scholes in `analysis/greeks_calculator.py`
   - Reference: py_vollib, scipy.stats

3. **Execution:**
   - [ ] Implement `InteractiveBrokersExecutor` or `AlpacaExecutor`
   - Test in paper trading mode first

## Database Tips

```bash
# Connect to PostgreSQL
psql -U postgres -d quant_trading

# Check schema
\dt  # List all tables

# Check ticks table
SELECT COUNT(*) FROM ticks;

# Check portfolio snapshots
SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 5;
```

## Logging

Logs go to `logs/quant_bot.log`

```bash
# Tail logs in real-time
tail -f logs/quant_bot.log | grep -i error
```

##Troubleshooting

**"API not implemented" error?**
- You're trying to use a stub connector that needs implementation
- Use MockDataConnector for backtest testing first
- Replace with real connector when ready (see "API Placeholders" above)

**PostgreSQL connection error?**
- Check PostgreSQL is running: `psql -U postgres -c "SELECT 1;"`
- Check connection string in config
- Verify database exists: `createdb quant_trading`

**Import errors?**
- Make sure venv is activated: `source venv/bin/activate`
- Reinstall requirements: `pip install -r requirements.txt`

**Slow backtests?**
- Use fewer ticks or longer timeframes
- Move PostgreSQL to SSD
- Consider Cython for hot loops

## Next Steps

1. Read [README.md](../README.md) for architecture overview
2. Review [docs/ROADMAP.md](../docs/ROADMAP.md) for full development plan
3. Start with Phase 1: **foundation** (Weeks 1-4)
4. Pick a task from the roadmap and begin!

## Support

- **Questions?** Check docs/ folder
- **Bug reports?** Add issue in project
- **Ideas?** Add to roadmap discussion
- **Help needed?** Ask in Slack/Discord

---

Good luck! This is a substantial project - break it into phases, test thoroughly, and celebrate wins along the way. 🚀
