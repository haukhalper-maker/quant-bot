"""
Test the backtest runner with mock data
Run with: python test_backtest.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import BotConfig, setup_logging
from src.main import BacktestRunner
from src.market_data import MockDataConnector

async def test_backtest():
    """Test the backtest runner"""
    setup_logging()

    # Create config
    config = BotConfig(
        backtesting_mode=True,
        paper_trading=True,
        symbols=["SPY"],
        max_portfolio_delta=5000.0,
        max_portfolio_gamma=1000.0,
    )

    # Create backtest runner with mock data
    data_connector = MockDataConnector()
    runner = BacktestRunner(config, data_connector)

    # Run backtest
    results = await runner.run("2023-01-01", "2023-01-31")

    print("\nBacktest Results:")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return results

if __name__ == "__main__":
    asyncio.run(test_backtest())