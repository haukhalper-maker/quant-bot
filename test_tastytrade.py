"""
Tastytrade API Integration Setup
Run this to test your Tastytrade API connection
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.market_data import TastytradeConnector

def test_tastytrade_connection():
    """Test Tastytrade API connection"""

    # Get API credentials from environment
    api_key = os.getenv("TASTYTRADE_API_KEY")
    api_secret = os.getenv("TASTYTRADE_API_SECRET")

    if not api_key or not api_secret:
        print("❌ Tastytrade API credentials not found!")
        print("")
        print("Set your environment variables:")
        print("export TASTYTRADE_API_KEY='your_api_key'")
        print("export TASTYTRADE_API_SECRET='your_api_secret'")
        print("")
        print("Get your credentials from: https://developer.tastytrade.com/")
        return False

    print("🔗 Testing Tastytrade API connection...")

    # Create connector
    connector = TastytradeConnector(api_key, api_secret)

    try:
        # Test connection (this will fail until we implement the real API)
        import asyncio
        result = asyncio.run(connector.connect())

        if result:
            print("✅ Tastytrade API connection successful!")
            return True
        else:
            print("❌ Tastytrade API connection failed!")
            return False

    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("")
        print("Note: The Tastytrade connector is currently a stub.")
        print("To implement:")
        print("1. Install tastytrade SDK: pip install tastytrade")
        print("2. Implement authentication in TastytradeConnector.connect()")
        print("3. Implement data fetching methods")
        return False

if __name__ == "__main__":
    test_tastytrade_connection()