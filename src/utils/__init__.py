"""
Utilities - Helpers, Constants, Data Structures
"""

from datetime import datetime, timedelta
from typing import List
import logging


# ============================================================================
# CONSTANTS
# ============================================================================

# Markets
SYMBOLS = ["SPY", "SPX"]
STANDARD_EXPIRATIONS = ["weekly", "monthly"]

# Greeks tolerance
DELTA_TOLERANCE = 0.01
GAMMA_TOLERANCE = 0.001

# Time formats
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ISOFORMAT = "%Y-%m-%dT%H:%M:%S"

# Trading hours (Eastern Time)
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
PRE_MARKET = "04:00"
AFTER_HOURS = "20:00"


# ============================================================================
# DATE/TIME UTILITIES
# ============================================================================


def next_friday_expiry() -> str:
    """Get next Friday's date (options expiration)"""
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    return next_friday.strftime(DATE_FORMAT)


def get_expiry_dtes(expirations: List[str]) -> dict:
    """
    Get days to expiration for each contract
    expirations: list of YYYY-MM-DD strings
    """
    today = datetime.now().date()
    dtes = {}
    for exp in expirations:
        exp_date = datetime.strptime(exp, DATE_FORMAT).date()
        dte_val = (exp_date - today).days
        dtes[exp] = dte_val
    return dtes


# ============================================================================
# VALIDATION
# ============================================================================


def is_market_hours(dt: datetime) -> bool:
    """Check if datetime is during market hours"""
    hour = dt.hour
    minute = dt.minute
    weekday = dt.weekday()

    # Market closed on weekends
    if weekday >= 5:
        return False

    # Market open 9:30-16:00 ET
    market_time = hour * 100 + minute
    return 930 <= market_time < 1600


def is_valid_strike(strike: float, spot: float) -> bool:
    """Validate strike price is reasonable"""
    if strike <= 0:
        return False
    if strike < spot * 0.5 or strike > spot * 2.0:
        return False  # Strike too far ITM/OTM
    return True


# ============================================================================
# FORMATTING
# ============================================================================


def format_price(price: float, decimals: int = 2) -> str:
    """Format price with $ and decimals"""
    return f"${price:,.{decimals}f}"


def format_quantity(qty: int) -> str:
    """Format quantity with commas"""
    return f"{qty:,}"


def format_greeks(greeks: dict) -> str:
    """Format Greeks for display"""
    return (
        f"Δ={greeks['delta']:+.2f} "
        f"Γ={greeks['gamma']:+.4f} "
        f"Ν={greeks['vega']:+.2f} "
        f"Θ={greeks['theta']:+.2f}"
    )


# ============================================================================
# ERROR HANDLING
# ============================================================================


class QuantBotException(Exception):
    """Base exception for bot"""

    pass


class DataConnectException(QuantBotException):
    """Data connection failed"""

    pass


class ExecutionException(QuantBotException):
    """Order execution failed"""

    pass


class RiskException(QuantBotException):
    """Risk limit violated"""

    pass


# ============================================================================
# LOGGING SETUP (Simple backup)
# ============================================================================


def setup_simple_logging(level=logging.INFO):
    """Setup console logging"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
