-- PostgreSQL Schema for Quant Bot
-- Initialize with: psql quant_trading < config/schema.sql

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Market data: High-resolution ticks (trades and quotes)
CREATE TABLE IF NOT EXISTS ticks (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    price NUMERIC(10, 2),
    size INTEGER,
    bid NUMERIC(10, 2),
    ask NUMERIC(10, 2),
    bid_size INTEGER,
    ask_size INTEGER,
    tick_type VARCHAR(20),  -- 'trade', 'bid', 'ask', 'mid'
    open_interest INTEGER,
    option_expiry DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_timestamp (symbol, timestamp)
);

-- Candle data (OHLCV at various timeframes)
CREATE TABLE IF NOT EXISTS candles (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,  -- '1m', '5m', '15m', '1h', '1d'
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC(10, 2),
    high NUMERIC(10, 2),
    low NUMERIC(10, 2),
    close NUMERIC(10, 2),
    volume BIGINT,
    tick_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp),
    INDEX idx_symbol_timeframe_timestamp (symbol, timeframe, timestamp)
);

-- Options universe: Contract specifications
CREATE TABLE IF NOT EXISTS options_contracts (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    expiry DATE NOT NULL,
    strike NUMERIC(10, 2) NOT NULL,
    option_type VARCHAR(4),  -- 'CALL', 'PUT'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, expiry, strike, option_type)
);

-- ============================================================================
-- ANALYSIS & GREEKS
-- ============================================================================

-- Greeks snapshots (aggregate Greeks per symbol/expiry at time intervals)
CREATE TABLE IF NOT EXISTS greeks_history (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    expiry DATE,
    strike NUMERIC(10, 2),
    timestamp TIMESTAMP NOT NULL,
    delta NUMERIC(8, 4),
    gamma NUMERIC(8, 6),
    vega NUMERIC(8, 4),
    theta NUMERIC(8, 4),
    rho NUMERIC(8, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_expiry_timestamp (symbol, expiry, timestamp)
);

-- Implied volatility surface history
CREATE TABLE IF NOT EXISTS iv_surface (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    expiry DATE NOT NULL,
    strike NUMERIC(10, 2) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    implied_vol NUMERIC(8, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_expiry_timestamp (symbol, expiry, timestamp)
);

-- ============================================================================
-- TRADING & EXECUTION
-- ============================================================================

-- Orders placed
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    option_type VARCHAR(20),
    strike NUMERIC(10, 2),
    expiry DATE,
    side VARCHAR(5),  -- 'BUY', 'SELL'
    quantity INTEGER,
    order_type VARCHAR(20),  -- 'MARKET', 'LIMIT', etc.
    limit_price NUMERIC(10, 2),
    created_at TIMESTAMP,
    status VARCHAR(20),  -- 'PENDING', 'FILLED', 'CANCELLED'
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price NUMERIC(10, 2),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_status (symbol, status),
    INDEX idx_order_id (order_id)
);

-- Order fills (individual fills for an order)
CREATE TABLE IF NOT EXISTS fills (
    id SERIAL PRIMARY KEY,
    fill_id VARCHAR(50) UNIQUE NOT NULL,
    order_id VARCHAR(50) NOT NULL,
    quantity INTEGER,
    price NUMERIC(10, 2),
    timestamp TIMESTAMP,
    commission NUMERIC(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- Daily trade activity
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(5),
    quantity INTEGER,
    avg_price NUMERIC(10, 2),
    pnl NUMERIC(12, 2),
    realized_pnl NUMERIC(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- RISK & PORTFOLIO
-- ============================================================================

-- Portfolio snapshots (Greeks, PnL at time intervals)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    position_count INTEGER,
    total_delta NUMERIC(12, 2),
    total_gamma NUMERIC(12, 4),
    total_vega NUMERIC(12, 2),
    total_theta NUMERIC(12, 2),
    cash NUMERIC(15, 2),
    realized_pnl NUMERIC(15, 2),
    unrealized_pnl NUMERIC(15, 2),
    gross_exposure NUMERIC(15, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp)
);

-- Daily P&L
CREATE TABLE IF NOT EXISTS daily_pnl (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    realized_pnl NUMERIC(12, 2),
    unrealized_pnl NUMERIC(12, 2),
    total_pnl NUMERIC(12, 2),
    max_drawdown NUMERIC(8, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Risk violations / alerts
CREATE TABLE IF NOT EXISTS risk_alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    alert_type VARCHAR(50),  -- 'DELTA_LIMIT', 'GAMMA_LIMIT', etc.
    severity VARCHAR(20),  -- 'WARNING', 'CRITICAL'
    message TEXT,
    data JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- STRATEGY & SIGNALS
-- ============================================================================

-- Strategy signals generated
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    strategy VARCHAR(50) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(10),
    strike NUMERIC(10, 2),
    expiry DATE,
    confidence NUMERIC(5, 4),
    position_size INTEGER,
    timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_strategy_timestamp (strategy, timestamp)
);

-- Strategy performance log
CREATE TABLE IF NOT EXISTS strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy VARCHAR(50) NOT NULL,
    date DATE,
    signals_generated INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    pnl NUMERIC(12, 2),
    win_rate NUMERIC(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- BACKTESTING
-- ============================================================================

-- Backtest runs
CREATE TABLE IF NOT EXISTS backtest_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    strategy VARCHAR(50),
    parameters JSONB,
    total_return NUMERIC(8, 4),
    sharpe_ratio NUMERIC(8, 4),
    max_drawdown NUMERIC(8, 4),
    trades_count INTEGER,
    win_rate NUMERIC(5, 4),
    pnl NUMERIC(15, 2),
    status VARCHAR(20),  -- 'RUNNING', 'COMPLETED', 'FAILED'
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_ticks_symbol_timestamp ON ticks(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe ON candles(symbol, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC);

-- ============================================================================
-- MATERIALIZED VIEWS (for reporting)
-- ============================================================================

-- Daily volume by symbol
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_volume AS
SELECT 
    symbol,
    DATE(timestamp) as date,
    SUM(size) as total_volume,
    COUNT(*) as tick_count
FROM ticks
GROUP BY symbol, DATE(timestamp);

-- Monthly Sharpe ratio
CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_metrics AS
SELECT 
    DATE_TRUNC('month', date)::DATE as month,
    SUM(pnl) as total_pnl,
    STDDEV(total_pnl) as pnl_stddev,
    AVG(total_pnl) as avg_daily_pnl
FROM daily_pnl
GROUP BY DATE_TRUNC('month', date);
