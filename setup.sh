#!/bin/bash
# Quant Bot Setup Script
# Run with: bash setup.sh

echo "🚀 Setting up Quant Options Trading Bot"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Are you in the quant bot directory?"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Setup PostgreSQL (if available)
echo "🗄️  Setting up PostgreSQL..."
if command -v psql &> /dev/null; then
    # Create database
    createdb quant_trading 2>/dev/null || echo "Database may already exist"

    # Initialize schema
    psql quant_trading < config/schema.sql
    echo "✅ PostgreSQL setup complete"
else
    echo "⚠️  PostgreSQL not found. Install it for full functionality."
fi

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v --tb=short

# Test backtest runner
echo "📊 Testing backtest runner..."
python test_backtest.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your Tastytrade API keys:"
echo "   export TASTYTRADE_API_KEY='your_key'"
echo "   export TASTYTRADE_API_SECRET='your_secret'"
echo ""
echo "2. Run a backtest:"
echo "   python -m src.main backtest --data-source mock"
echo ""
echo "3. Run with Tastytrade data:"
echo "   python -m src.main backtest --data-source tastytrade"
echo ""
echo "Happy trading! 📈"