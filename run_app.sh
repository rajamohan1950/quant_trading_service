#!/bin/bash

# Quant Trading Service Application Runner
# This script ensures the application runs with the correct Python environment

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Quant Trading Service...${NC}"

# Check if plotly is installed in system Python
if ! /Library/Frameworks/Python.framework/Versions/3.9/bin/python3.9 -c "import plotly" 2>/dev/null; then
    echo -e "${BLUE}📦 Installing plotly in system Python...${NC}"
    /Library/Frameworks/Python.framework/Versions/3.9/bin/pip install plotly
fi

# Clean up any existing database locks
if [ -f "stock_data.duckdb.lock" ]; then
    echo -e "${BLUE}🔧 Cleaning up database lock...${NC}"
    rm -f stock_data.duckdb.lock
fi

# Kill any existing streamlit processes on port 8501
if lsof -ti:8501 > /dev/null 2>&1; then
    echo -e "${BLUE}🔄 Stopping existing Streamlit process...${NC}"
    lsof -ti:8501 | xargs kill -9
    sleep 2
fi

echo -e "${BLUE}🌐 Starting Streamlit application...${NC}"
echo -e "${GREEN}📊 Application will be available at: http://localhost:8501${NC}"
echo -e "${GREEN}⚡ Latency Monitor tab: http://localhost:8501 → '⚡ Latency Monitor'${NC}"

# Run the application with system Python's streamlit
/Library/Frameworks/Python.framework/Versions/3.9/bin/streamlit run app.py --server.port 8501 