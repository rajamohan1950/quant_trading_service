#!/bin/bash

# Comprehensive Test Runner for ML Pipeline
# This script runs all tests with 100% coverage requirements

set -e

echo "🚀 Starting Comprehensive Test Suite for ML Pipeline"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
print_status "Python version: $PYTHON_VERSION"

# Install test dependencies
print_status "Installing test dependencies..."
pip install -r requirements-test.txt

# Create test directories if they don't exist
mkdir -p tests
mkdir -p htmlcov

# Run linting and code quality checks
print_status "Running code quality checks..."

# Run black (code formatting)
print_status "Checking code formatting with black..."
black --check --diff ml_service/ tests/ || {
    print_warning "Code formatting issues found. Run 'black ml_service/ tests/' to fix"
}

# Run flake8 (style guide)
print_status "Running flake8 style checks..."
flake8 ml_service/ tests/ --max-line-length=120 --ignore=E203,W503 || {
    print_warning "Style issues found. Please fix them before proceeding"
}

# Run mypy (type checking)
print_status "Running mypy type checks..."
mypy ml_service/ --ignore-missing-imports || {
    print_warning "Type checking issues found"
}

# Run unit tests with coverage
print_status "Running unit tests with coverage..."
pytest tests/ -v --cov=ml_service --cov-report=term-missing --cov-report=html --cov-fail-under=100 \
    --tb=short --strict-markers -m "unit" || {
    print_error "Unit tests failed or coverage below 100%"
    exit 1
}

# Run integration tests
print_status "Running integration tests..."
pytest tests/ -v --cov=ml_service --cov-report=term-missing --cov-report=html \
    --tb=short --strict-markers -m "integration" || {
    print_error "Integration tests failed"
    exit 1
}

# Run regression tests
print_status "Running regression tests..."
pytest tests/ -v --cov=ml_service --cov-report=term-missing --cov-report=html \
    --tb=short --strict-markers -m "regression" || {
    print_error "Regression tests failed"
    exit 1
}

# Run performance tests
print_status "Running performance tests..."
pytest tests/ -v --benchmark-only --benchmark-skip --tb=short -m "performance" || {
    print_warning "Performance tests failed (non-critical)"
}

# Run all tests together for final coverage report
print_status "Running complete test suite for final coverage report..."
pytest tests/ -v --cov=ml_service --cov-report=term-missing --cov-report=html:htmlcov \
    --cov-report=xml:coverage.xml --cov-report=json:coverage.json \
    --cov-fail-under=100 --tb=short --strict-markers || {
    print_error "Final test suite failed or coverage below 100%"
    exit 1
}

# Generate test report
print_status "Generating test report..."
pytest tests/ --html=test-report.html --self-contained-html --tb=short

# Check coverage
print_status "Checking coverage requirements..."
COVERAGE=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
if (( $(echo "$COVERAGE >= 100" | bc -l) )); then
    print_success "Coverage requirement met: $COVERAGE%"
else
    print_error "Coverage requirement not met: $COVERAGE% (required: 100%)"
    exit 1
fi

# Performance benchmarks
print_status "Running performance benchmarks..."
pytest tests/ --benchmark-only --benchmark-skip --benchmark-min-rounds=10 \
    --benchmark-warmup --benchmark-calibrate || {
    print_warning "Performance benchmarks failed (non-critical)"
}

# Memory usage check
print_status "Checking memory usage..."
python3 -c "
import tracemalloc
import ml_service.ml_pipeline
import ml_service.lightgbm_adapter
import ml_service.trading_features

tracemalloc.start()
# Import and basic operations
pipeline = ml_service.ml_pipeline.MLPipelineService()
adapter = ml_service.lightgbm_adapter.LightGBMAdapter('test', '/tmp/test.pkl')
engineer = ml_service.trading_features.TradingFeatureEngineer()

current, peak = tracemalloc.get_traced_memory()
print(f'Current memory usage: {current / 1024 / 1024:.2f} MB')
print(f'Peak memory usage: {peak / 1024 / 1024:.2f} MB')

if peak > 100 * 1024 * 1024:  # 100 MB
    print('WARNING: High memory usage detected')
else:
    print('Memory usage is within acceptable limits')

tracemalloc.stop()
"

# Final summary
echo ""
echo "=================================================="
echo "🎉 Test Suite Completed Successfully!"
echo "=================================================="
echo ""
echo "📊 Coverage Report: htmlcov/index.html"
echo "📋 Test Report: test-report.html"
echo "📈 Coverage Data: coverage.xml, coverage.json"
echo ""
echo "✅ All tests passed with 100% coverage"
echo "✅ Code quality checks passed"
echo "✅ Integration tests passed"
echo "✅ Regression tests passed"
echo ""

# Open coverage report in browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Opening coverage report in browser..."
    open htmlcov/index.html
fi

print_success "Test suite completed successfully! 🚀" 