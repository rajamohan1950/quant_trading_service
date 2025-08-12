#!/bin/bash

# Release v2.0.0 Script for Quant Trading Service
# This script automates the release process for version 2.0.0

set -e  # Exit on any error

echo "🚀 Starting Release v2.0.0 Process..."
echo "======================================"

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

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "feature/v2.0-release" ]; then
    print_error "Must be on feature/v2.0-release branch to release. Current branch: $CURRENT_BRANCH"
    exit 1
fi

print_status "Current branch: $CURRENT_BRANCH"

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    print_error "There are uncommitted changes. Please commit or stash them first."
    git status --short
    exit 1
fi

print_success "No uncommitted changes found"

# Check if remote branch exists and is up to date
print_status "Checking remote branch status..."
git fetch origin

LOCAL_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse origin/feature/v2.0-release)

if [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
    print_error "Local branch is not up to date with remote. Please push your changes first."
    exit 1
fi

print_success "Branch is up to date with remote"

# Verify version number
VERSION=$(cat VERSION)
if [ "$VERSION" != "2.0.0" ]; then
    print_error "Version file does not contain 2.0.0. Current version: $VERSION"
    exit 1
fi

print_success "Version verified: $VERSION"

# Run tests to ensure everything works
print_status "Running tests to ensure release quality..."
if [ -f "test_ml_setup.py" ]; then
    print_status "Running ML pipeline setup test..."
    python3 test_ml_setup.py
    if [ $? -eq 0 ]; then
        print_success "ML pipeline setup test passed"
    else
        print_error "ML pipeline setup test failed"
        exit 1
    fi
fi

# Create release tag
print_status "Creating release tag v2.0.0..."
git tag -a v2.0.0 -m "Release v2.0.0 - Enhanced ML Pipeline with LightGBM Integration

- Enhanced ML pipeline with robust fallback mechanisms
- Advanced LightGBM model adapter with scikit-learn RandomForest fallback
- Comprehensive training scripts for production-ready ML models
- Advanced feature engineering with 25+ technical indicators
- Enhanced model performance metrics (Macro-F1, PR-AUC)
- Improved error handling and safety checks throughout the pipeline
- Modular architecture with adapter pattern for easy model integration
- Comprehensive test suite for ML pipeline components
- Fixed KeyError issues and database connection conflicts
- Enhanced UI with better error handling and user experience"

print_success "Release tag v2.0.0 created"

# Push the tag to remote
print_status "Pushing release tag to remote..."
git push origin v2.0.0

print_success "Release tag pushed to remote"

# Create release summary
echo ""
echo "🎉 Release v2.0.0 Process Completed Successfully!"
echo "=================================================="
echo ""
echo "📋 Release Summary:"
echo "   Version: $VERSION"
echo "   Branch: $CURRENT_BRANCH"
echo "   Tag: v2.0.0"
echo "   Commit: $LOCAL_COMMIT"
echo ""
echo "📁 Files Added/Modified:"
echo "   - ml_service/lightgbm_adapter.py"
echo "   - ml_service/ml_pipeline.py"
echo "   - ml_service/train_lightgbm_model.py"
echo "   - ml_service/train_real_model.py"
echo "   - test_lightgbm_import.py"
echo "   - test_ml_setup.py"
echo "   - ui/pages/ml_pipeline.py"
echo "   - VERSION (updated to 2.0.0)"
echo "   - RELEASE_NOTES.md"
echo ""
echo "🚀 Next Steps:"
echo "   1. Create Pull Request on GitHub:"
echo "      https://github.com/rajamohan1950/quant_trading_service/pull/new/feature/v2.0-release"
echo "   2. Review and merge the PR"
echo "   3. Deploy to production"
echo "   4. Monitor ML pipeline performance"
echo "   5. Validate fallback mechanisms"
echo ""
echo "📚 Documentation:"
echo "   - Pull Request: PULL_REQUEST_v2.0.0.md"
echo "   - Release Notes: RELEASE_NOTES.md"
echo "   - Version: VERSION"
echo ""
echo "✅ Release process completed successfully!" 