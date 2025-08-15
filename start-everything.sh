#!/bin/bash

# 🚀 ONE SCRIPT TO START EVERYTHING - ML Trading System
# This script does everything: builds, starts, and sets up the complete system

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}[HEADER]${NC} $1"; }

echo "🚀 ML TRADING SYSTEM - ONE SCRIPT STARTUP"
echo "=========================================="

# Check Docker
if ! docker info > /dev/null 2>&1; then
    print_error "Docker not running. Start Docker Desktop first!"
    exit 1
fi

# Create directories
print_status "Creating directories..."
mkdir -p {ml_models,data,logs,monitoring/grafana/provisioning,nginx/ssl,init-scripts}

# Make scripts executable
print_status "Making scripts executable..."
chmod +x scripts/*.sh 2>/dev/null || true

# Build images
print_status "Building Docker images..."
docker build -t ml-trading-system:latest . &
docker build -f Dockerfile.dev -t ml-trading-system:dev . &
wait

# Start development stack
print_status "Starting development stack..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for services
print_status "Waiting for services to be ready..."
sleep 30

# Check status
print_status "Service Status:"
docker-compose -f docker-compose.dev.yml ps

echo ""
echo "🎉 SYSTEM STARTED SUCCESSFULLY!"
echo ""
echo "🌐 Access Points:"
echo "  • ML Trading App: http://localhost:8501"
echo "  • Redis: localhost:6380"
echo "  • PostgreSQL: localhost:5433"
echo ""
echo "📚 Quick Commands:"
echo "  • View logs: docker-compose -f docker-compose.dev.yml logs -f ml-trading-app-dev"
echo "  • Stop: docker-compose -f docker-compose.dev.yml down"
echo "  • Restart: docker-compose -f docker-compose.dev.yml restart"
echo ""
echo "🚀 Your ML Trading System is ready!"
