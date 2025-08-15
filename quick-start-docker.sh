#!/bin/bash

# Quick Start Script for Docker Deployment
# ML Trading System

set -e

echo "🚀 Quick Start: ML Trading System with Docker"
echo "=============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it first."
    exit 1
fi

echo "✅ docker-compose is available"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p ml_models data logs monitoring/grafana/provisioning nginx/ssl init-scripts

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.sh

# Build images
echo "🔨 Building Docker images..."
docker build -t ml-trading-system:latest .
docker build -f Dockerfile.dev -t ml-trading-system:dev .

echo "✅ Images built successfully"

# Start development stack
echo "🚀 Starting development stack..."
docker-compose -f docker-compose.dev.yml up -d

echo "⏳ Waiting for services to be ready..."
sleep 20

# Check service status
echo "📊 Service Status:"
docker-compose -f docker-compose.dev.yml ps

echo ""
echo "🎉 Development stack started successfully!"
echo ""
echo "🌐 Access Points:"
echo "  • ML Trading App: http://localhost:8501"
echo "  • Redis: localhost:6380"
echo "  • PostgreSQL: localhost:5433"
echo ""
echo "📚 Available Commands:"
echo "  • View logs: ./scripts/docker-manager.sh logs ml-trading-app-dev development"
echo "  • Open shell: ./scripts/docker-manager.sh exec ml-trading-app-dev bash development"
echo "  • Run tests: ./scripts/docker-manager.sh test"
echo "  • Start Jupyter: ./scripts/docker-manager.sh start-profile jupyter"
echo "  • Stop services: ./scripts/docker-manager.sh stop-dev"
echo ""
echo "🚀 Your ML Trading System is ready!"
