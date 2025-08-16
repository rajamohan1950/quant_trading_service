#!/bin/bash

# Simple B2C Investment Platform Docker Runner
# This script runs just the main platform without optional services

set -e

echo "🚀 Starting B2C Investment Platform (Simple Mode)..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install it and try again."
    exit 1
fi

print_success "Docker Compose is available"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p ml_models data logs

# Copy requirements file if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    print_status "Copying B2C requirements file..."
    cp requirements.b2c.txt requirements.txt
fi

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose -f docker-compose.b2c.simple.yml down 2>/dev/null || true

# Run the application
print_status "Starting B2C Investment Platform..."
docker-compose -f docker-compose.b2c.simple.yml up -d

if [ $? -eq 0 ]; then
    print_success "B2C Investment Platform started successfully!"
    echo ""
    echo "🌐 Access the application at: http://localhost:8502"
    echo "📊 Streamlit interface will be available in a few moments"
    echo ""
    echo "📋 Useful commands:"
    echo "  - View logs: docker-compose -f docker-compose.b2c.simple.yml logs -f"
    echo "  - Stop: docker-compose -f docker-compose.b2c.simple.yml down"
    echo "  - Restart: docker-compose -f docker-compose.b2c.simple.yml restart"
    echo ""
    echo "🔍 Checking container status..."
    docker-compose -f docker-compose.b2c.simple.yml ps
else
    echo "❌ Failed to start B2C Investment Platform"
    exit 1
fi

# Wait a moment and check health
sleep 5
print_status "Checking application health..."
if curl -f http://localhost:8502/_stcore/health > /dev/null 2>&1; then
    print_success "Application is healthy and responding"
else
    print_status "Application may still be starting up. Please wait a moment and check http://localhost:8502"
fi

echo ""
print_success "Setup complete! 🎉"
echo "The B2C Investment Platform is now running in Docker."
echo "You can now train LightGBM and Extreme Trees models, compare their performance,"
echo "and analyze trading results with comprehensive metrics."
echo ""
echo "🎯 Open your browser and go to: http://localhost:8502"
