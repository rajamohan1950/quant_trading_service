#!/bin/bash

# B2C Investment Platform Docker Runner
# This script builds and runs the B2C investment platform in Docker

set -e

echo "🚀 Starting B2C Investment Platform Docker Setup..."

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install it and try again."
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

# Build the Docker image
print_status "Building Docker image..."
docker build -f Dockerfile.b2c -t b2c-investment-platform .

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Run the application
print_status "Starting B2C Investment Platform..."
docker-compose -f docker-compose.b2c.yml up -d

if [ $? -eq 0 ]; then
    print_success "B2C Investment Platform started successfully!"
    echo ""
    echo "🌐 Access the application at: http://localhost:8502"
    echo "📊 Streamlit interface will be available in a few moments"
    echo ""
    echo "📋 Useful commands:"
    echo "  - View logs: docker-compose -f docker-compose.b2c.yml logs -f"
    echo "  - Stop: docker-compose -f docker-compose.b2c.yml down"
    echo "  - Restart: docker-compose -f docker-compose.b2c.yml restart"
    echo ""
    echo "🔍 Checking container status..."
    docker-compose -f docker-compose.b2c.yml ps
else
    print_error "Failed to start B2C Investment Platform"
    exit 1
fi

# Wait a moment and check health
sleep 5
print_status "Checking application health..."
if curl -f http://localhost:8502/_stcore/health > /dev/null 2>&1; then
    print_success "Application is healthy and responding"
else
    print_warning "Application may still be starting up. Please wait a moment and check http://localhost:8502"
fi

echo ""
print_success "Setup complete! 🎉"
echo "The B2C Investment Platform is now running in Docker."
echo "You can now train LightGBM and Extreme Trees models, compare their performance,"
echo "and analyze trading results with comprehensive metrics."
