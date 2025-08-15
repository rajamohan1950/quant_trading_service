#!/bin/bash

echo "🔧 Fixing Docker Container - Adding Missing Dependencies"
echo "========================================================"

# Stop current services
echo "Stopping current services..."
docker-compose -f docker-compose.dev.yml down

# Rebuild with new requirements
echo "Rebuilding container with new dependencies..."
docker build -f Dockerfile.dev -t ml-trading-system:dev .

# Start services again
echo "Starting services..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for services
echo "Waiting for services to be ready..."
sleep 20

# Check status
echo "Service Status:"
docker-compose -f docker-compose.dev.yml ps

echo ""
echo "✅ Container rebuilt with python-dotenv dependency!"
echo "🌐 Access your ML Trading App at: http://localhost:8501"
