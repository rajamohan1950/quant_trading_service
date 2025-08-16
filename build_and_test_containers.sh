#!/bin/bash

# Build and Test All Containers Script
# This script builds all containers and tests them for basic functionality

set -e

echo "🚀 Building and Testing All Containers..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
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

print_status "Docker is running. Proceeding with build and test..."

# Build all containers
print_status "Building containers..."

# Build Inference Container
print_status "Building inference-container..."
docker build -t inference-container:latest ./containers/inference_container/
if [ $? -eq 0 ]; then
    print_status "✅ inference-container built successfully"
else
    print_error "❌ Failed to build inference-container"
    exit 1
fi

# Build Order Execution Container
print_status "Building order-execution-container..."
docker build -t order-execution-container:latest ./containers/order_execution_container/
if [ $? -eq 0 ]; then
    print_status "✅ order-execution-container built successfully"
else
    print_error "❌ Failed to build order-execution-container"
    exit 1
fi

# Build Data Synthesizer Container
print_status "Building data-synthesizer-container..."
docker build -t data-synthesizer-container:latest ./containers/data_synthesizer_container/
if [ $? -eq 0 ]; then
    print_status "✅ data-synthesizer-container built successfully"
else
    print_error "❌ Failed to build data-synthesizer-container"
    exit 1
fi

# Build Training Pipeline Container
print_status "Building training-pipeline-container..."
docker build -t training-pipeline-container:latest ./containers/training_pipeline_container/
if [ $? -eq 0 ]; then
    print_status "✅ training-pipeline-container built successfully"
else
    print_error "❌ Failed to build training-pipeline-container"
    exit 1
fi

# Build B2C Investor Platform
print_status "Building b2c-investor-platform..."
docker build -t b2c-investor-platform:latest -f Dockerfile.b2c .
if [ $? -eq 0 ]; then
    print_status "✅ b2c-investor-platform built successfully"
else
    print_error "❌ Failed to build b2c-investor-platform"
    exit 1
fi

print_status "All containers built successfully!"

# Test containers individually
print_status "Testing containers individually..."

# Test Inference Container
print_status "Testing inference-container..."
docker run --rm --name test-inference inference-container:latest python -c "
import streamlit as st
import pandas as pd
import numpy as np
print('✅ inference-container: All imports successful')
"
if [ $? -eq 0 ]; then
    print_status "✅ inference-container test passed"
else
    print_error "❌ inference-container test failed"
fi

# Test Order Execution Container
print_status "Testing order-execution-container..."
docker run --rm --name test-order-execution order-execution-container:latest python -c "
import streamlit as st
import pandas as pd
import requests
print('✅ order-execution-container: All imports successful')
"
if [ $? -eq 0 ]; then
    print_status "✅ order-execution-container test passed"
else
    print_error "❌ order-execution-container test failed"
fi

# Test Data Synthesizer Container
print_status "Testing data-synthesizer-container..."
docker run --rm --name test-data-synthesizer data-synthesizer-container:latest python -c "
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
print('✅ data-synthesizer-container: All imports successful')
"
if [ $? -eq 0 ]; then
    print_status "✅ data-synthesizer-container test passed"
else
    print_error "❌ data-synthesizer-container test failed"
fi

# Test Training Pipeline Container
print_status "Testing training-pipeline-container..."
docker run --rm --name test-training-pipeline training-pipeline-container:latest python -c "
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
print('✅ training-pipeline-container: All imports successful')
"
if [ $? -eq 0 ]; then
    print_status "✅ training-pipeline-container test passed"
else
    print_error "❌ training-pipeline-container test failed"
fi

# Test B2C Investor Platform
print_status "Testing b2c-investor-platform..."
docker run --rm --name test-b2c-investor b2c-investor-platform:latest python -c "
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
print('✅ b2c-investor-platform: All imports successful')
"
if [ $? -eq 0 ]; then
    print_status "✅ b2c-investor-platform test passed"
else
    print_error "❌ b2c-investor-platform test failed"
fi

print_status "Individual container tests completed!"

# Test Docker Compose
print_status "Testing Docker Compose setup..."
docker-compose -f docker-compose.v2.3.yml config > /dev/null
if [ $? -eq 0 ]; then
    print_status "✅ Docker Compose configuration is valid"
else
    print_error "❌ Docker Compose configuration has errors"
    exit 1
fi

print_status "All tests completed successfully!"
echo "=================================================="
echo "🎉 All containers are ready for deployment!"
echo ""
echo "To run the complete system:"
echo "  docker-compose -f docker-compose.v2.3.yml up -d"
echo ""
echo "To access the applications:"
echo "  B2C Investor: http://localhost:8501"
echo "  Inference: http://localhost:8502"
echo "  Order Execution: http://localhost:8503"
echo "  Data Synthesizer: http://localhost:8504"
echo "  Training Pipeline: http://localhost:8505"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana: http://localhost:3000 (admin/admin)"
