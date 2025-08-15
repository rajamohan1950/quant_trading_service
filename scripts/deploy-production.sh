#!/bin/bash

# Production Deployment Script
# ML Trading System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root"
        exit 1
    fi
}

# Check Docker installation
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install it first."
        exit 1
    fi
    
    print_status "Docker environment check passed"
}

# Pre-deployment checks
pre_deployment_checks() {
    print_header "Running Pre-deployment Checks"
    
    check_root
    check_docker
    
    # Check available disk space
    local available_space=$(df . | awk 'NR==2 {print $4}')
    local required_space=5000000  # 5GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        print_warning "Low disk space. Available: ${available_space}KB, Required: ${required_space}KB"
        read -p "Continue anyway? (y/N): " -r response
        if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            exit 1
        fi
    fi
    
    # Check if ports are available
    local ports=(80 443 8501 3000 9090 5432 6379)
    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_warning "Port $port is already in use"
            read -p "Continue anyway? (y/N): " -r response
            if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                exit 1
            fi
        fi
    done
    
    print_status "Pre-deployment checks completed"
}

# Create production environment
create_production_env() {
    print_header "Creating Production Environment"
    
    # Create necessary directories
    print_status "Creating directories..."
    mkdir -p {ml_models,data,logs,monitoring/grafana/provisioning,nginx/ssl,init-scripts}
    
    # Create production environment file
    print_status "Creating production environment file..."
    cat > .env.production << EOF
# Production Environment Variables
ENVIRONMENT=production
LOG_LEVEL=INFO
DB_FILE=/app/data/stock_data.duckdb
MODEL_DIR=/app/ml_models
REDIS_URL=redis://redis:6379
POSTGRES_URL=postgresql://trading_user:trading_password@postgres:5432/trading_system
PROMETHEUS_URL=http://monitoring:9090
GRAFANA_URL=http://grafana:3000
EOF
    
    print_status "Production environment created"
}

# Build production images
build_production_images() {
    print_header "Building Production Images"
    
    print_status "Building main application image..."
    docker build -t ml-trading-system:latest .
    
    print_status "Production images built successfully"
}

# Deploy production stack
deploy_production_stack() {
    print_header "Deploying Production Stack"
    
    print_status "Starting production services..."
    docker-compose up -d
    
    print_status "Waiting for services to be ready..."
    sleep 60
    
    # Check service health
    print_status "Checking service health..."
    docker-compose ps
    
    # Wait for all services to be healthy
    print_status "Waiting for all services to be healthy..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        local unhealthy_services=$(docker-compose ps --format "table {{.Name}}\t{{.Status}}" | grep -c "unhealthy\|starting" || true)
        
        if [ "$unhealthy_services" -eq 0 ]; then
            print_status "All services are healthy"
            break
        fi
        
        print_status "Waiting for services to be healthy... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        print_error "Some services failed to become healthy"
        docker-compose ps
        exit 1
    fi
    
    print_status "Production stack deployed successfully"
}

# Post-deployment verification
post_deployment_verification() {
    print_header "Post-deployment Verification"
    
    # Check if services are responding
    print_status "Verifying service endpoints..."
    
    # Check main app
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        print_status "✅ ML Trading App is responding"
    else
        print_error "❌ ML Trading App is not responding"
    fi
    
    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        print_status "✅ Prometheus is responding"
    else
        print_error "❌ Prometheus is not responding"
    fi
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        print_status "✅ Grafana is responding"
    else
        print_error "❌ Grafana is not responding"
    fi
    
    # Check Nginx
    if curl -f http://localhost/health > /dev/null 2>&1; then
        print_status "✅ Nginx is responding"
    else
        print_error "❌ Nginx is not responding"
    fi
    
    print_status "Post-deployment verification completed"
}

# Show deployment summary
show_deployment_summary() {
    print_header "Deployment Summary"
    
    echo ""
    echo "🎉 Production ML Trading System Deployed Successfully!"
    echo ""
    echo "🌐 Access Points:"
    echo "  • Main Application: http://localhost:8501"
    echo "  • Nginx Proxy: http://localhost"
    echo "  • Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "  • Prometheus Metrics: http://localhost:9090"
    echo ""
    echo "📊 Service Status:"
    docker-compose ps
    echo ""
    echo "📚 Management Commands:"
    echo "  • View logs: docker-compose logs -f [service_name]"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart services: docker-compose restart"
    echo "  • Update services: docker-compose pull && docker-compose up -d"
    echo ""
    echo "🔧 Monitoring:"
    echo "  • System metrics available in Prometheus"
    echo "  • Dashboards available in Grafana"
    echo "  • Application logs: docker-compose logs -f ml-trading-app"
    echo ""
    echo "🚀 Your production trading system is ready!"
}

# Main deployment function
main() {
    print_header "Starting Production Deployment"
    
    pre_deployment_checks
    create_production_env
    build_production_images
    deploy_production_stack
    post_deployment_verification
    show_deployment_summary
}

# Run main function
main "$@"
