#!/bin/bash

# Docker Management Script for ML Trading System
# Comprehensive script to manage all Docker services

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

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop first."
        exit 1
    fi
    print_status "Docker is running"
}

# Build production images
build_production() {
    print_header "Building Production Images"
    
    print_status "Building main application image..."
    docker build -t ml-trading-system:latest .
    
    print_status "Production images built successfully"
}

# Build development images
build_development() {
    print_header "Building Development Images"
    
    print_status "Building development application image..."
    docker build -f Dockerfile.dev -t ml-trading-system:dev .
    
    print_status "Development images built successfully"
}

# Start production services
start_production() {
    print_header "Starting Production Services"
    
    check_docker
    
    print_status "Starting production stack..."
    docker-compose up -d
    
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    print_status "Checking service health..."
    docker-compose ps
    
    print_status "Production services started successfully"
    print_status "Access points:"
    print_status "  • ML Trading App: http://localhost:8501"
    print_status "  • Prometheus: http://localhost:9090"
    print_status "  • Grafana: http://localhost:3000 (admin/admin)"
    print_status "  • Nginx: http://localhost:80"
}

# Start development services
start_development() {
    print_header "Starting Development Services"
    
    check_docker
    
    print_status "Starting development stack..."
    docker-compose -f docker-compose.dev.yml up -d
    
    print_status "Waiting for services to be ready..."
    sleep 20
    
    # Check service health
    print_status "Checking service health..."
    docker-compose -f docker-compose.dev.yml ps
    
    print_status "Development services started successfully"
    print_status "Access points:"
    print_status "  • ML Trading App: http://localhost:8501"
    print_status "  • Redis: localhost:6380"
    print_status "  • PostgreSQL: localhost:5433"
}

# Start specific service profiles
start_profile() {
    local profile=$1
    
    print_header "Starting Profile: $profile"
    
    check_docker
    
    case $profile in
        "training")
            print_status "Starting ML training service..."
            docker-compose --profile training up ml-training
            ;;
        "ingestion")
            print_status "Starting data ingestion service..."
            docker-compose --profile ingestion up data-ingestion
            ;;
        "feature-eng")
            print_status "Starting feature engineering service..."
            docker-compose --profile feature-eng up feature-engineering
            ;;
        "testing")
            print_status "Starting testing environment..."
            docker-compose -f docker-compose.dev.yml --profile testing up ml-training-dev
            ;;
        "jupyter")
            print_status "Starting Jupyter Lab..."
            docker-compose -f docker-compose.dev.yml --profile jupyter up jupyter
            print_status "Jupyter Lab available at: http://localhost:8888"
            ;;
        *)
            print_error "Unknown profile: $profile"
            print_status "Available profiles: training, ingestion, feature-eng, testing, jupyter"
            exit 1
            ;;
    esac
}

# Stop services
stop_services() {
    local env=${1:-production}
    
    print_header "Stopping $env Services"
    
    if [ "$env" = "development" ]; then
        docker-compose -f docker-compose.dev.yml down
    else
        docker-compose down
    fi
    
    print_status "$env services stopped successfully"
}

# Stop all services
stop_all() {
    print_header "Stopping All Services"
    
    docker-compose down
    docker-compose -f docker-compose.dev.yml down
    
    print_status "All services stopped successfully"
}

# View logs
view_logs() {
    local service=${1:-ml-trading-app}
    local env=${2:-production}
    
    print_header "Viewing Logs for $service ($env)"
    
    if [ "$env" = "development" ]; then
        docker-compose -f docker-compose.dev.yml logs -f $service
    else
        docker-compose logs -f $service
    fi
}

# Execute commands in containers
exec_command() {
    local service=${1:-ml-trading-app}
    local command=${2:-bash}
    local env=${3:-production}
    
    print_header "Executing command in $service ($env)"
    
    if [ "$env" = "development" ]; then
        docker-compose -f docker-compose.dev.yml exec $service $command
    else
        docker-compose exec $service $command
    fi
}

# Run tests
run_tests() {
    print_header "Running Tests in Container"
    
    check_docker
    
    print_status "Running production system tests..."
    docker run --rm \
        -v $(pwd):/app \
        -w /app \
        ml-trading-system:dev \
        python -m pytest tests/test_production_system.py -v
    
    print_status "Tests completed"
}

# Clean up
cleanup() {
    print_header "Cleaning Up Docker Resources"
    
    print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Removing containers..."
        docker-compose down -v
        docker-compose -f docker-compose.dev.yml down -v
        
        print_status "Removing images..."
        docker rmi ml-trading-system:latest ml-trading-system:dev 2>/dev/null || true
        
        print_status "Removing unused volumes..."
        docker volume prune -f
        
        print_status "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Show status
show_status() {
    print_header "Service Status"
    
    print_status "Production Services:"
    docker-compose ps 2>/dev/null || print_warning "Production stack not running"
    
    echo
    print_status "Development Services:"
    docker-compose -f docker-compose.dev.yml ps 2>/dev/null || print_warning "Development stack not running"
    
    echo
    print_status "Docker Resources:"
    docker system df
}

# Show help
show_help() {
    echo "Docker Management Script for ML Trading System"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  build-prod          Build production images"
    echo "  build-dev           Build development images"
    echo "  start-prod          Start production services"
    echo "  start-dev           Start development services"
    echo "  start-profile       Start specific service profile"
    echo "  stop-prod           Stop production services"
    echo "  stop-dev            Stop development services"
    echo "  stop-all            Stop all services"
    echo "  logs [SERVICE]      View logs for service"
    echo "  exec [SERVICE]      Execute command in container"
    echo "  test                Run tests in container"
    echo "  cleanup             Clean up all Docker resources"
    echo "  status              Show service status"
    echo "  help                Show this help message"
    echo
    echo "Examples:"
    echo "  $0 start-prod                    # Start production stack"
    echo "  $0 start-dev                     # Start development stack"
    echo "  $0 start-profile training        # Start ML training service"
    echo "  $0 start-profile jupyter         # Start Jupyter Lab"
    echo "  $0 logs ml-trading-app           # View app logs"
    echo "  $0 exec ml-trading-app bash      # Open bash in app container"
    echo "  $0 test                          # Run tests"
}

# Main script logic
main() {
    case "${1:-help}" in
        "build-prod")
            build_production
            ;;
        "build-dev")
            build_development
            ;;
        "start-prod")
            start_production
            ;;
        "start-dev")
            start_development
            ;;
        "start-profile")
            start_profile "${2:-training}"
            ;;
        "stop-prod")
            stop_services "production"
            ;;
        "stop-dev")
            stop_services "development"
            ;;
        "stop-all")
            stop_all
            ;;
        "logs")
            view_logs "${2:-ml-trading-app}" "${3:-production}"
            ;;
        "exec")
            exec_command "${2:-ml-trading-app}" "${3:-bash}" "${4:-production}"
            ;;
        "test")
            run_tests
            ;;
        "cleanup")
            cleanup
            ;;
        "status")
            show_status
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
