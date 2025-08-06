#!/bin/bash

# Latency Monitor System Runner
# This script starts the complete tick generator and latency monitoring system

set -e

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
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install it and try again."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Build Rust components
build_rust_components() {
    print_status "Building Rust components..."
    
    # Build tick generator
    cd rust_components/tick_generator
    cargo build --release
    print_success "Tick generator built successfully"
    cd ../..
    
    # Build WebSocket server
    cd rust_components/websocket_server
    cargo build --release
    print_success "WebSocket server built successfully"
    cd ../..
    
    # Build Kafka consumer
    cd rust_components/kafka_consumer
    cargo build --release
    print_success "Kafka consumer built successfully"
    cd ../..
}

# Start the system
start_system() {
    print_status "Starting latency monitoring system..."
    
    # Start Kafka and Zookeeper
    docker-compose -f docker-compose.latency.yml up -d zookeeper kafka
    
    print_status "Waiting for Kafka to be ready..."
    sleep 30
    
    # Start WebSocket server
    docker-compose -f docker-compose.latency.yml up -d websocket-server
    
    print_status "Waiting for WebSocket server to be ready..."
    sleep 10
    
    # Start Kafka consumer
    docker-compose -f docker-compose.latency.yml up -d kafka-consumer
    
    print_status "Waiting for Kafka consumer to be ready..."
    sleep 10
    
    # Start Streamlit app
    docker-compose -f docker-compose.latency.yml up -d streamlit-app
    
    print_success "All services started successfully!"
}

# Start tick generator
start_tick_generator() {
    print_status "Starting tick generator..."
    
    # Run tick generator locally (not in Docker for better performance)
    cd rust_components/tick_generator
    cargo run --release -- \
        --websocket-url ws://localhost:8080 \
        --tick-rate 100 \
        --symbols NIFTY,BANKNIFTY,RELIANCE,TCS \
        --duration 60 &
    
    TICK_GENERATOR_PID=$!
    print_success "Tick generator started with PID: $TICK_GENERATOR_PID"
    cd ../..
}

# Show system status
show_status() {
    print_status "System Status:"
    echo "  📊 Kafka: $(docker-compose -f docker-compose.latency.yml ps kafka | grep -c 'Up' || echo 'Down')"
    echo "  🔌 WebSocket Server: $(docker-compose -f docker-compose.latency.yml ps websocket-server | grep -c 'Up' || echo 'Down')"
    echo "  📈 Kafka Consumer: $(docker-compose -f docker-compose.latency.yml ps kafka-consumer | grep -c 'Up' || echo 'Down')"
    echo "  🌐 Streamlit App: $(docker-compose -f docker-compose.latency.yml ps streamlit-app | grep -c 'Up' || echo 'Down')"
    echo "  ⚡ Tick Generator: $(ps -p $TICK_GENERATOR_PID > /dev/null && echo 'Running' || echo 'Stopped')"
}

# Stop the system
stop_system() {
    print_status "Stopping latency monitoring system..."
    
    # Stop tick generator
    if [ ! -z "$TICK_GENERATOR_PID" ]; then
        kill $TICK_GENERATOR_PID 2>/dev/null || true
        print_status "Tick generator stopped"
    fi
    
    # Stop Docker services
    docker-compose -f docker-compose.latency.yml down
    
    print_success "All services stopped"
}

# Show logs
show_logs() {
    print_status "Showing logs for all services..."
    docker-compose -f docker-compose.latency.yml logs -f
}

# Main script
main() {
    case "${1:-start}" in
        "start")
            check_docker
            check_docker_compose
            build_rust_components
            start_system
            start_tick_generator
            show_status
            print_success "🎉 Latency monitoring system is ready!"
            print_status "📊 Access the UI at: http://localhost:8501"
            print_status "🔌 WebSocket server at: ws://localhost:8080"
            print_status "📈 Kafka at: localhost:9092"
            ;;
        "stop")
            stop_system
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "restart")
            stop_system
            sleep 5
            main start
            ;;
        *)
            echo "Usage: $0 {start|stop|status|logs|restart}"
            echo "  start   - Start the complete system"
            echo "  stop    - Stop all services"
            echo "  status  - Show system status"
            echo "  logs    - Show logs for all services"
            echo "  restart - Restart the complete system"
            exit 1
            ;;
    esac
}

# Trap to stop services on script exit
trap stop_system EXIT

main "$@" 