# ⚡ Tick Generator & Latency Monitor

A comprehensive end-to-end latency monitoring system for quantitative trading applications, built with Rust and Python.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Rust Tick     │───▶│  WebSocket      │───▶│   Kafka         │───▶│   Kafka         │
│   Generator     │    │   Server        │    │   Producer      │    │   Consumer      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
   Latency: T1              Latency: T2              Latency: T3              Latency: T4
```

## 🎯 Features

### 📊 Real-time Latency Monitoring
- **Tick Generator → WebSocket**: Measures latency from tick generation to WebSocket transmission
- **WebSocket → Kafka Producer**: Tracks latency from WebSocket reception to Kafka ingestion
- **Kafka Producer → Consumer**: Monitors Kafka message processing latency
- **End-to-End Latency**: Complete pipeline latency measurement

### 📈 Advanced Analytics
- Real-time latency charts with Plotly
- Statistical analysis (mean, std, min, max, percentiles)
- Performance metrics dashboard
- Export capabilities for data analysis

### 🔧 System Components

#### 1. **Rust Tick Generator** (`rust_components/tick_generator/`)
- High-performance market data generation
- Configurable tick rates (1-1000 ticks/sec)
- Realistic price movement simulation
- Multiple symbol support (NIFTY, BANKNIFTY, etc.)

#### 2. **WebSocket Server** (`rust_components/websocket_server/`)
- Asynchronous WebSocket server
- Real-time message broadcasting
- Kafka producer integration
- Connection management

#### 3. **Kafka Consumer** (`rust_components/kafka_consumer/`)
- High-throughput message consumption
- Latency statistics calculation
- Real-time performance monitoring

#### 4. **Streamlit UI** (`ui/pages/latency_monitor.py`)
- Interactive dashboard
- Real-time metrics display
- Configuration controls
- Data export functionality

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Rust toolchain
- Python 3.9+

### 1. Start the Complete System
```bash
# Make script executable (if not already)
chmod +x scripts/run_latency_monitor.sh

# Start all services
./scripts/run_latency_monitor.sh start
```

### 2. Access the UI
- **Streamlit Dashboard**: http://localhost:8501
- **WebSocket Server**: ws://localhost:8080
- **Kafka**: localhost:9092

### 3. Monitor System Status
```bash
# Check system status
./scripts/run_latency_monitor.sh status

# View logs
./scripts/run_latency_monitor.sh logs

# Stop all services
./scripts/run_latency_monitor.sh stop
```

## 📊 UI Features

### 🎛️ Configuration Panel
- **Tick Rate**: 1-1000 ticks per second
- **Symbols**: Multi-select for different instruments
- **Test Duration**: 10-300 seconds
- **WebSocket Settings**: Host and port configuration
- **Kafka Settings**: Bootstrap servers and topic

### 📈 Real-time Metrics
- **Latency Breakdown**: T1, T2, T3, and end-to-end
- **Statistical Analysis**: Mean, standard deviation, min/max
- **Percentile Charts**: 50th, 75th, 90th, 95th, 99th percentiles
- **Performance Trends**: Real-time latency graphs

### 🔧 System Status
- **Component Health**: Real-time status of all services
- **Message Count**: Total processed messages
- **Error Monitoring**: Connection and processing errors

## 🛠️ Manual Setup

### 1. Build Rust Components
```bash
# Build tick generator
cd rust_components/tick_generator
cargo build --release
cd ../..

# Build WebSocket server
cd rust_components/websocket_server
cargo build --release
cd ../..

# Build Kafka consumer
cd rust_components/kafka_consumer
cargo build --release
cd ../..
```

### 2. Start Kafka Infrastructure
```bash
docker-compose -f docker-compose.latency.yml up -d zookeeper kafka
```

### 3. Start WebSocket Server
```bash
cd rust_components/websocket_server
cargo run --release -- --port 8080 --kafka-bootstrap localhost:9092 --kafka-topic tick-data
```

### 4. Start Kafka Consumer
```bash
cd rust_components/kafka_consumer
cargo run --release -- --kafka-bootstrap localhost:9092 --kafka-topic tick-data
```

### 5. Start Tick Generator
```bash
cd rust_components/tick_generator
cargo run --release -- --websocket-url ws://localhost:8080 --tick-rate 100 --symbols NIFTY,BANKNIFTY --duration 60
```

### 6. Start Streamlit App
```bash
streamlit run app.py
```

## 📊 Performance Metrics

### Latency Targets
- **Tick Generator → WebSocket**: < 2ms
- **WebSocket → Kafka Producer**: < 5ms
- **Kafka Producer → Consumer**: < 10ms
- **End-to-End Latency**: < 15ms

### Throughput Capabilities
- **Maximum Tick Rate**: 1000 ticks/sec
- **Concurrent Connections**: 100+ WebSocket clients
- **Kafka Throughput**: 10,000+ messages/sec
- **UI Refresh Rate**: Real-time updates

## 🔧 Configuration

### Environment Variables
```bash
# WebSocket Configuration
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8080

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=tick-data

# Tick Generator Configuration
TICK_RATE=100
SYMBOLS=NIFTY,BANKNIFTY,RELIANCE,TCS
DURATION=60
```

### Docker Configuration
```yaml
# docker-compose.latency.yml
services:
  kafka:
    ports:
      - "9092:9092"
  websocket-server:
    ports:
      - "8080:8080"
  streamlit-app:
    ports:
      - "8501:8501"
```

## 📈 Monitoring & Analytics

### Real-time Dashboards
- **Latency Trends**: Time-series charts of all latency components
- **Performance Metrics**: Statistical analysis of latency distributions
- **System Health**: Component status and error rates
- **Throughput Analysis**: Message processing rates

### Data Export
- **CSV Export**: Complete latency data for external analysis
- **JSON API**: Programmatic access to metrics
- **Log Files**: Detailed system logs for debugging

## 🐛 Troubleshooting

### Common Issues

#### 1. Docker Services Not Starting
```bash
# Check Docker status
docker info

# Restart Docker services
docker-compose -f docker-compose.latency.yml down
docker-compose -f docker-compose.latency.yml up -d
```

#### 2. Rust Build Errors
```bash
# Update Rust toolchain
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

#### 3. WebSocket Connection Issues
```bash
# Check WebSocket server status
curl -I http://localhost:8080

# Test WebSocket connection
wscat -c ws://localhost:8080
```

#### 4. Kafka Connection Issues
```bash
# Check Kafka status
docker-compose -f docker-compose.latency.yml ps kafka

# View Kafka logs
docker-compose -f docker-compose.latency.yml logs kafka
```

### Performance Optimization

#### 1. High Tick Rates
- Use `--release` build for Rust components
- Increase system resources for Docker containers
- Optimize network configuration

#### 2. Low Latency Requirements
- Run tick generator on dedicated hardware
- Use localhost connections for minimal network latency
- Configure Kafka for low-latency operations

## 🔮 Future Enhancements

### Planned Features
- **Machine Learning Integration**: Predictive latency analysis
- **Alert System**: Automated alerts for latency spikes
- **Distributed Deployment**: Multi-node cluster support
- **Custom Strategies**: User-defined tick generation patterns
- **Advanced Analytics**: Correlation analysis and anomaly detection

### Performance Improvements
- **Zero-copy Networking**: Optimize data transmission
- **Memory Pooling**: Reduce allocation overhead
- **SIMD Optimization**: Vectorized data processing
- **GPU Acceleration**: Parallel processing for high-frequency data

## 📚 API Documentation

### WebSocket API
```json
{
  "timestamp": "2025-08-04T12:00:00Z",
  "symbol": "NIFTY",
  "price": 19000.50,
  "volume": 1000,
  "bid": 18999.00,
  "ask": 19001.00,
  "bid_size": 500,
  "ask_size": 500,
  "latency_t1": 1.25,
  "latency_t2": 2.50,
  "latency_t3": 5.75,
  "total_latency": 9.50
}
```

### Kafka Topics
- **tick-data**: Raw tick data with latency metrics
- **latency-stats**: Aggregated latency statistics
- **system-health**: Component health and status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for quantitative trading enthusiasts**

**For support and questions, please refer to the documentation or create an issue on GitHub.** 