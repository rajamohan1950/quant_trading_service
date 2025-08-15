# 🐳 Docker Setup for ML Trading System

## 🎯 Overview

This document provides comprehensive instructions for running the ML Trading System using Docker containers. This approach ensures consistent deployment across different environments and avoids compatibility issues with M1 Macs and other architectures.

## 🚀 Quick Start

### Prerequisites

1. **Docker Desktop** installed and running
2. **docker-compose** available
3. **Git** for cloning the repository

### One-Command Setup

```bash
# Make scripts executable and start development stack
chmod +x scripts/*.sh quick-start-docker.sh
./quick-start-docker.sh
```

## 🏗️ Architecture

### Service Components

- **ML Trading App**: Main Streamlit application (Port 8501)
- **Redis**: Caching and real-time data (Port 6379)
- **PostgreSQL**: Structured data storage (Port 5432)
- **Prometheus**: Metrics collection (Port 9090)
- **Grafana**: Visualization dashboards (Port 3000)
- **Nginx**: Reverse proxy and load balancing (Port 80/443)

### Network Configuration

- **Production Network**: `172.20.0.0/16`
- **Development Network**: `172.21.0.0/16`
- **Service Discovery**: Automatic via Docker Compose

## 📋 Available Commands

### Docker Management Script

```bash
# View all available commands
./scripts/docker-manager.sh help

# Build images
./scripts/docker-manager.sh build-prod    # Production
./scripts/docker-manager.sh build-dev     # Development

# Start services
./scripts/docker-manager.sh start-prod    # Production stack
./scripts/docker-manager.sh start-dev     # Development stack

# Start specific profiles
./scripts/docker-manager.sh start-profile training    # ML training service
./scripts/docker-manager.sh start-profile jupyter     # Jupyter Lab
./scripts/docker-manager.sh start-profile testing     # Test environment

# Stop services
./scripts/docker-manager.sh stop-prod     # Stop production
./scripts/docker-manager.sh stop-dev      # Stop development
./scripts/docker-manager.sh stop-all      # Stop everything

# View logs
./scripts/docker-manager.sh logs ml-trading-app
./scripts/docker-manager.sh logs ml-trading-app development

# Execute commands in containers
./scripts/docker-manager.sh exec ml-trading-app bash
./scripts/docker-manager.sh exec ml-trading-app python -c "print('Hello')"

# Run tests
./scripts/docker-manager.sh test

# Clean up
./scripts/docker-manager.sh cleanup

# Show status
./scripts/docker-manager.sh status
```

### Production Deployment

```bash
# Deploy production stack
./scripts/deploy-production.sh
```

## 🔧 Development Workflow

### 1. Start Development Environment

```bash
# Start development stack with hot reload
./scripts/docker-manager.sh start-dev
```

### 2. Access Development Services

- **ML Trading App**: http://localhost:8501
- **Redis**: localhost:6380
- **PostgreSQL**: localhost:5433

### 3. Development Features

- **Hot Reload**: Code changes automatically reload
- **Debug Port**: Available on port 5678
- **Volume Mounting**: Local code changes reflected immediately
- **Jupyter Lab**: Available on port 8888

### 4. Running Tests

```bash
# Run tests in container
./scripts/docker-manager.sh test

# Or run specific test files
docker-compose -f docker-compose.dev.yml exec ml-trading-app-dev \
    python -m pytest tests/test_production_system.py -v
```

## 🚀 Production Deployment

### 1. Deploy Production Stack

```bash
# Full production deployment
./scripts/deploy-production.sh
```

### 2. Production Access Points

- **Main App**: http://localhost:8501
- **Nginx Proxy**: http://localhost
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### 3. Production Features

- **Load Balancing**: Via Nginx
- **Health Checks**: Automatic service monitoring
- **Metrics Collection**: Prometheus integration
- **Visualization**: Grafana dashboards
- **SSL Ready**: HTTPS configuration available

## 📊 Monitoring & Observability

### Prometheus Metrics

- **Application Metrics**: Custom trading system metrics
- **System Metrics**: Container and host performance
- **Business Metrics**: Trading performance indicators

### Grafana Dashboards

- **System Overview**: Container health and performance
- **Trading Metrics**: Model performance and predictions
- **Custom Dashboards**: Configurable for specific needs

### Health Checks

```bash
# Check service health
docker-compose ps

# View health check logs
docker-compose logs ml-trading-app | grep health
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Port Conflicts

```bash
# Check what's using a port
lsof -i :8501

# Stop conflicting services
sudo lsof -ti:8501 | xargs kill -9
```

#### 2. Container Won't Start

```bash
# Check container logs
docker-compose logs ml-trading-app

# Check container status
docker-compose ps

# Restart specific service
docker-compose restart ml-trading-app
```

#### 3. Permission Issues

```bash
# Fix script permissions
chmod +x scripts/*.sh

# Fix volume permissions
sudo chown -R $USER:$USER ml_models data logs
```

#### 4. Memory Issues

```bash
# Check Docker resource usage
docker system df

# Clean up unused resources
docker system prune -a
```

### Debug Commands

```bash
# Enter container for debugging
docker-compose exec ml-trading-app bash

# View real-time logs
docker-compose logs -f ml-trading-app

# Check container resources
docker stats

# Inspect container configuration
docker inspect ml-trading-app
```

## 📁 File Structure

```
├── Dockerfile                 # Production Docker image
├── Dockerfile.dev            # Development Docker image
├── docker-compose.yml        # Production services
├── docker-compose.dev.yml    # Development services
├── scripts/
│   ├── docker-manager.sh     # Docker management script
│   └── deploy-production.sh  # Production deployment
├── monitoring/
│   └── prometheus.yml        # Prometheus configuration
├── nginx/
│   └── nginx.conf           # Nginx configuration
├── quick-start-docker.sh     # Quick start script
└── DOCKER_README.md         # This file
```

## 🔐 Security Considerations

### Production Security

- **Network Isolation**: Services isolated in Docker networks
- **Rate Limiting**: API rate limiting via Nginx
- **Security Headers**: Security headers configured in Nginx
- **SSL Ready**: HTTPS configuration available
- **Health Checks**: Automatic service monitoring

### Development Security

- **Local Only**: Services only accessible locally
- **Debug Ports**: Available for development debugging
- **Volume Mounting**: Secure local development

## 📈 Scaling & Performance

### Horizontal Scaling

```bash
# Scale specific services
docker-compose up -d --scale ml-trading-app=3

# Load balancing via Nginx
# (Configured automatically)
```

### Performance Optimization

- **Multi-stage Builds**: Optimized Docker images
- **Layer Caching**: Efficient image builds
- **Resource Limits**: Configurable container resources
- **Monitoring**: Performance metrics collection

## 🚀 Next Steps

### 1. Customize Configuration

- Modify `docker-compose.yml` for your environment
- Adjust resource limits in Docker Compose
- Configure custom monitoring dashboards

### 2. Add SSL Certificates

```bash
# Place certificates in nginx/ssl/
# Uncomment HTTPS configuration in nginx/nginx.conf
```

### 3. Set Up CI/CD

- Integrate with GitHub Actions
- Automated testing in containers
- Automated deployment pipelines

### 4. Production Monitoring

- Set up alerting in Prometheus
- Configure Grafana dashboards
- Set up log aggregation

## 📞 Support

### Getting Help

1. **Check Logs**: Use `docker-compose logs` commands
2. **Verify Configuration**: Check Docker Compose files
3. **Test Connectivity**: Use health check endpoints
4. **Review Documentation**: Check this README and code comments

### Useful Commands

```bash
# Quick status check
./scripts/docker-manager.sh status

# View all logs
docker-compose logs

# Restart everything
docker-compose down && docker-compose up -d

# Clean rebuild
docker-compose down -v && docker-compose up -d --build
```

---

**🎉 Your ML Trading System is now containerized and ready for production!**

For more information, check the main project README and individual service documentation.
