# 🚀 Deployment Guide - Quant Trading Service

## 📋 Overview

This guide covers the deployment of the Quant Trading Service using various methods including Docker, local development, and cloud deployment.

## 🎯 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: Minimum 2GB RAM (4GB recommended)
- **Storage**: At least 1GB free space
- **Network**: Internet connection for API access

### Required Accounts
- **Kite Connect**: API credentials for market data
- **GitHub**: For source code and releases
- **Docker Hub** (optional): For container registry

## 🔧 Local Development Deployment

### 1. Clone Repository
```bash
git clone https://github.com/rajamohan1950/quant_trading_service.git
cd quant_trading_service
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Create .env file
cp .env.example .env

# Edit .env with your credentials
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_ACCESS_TOKEN=your_access_token
```

### 4. Run Application
```bash
streamlit run app.py
```

**Access the application at:** `http://localhost:8501`

## 🐳 Docker Deployment

### 1. Build Docker Image
```bash
# Build the image
docker build -t quant-trading-service:latest .

# Verify the image was created
docker images | grep quant-trading-service
```

### 2. Run with Docker Compose
```bash
# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### 3. Run with Docker Directly
```bash
# Run the container
docker run -d \
  --name quant-trading-service \
  -p 8501:8501 \
  -e KITE_API_KEY=your_api_key \
  -e KITE_API_SECRET=your_api_secret \
  -e KITE_ACCESS_TOKEN=your_access_token \
  -v $(pwd)/data:/app/data \
  quant-trading-service:latest

# Check container status
docker ps

# View logs
docker logs quant-trading-service
```

## ☁️ Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup
```bash
# Launch EC2 instance (t3.medium recommended)
# Ubuntu 20.04 LTS

# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose

# Add user to docker group
sudo usermod -a -G docker $USER

# Clone repository
git clone https://github.com/rajamohan1950/quant_trading_service.git
cd quant_trading_service

# Set environment variables
export KITE_API_KEY=your_api_key
export KITE_API_SECRET=your_api_secret
export KITE_ACCESS_TOKEN=your_access_token

# Run with docker-compose
docker-compose up -d
```

#### 2. Security Group Configuration
- **Inbound Rules**:
  - Port 8501 (HTTP) - For application access
  - Port 22 (SSH) - For server management

#### 3. Load Balancer (Optional)
```bash
# Create Application Load Balancer
# Target Group: Port 8501
# Health Check: /_stcore/health
```

### Azure Deployment

#### 1. Azure Container Instances
```bash
# Create resource group
az group create --name quant-trading-rg --location eastus

# Create container instance
az container create \
  --resource-group quant-trading-rg \
  --name quant-trading-service \
  --image quant-trading-service:latest \
  --ports 8501 \
  --environment-variables \
    KITE_API_KEY=your_api_key \
    KITE_API_SECRET=your_api_secret \
    KITE_ACCESS_TOKEN=your_access_token
```

#### 2. Azure App Service
```bash
# Create App Service Plan
az appservice plan create \
  --name quant-trading-plan \
  --resource-group quant-trading-rg \
  --sku B1

# Create Web App
az webapp create \
  --name quant-trading-service \
  --resource-group quant-trading-rg \
  --plan quant-trading-plan \
  --deployment-container-image-name quant-trading-service:latest
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The project includes a comprehensive CI/CD pipeline that:

1. **Tests**: Runs unit tests and linting
2. **Security**: Performs security scanning
3. **Builds**: Creates Docker images
4. **Deploys**: Automatically deploys to staging/production

### Manual Release Process

```bash
# Create a new release
./scripts/release.sh [patch|minor|major]

# Example: Create a patch release
./scripts/release.sh patch
```

### Automated Release Process

1. **Merge to main**: Triggers CI/CD pipeline
2. **Tests pass**: Security and quality checks
3. **Build succeeds**: Docker image creation
4. **Deploy to staging**: Automatic staging deployment
5. **Manual approval**: Production deployment
6. **Release created**: GitHub release with notes

## 📊 Monitoring and Health Checks

### Health Check Endpoint
```
GET http://localhost:8501/_stcore/health
```

### Application Metrics
- **Response Time**: < 1 second for data queries
- **Memory Usage**: Monitor with `docker stats`
- **CPU Usage**: Track resource utilization
- **Error Rate**: Monitor application logs

### Logging
```bash
# View application logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f quant-trading-service

# Access logs inside container
docker exec -it quant-trading-service cat /app/logs/app.log
```

## 🔒 Security Considerations

### Environment Variables
- **Never commit credentials** to version control
- **Use secrets management** in production
- **Rotate API keys** regularly
- **Monitor access logs** for suspicious activity

### Network Security
- **Use HTTPS** in production
- **Configure firewalls** appropriately
- **Limit port exposure** to necessary services
- **Use VPN** for secure access

### Data Security
- **Encrypt sensitive data** at rest
- **Backup database** regularly
- **Validate all inputs** to prevent injection
- **Monitor for anomalies** in data access

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using port 8501
lsof -i :8501

# Kill the process
kill -9 <PID>

# Or use different port
streamlit run app.py --server.port 8502
```

#### 2. Docker Build Fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t quant-trading-service:latest .
```

#### 3. Database Connection Issues
```bash
# Check database file permissions
ls -la stock_data.duckdb

# Recreate database if corrupted
rm stock_data.duckdb
streamlit run app.py
```

#### 4. API Authentication Issues
```bash
# Verify environment variables
echo $KITE_API_KEY
echo $KITE_API_SECRET
echo $KITE_ACCESS_TOKEN

# Test API connection
python -c "from kiteconnect import KiteConnect; print('API connection test')"
```

### Performance Optimization

#### 1. Memory Optimization
```bash
# Monitor memory usage
docker stats quant-trading-service

# Increase memory limit
docker run -m 4g quant-trading-service:latest
```

#### 2. Database Optimization
```sql
-- Optimize DuckDB
PRAGMA optimize;
PRAGMA vacuum;
```

#### 3. Caching Strategy
- **Session State**: Leverage Streamlit session state
- **Database Indexing**: Optimize query performance
- **API Caching**: Cache frequently accessed data

## 📈 Scaling Considerations

### Horizontal Scaling
```bash
# Run multiple instances
docker-compose up -d --scale quant-trading-service=3

# Use load balancer
# Configure nginx or HAProxy
```

### Vertical Scaling
```bash
# Increase container resources
docker run --cpus=2 --memory=4g quant-trading-service:latest
```

### Database Scaling
- **Read Replicas**: For read-heavy workloads
- **Sharding**: For large datasets
- **Caching**: Redis for frequently accessed data

## 🔄 Backup and Recovery

### Database Backup
```bash
# Create backup
cp stock_data.duckdb backup/stock_data_$(date +%Y%m%d_%H%M%S).duckdb

# Automated backup script
#!/bin/bash
BACKUP_DIR="/app/backup"
DATE=$(date +%Y%m%d_%H%M%S)
cp stock_data.duckdb "$BACKUP_DIR/stock_data_$DATE.duckdb"
find "$BACKUP_DIR" -name "*.duckdb" -mtime +7 -delete
```

### Application Backup
```bash
# Backup entire application
tar -czf backup/app_$(date +%Y%m%d_%H%M%S).tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  .
```

### Recovery Procedures
1. **Stop application**: `docker-compose down`
2. **Restore database**: Copy backup file
3. **Restart application**: `docker-compose up -d`
4. **Verify functionality**: Check health endpoint

## 📞 Support

### Getting Help
- **Documentation**: Check README.md and design documents
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub Discussions for questions
- **Releases**: Check release notes for updates

### Contact Information
- **Repository**: https://github.com/rajamohan1950/quant_trading_service
- **Issues**: https://github.com/rajamohan1950/quant_trading_service/issues
- **Releases**: https://github.com/rajamohan1950/quant_trading_service/releases

---

**Built with ❤️ for quantitative trading enthusiasts** 