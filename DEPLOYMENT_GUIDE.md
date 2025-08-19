# 🚀 Deployment Guide

This guide provides detailed instructions for deploying the Professional Crypto Trading System in various environments.

## 📋 Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended for production)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 50GB+ SSD for databases and logs
- **Network**: Stable internet with low latency to exchanges
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for local development)
- Git

### Exchange Requirements
- Valid API keys for supported exchanges
- Proper API permissions enabled
- Sufficient account balance for trading

## 🏗️ Deployment Options

### Option 1: Docker Compose (Recommended)

#### Quick Production Deployment
```bash
# 1. Clone repository
git clone <repository-url>
cd crypto_trading_0817

# 2. Create production environment file
cp env.example .env.prod

# 3. Configure for production
nano .env.prod
```

**Production Environment Configuration:**
```bash
# Database URLs
DATABASE_URL=postgresql://trader:STRONG_PASSWORD@postgres:5432/crypto_trading
REDIS_URL=redis://redis:6379

# Exchange API Keys (REQUIRED for live trading)
BINANCE_API_KEY=your_production_api_key
BINANCE_SECRET_KEY=your_production_secret_key

# Trading Configuration
INITIAL_CAPITAL=50000.0
MAX_POSITION_SIZE=0.05
MAX_DRAWDOWN=0.10

# Production Settings
ENABLE_LIVE_TRADING=true
ENABLE_PAPER_TRADING=false
LOG_LEVEL=INFO

# Security
PROMETHEUS_PORT=8000
```

#### Start Production Services
```bash
# 4. Create production docker-compose file
cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - internal

  postgres:
    image: postgres:15
    restart: always
    environment:
      POSTGRES_DB: crypto_trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - internal

  trading_system:
    build: .
    restart: always
    depends_on:
      - redis
      - postgres
    env_file:
      - .env.prod
    environment:
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - DATABASE_URL=postgresql://trader:${POSTGRES_PASSWORD}@postgres:5432/crypto_trading
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/models
      - /etc/ssl/certs:/etc/ssl/certs:ro
    networks:
      - internal
      - web
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.trading.rule=Host(\`trading.yourdomain.com\`)"
      - "traefik.http.routers.trading.tls.certresolver=letsencrypt"

  prometheus:
    image: prom/prometheus:latest
    restart: always
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - internal

  grafana:
    image: grafana/grafana:latest
    restart: always
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_INSTALL_PLUGINS: grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - internal
      - web

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  internal:
    driver: bridge
  web:
    external: true
EOF

# 5. Set secure passwords
export REDIS_PASSWORD=$(openssl rand -base64 32)
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
export GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Save passwords securely
echo "REDIS_PASSWORD=${REDIS_PASSWORD}" >> .env.prod
echo "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}" >> .env.prod
echo "GRAFANA_PASSWORD=${GRAFANA_PASSWORD}" >> .env.prod

# 6. Deploy
docker-compose -f docker-compose.prod.yml up -d

# 7. Verify deployment
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs -f trading_system
```

### Option 2: Kubernetes Deployment

#### Prerequisites
```bash
# Install kubectl and helm
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

#### Kubernetes Manifests
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: trading-system

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: trading-secrets
  namespace: trading-system
type: Opaque
stringData:
  redis-password: "your-redis-password"
  postgres-password: "your-postgres-password"
  binance-api-key: "your-binance-api-key"
  binance-secret-key: "your-binance-secret-key"

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-config
  namespace: trading-system
data:
  INITIAL_CAPITAL: "50000.0"
  MAX_POSITION_SIZE: "0.05"
  ENABLE_LIVE_TRADING: "true"
  ENABLE_PAPER_TRADING: "false"
  LOG_LEVEL: "INFO"

---
# k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: trading-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--appendonly", "yes", "--requirepass", "$(REDIS_PASSWORD)"]
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: redis-password
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc

---
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: trading-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: crypto_trading
        - name: POSTGRES_USER
          value: trader
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-data
        persistentVolumeClaim:
          claimName: postgres-pvc

---
# k8s/trading-system.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-system
  namespace: trading-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trading-system
  template:
    metadata:
      labels:
        app: trading-system
    spec:
      containers:
      - name: trading-system
        image: your-registry/crypto-trading:latest
        envFrom:
        - configMapRef:
            name: trading-config
        env:
        - name: DATABASE_URL
          value: "postgresql://trader:$(POSTGRES_PASSWORD)@postgres:5432/crypto_trading"
        - name: REDIS_URL
          value: "redis://:$(REDIS_PASSWORD)@redis:6379"
        - name: BINANCE_API_KEY
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: binance-api-key
        - name: BINANCE_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: binance-secret-key
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: postgres-password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: redis-password
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: models
          mountPath: /app/models
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: logs-pvc
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

#### Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n trading-system
kubectl logs -f deployment/trading-system -n trading-system
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

#### AWS ECS Deployment

**1. Create Task Definition:**
```json
{
    "family": "crypto-trading-system",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "2048",
    "memory": "4096",
    "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
    "containerDefinitions": [
        {
            "name": "trading-system",
            "image": "your-account.dkr.ecr.region.amazonaws.com/crypto-trading:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "INITIAL_CAPITAL",
                    "value": "50000.0"
                }
            ],
            "secrets": [
                {
                    "name": "BINANCE_API_KEY",
                    "valueFrom": "arn:aws:secretsmanager:region:account:secret:trading/binance-api-key"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/crypto-trading",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

**2. Deploy with Terraform:**
```hcl
# terraform/main.tf
provider "aws" {
  region = var.aws_region
}

resource "aws_ecs_cluster" "trading_cluster" {
  name = "crypto-trading"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_service" "trading_service" {
  name            = "crypto-trading-service"
  cluster         = aws_ecs_cluster.trading_cluster.id
  task_definition = aws_ecs_task_definition.trading_task.arn
  desired_count   = 1

  launch_type = "FARGATE"

  network_configuration {
    subnets         = var.subnet_ids
    security_groups = [aws_security_group.trading_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.trading_tg.arn
    container_name   = "trading-system"
    container_port   = 8000
  }
}
```

## 🔧 Configuration Management

### Environment Variables
Create environment-specific configuration files:

#### Development (.env.dev)
```bash
# Development settings
ENABLE_LIVE_TRADING=false
ENABLE_PAPER_TRADING=true
INITIAL_CAPITAL=10000.0
LOG_LEVEL=DEBUG
MAX_DAILY_TRADES=50
```

#### Staging (.env.staging)
```bash
# Staging settings
ENABLE_LIVE_TRADING=false
ENABLE_PAPER_TRADING=true
INITIAL_CAPITAL=25000.0
LOG_LEVEL=INFO
MAX_DAILY_TRADES=100
```

#### Production (.env.prod)
```bash
# Production settings
ENABLE_LIVE_TRADING=true
ENABLE_PAPER_TRADING=false
INITIAL_CAPITAL=100000.0
LOG_LEVEL=WARNING
MAX_DAILY_TRADES=500
```

### Secret Management

#### Using Docker Secrets
```bash
# Create secrets
echo "your_api_key" | docker secret create binance_api_key -
echo "your_secret_key" | docker secret create binance_secret_key -

# Use in docker-compose
services:
  trading_system:
    secrets:
      - binance_api_key
      - binance_secret_key
    environment:
      - BINANCE_API_KEY_FILE=/run/secrets/binance_api_key
      - BINANCE_SECRET_KEY_FILE=/run/secrets/binance_secret_key

secrets:
  binance_api_key:
    external: true
  binance_secret_key:
    external: true
```

#### Using HashiCorp Vault
```bash
# Store secrets in Vault
vault kv put secret/trading \
  binance_api_key="your_api_key" \
  binance_secret_key="your_secret_key"

# Access from application
vault kv get -field=binance_api_key secret/trading
```

## 📊 Monitoring Setup

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'crypto-trading-system'
    static_configs:
      - targets: ['trading_system:8000']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### Grafana Provisioning
```yaml
# monitoring/grafana/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

### Alert Rules
```yaml
# monitoring/alert_rules.yml
groups:
  - name: trading_system_alerts
    rules:
      - alert: HighDrawdown
        expr: risk_max_drawdown_pct > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High drawdown detected"
          description: "Portfolio drawdown exceeded 10%"

      - alert: SystemDown
        expr: up{job="crypto-trading-system"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Trading system is down"
          description: "The crypto trading system is not responding"
```

## 🔐 Security Hardening

### Container Security
```dockerfile
# Use non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Use specific image versions
FROM python:3.11.6-slim

# Remove unnecessary packages
RUN apt-get remove --purge --auto-remove \
    && rm -rf /var/lib/apt/lists/*

# Set security headers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
```

### Network Security
```yaml
# Restrict network access
version: '3.8'
services:
  trading_system:
    networks:
      - internal
  
  nginx:
    ports:
      - "443:443"
    networks:
      - internal
      - web

networks:
  internal:
    driver: bridge
    internal: true
  web:
    driver: bridge
```

### SSL/TLS Configuration
```nginx
# nginx/ssl.conf
server {
    listen 443 ssl http2;
    server_name trading.yourdomain.com;

    ssl_certificate /etc/ssl/certs/trading.crt;
    ssl_certificate_key /etc/ssl/private/trading.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://trading_system:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🚨 Backup and Recovery

### Database Backups
```bash
#!/bin/bash
# backup_database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
CONTAINER_NAME="crypto_trading_postgres"

# Create backup
docker exec $CONTAINER_NAME pg_dump -U trader crypto_trading > $BACKUP_DIR/db_backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/db_backup_$DATE.sql

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +30 -delete

echo "Database backup completed: db_backup_$DATE.sql.gz"
```

### Model Backups
```bash
#!/bin/bash
# backup_models.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
MODELS_DIR="/app/models"

# Create backup
tar -czf $BACKUP_DIR/models_backup_$DATE.tar.gz -C $MODELS_DIR .

# Clean old backups
find $BACKUP_DIR -name "models_backup_*.tar.gz" -mtime +7 -delete

echo "Models backup completed: models_backup_$DATE.tar.gz"
```

### Recovery Procedures
```bash
#!/bin/bash
# restore_database.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop trading system
docker-compose stop trading_system

# Restore database
gunzip -c $BACKUP_FILE | docker exec -i crypto_trading_postgres psql -U trader -d crypto_trading

# Restart trading system
docker-compose start trading_system

echo "Database restored from $BACKUP_FILE"
```

## 📈 Performance Optimization

### Resource Allocation
```yaml
# docker-compose optimization
version: '3.8'
services:
  trading_system:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    ulimits:
      memlock:
        soft: -1
        hard: -1
```

### Database Optimization
```sql
-- postgres optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

SELECT pg_reload_conf();
```

### Redis Optimization
```conf
# redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## 🔍 Troubleshooting

### Common Deployment Issues

#### Container Startup Failures
```bash
# Check logs
docker-compose logs -f trading_system

# Check resource usage
docker stats

# Verify environment variables
docker-compose exec trading_system env | grep -E "(API_KEY|DATABASE_URL)"
```

#### Database Connection Issues
```bash
# Test database connectivity
docker-compose exec postgres psql -U trader -d crypto_trading -c "SELECT 1"

# Check database logs
docker-compose logs postgres

# Verify network connectivity
docker-compose exec trading_system ping postgres
```

#### Exchange API Issues
```bash
# Test API connectivity
docker-compose exec trading_system python -c "
from binance.client import Client
client = Client(api_key='your_key', api_secret='your_secret')
print(client.ping())
"

# Check API permissions
curl -X GET 'https://api.binance.com/api/v3/account' \
  -H 'X-MBX-APIKEY: your_api_key'
```

### Performance Issues

#### High Memory Usage
```bash
# Monitor memory usage
docker stats --no-stream

# Check for memory leaks
docker-compose exec trading_system python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

#### High CPU Usage
```bash
# Profile application
docker-compose exec trading_system python -m cProfile -o profile.stats -m src.main

# Analyze profile
python -c "
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative').print_stats(20)
"
```

## 📞 Support and Maintenance

### Health Checks
```bash
#!/bin/bash
# health_check.sh

# Check system status
curl -f http://localhost:8000/status || exit 1

# Check database
docker-compose exec postgres pg_isready -U trader || exit 1

# Check Redis
docker-compose exec redis redis-cli ping || exit 1

echo "All systems healthy"
```

### Automated Maintenance
```bash
#!/bin/bash
# maintenance.sh

# Update system packages
docker-compose pull

# Backup data
./backup_database.sh
./backup_models.sh

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Restart services
docker-compose restart

echo "Maintenance completed"
```

### Monitoring Script
```bash
#!/bin/bash
# monitor.sh

while true; do
    # Check system health
    if ! curl -f http://localhost:8000/status > /dev/null 2>&1; then
        echo "$(date): System unhealthy, restarting..."
        docker-compose restart trading_system
        sleep 60
    fi
    
    # Check resource usage
    MEMORY_USAGE=$(docker stats --no-stream --format "table {{.MemPerc}}" trading_system | tail -1 | tr -d '%')
    if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
        echo "$(date): High memory usage: $MEMORY_USAGE%"
        # Send alert or restart
    fi
    
    sleep 30
done
```

This deployment guide provides comprehensive instructions for deploying the crypto trading system in various environments with proper security, monitoring, and maintenance procedures. Choose the deployment method that best fits your infrastructure and requirements.
