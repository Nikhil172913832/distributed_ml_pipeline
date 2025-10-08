#!/bin/bash
# Setup script for SECOM ML Pipeline

set -e

echo "=========================================="
echo "SECOM ML Pipeline - Setup Script"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo -e "${YELLOW}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p logs
mkdir -p data
mkdir -p models
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources

# Copy .env.example to .env if not exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}Please update .env file with your configuration${NC}"
fi

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo -e "${GREEN}✓ Python environment setup complete${NC}"

# Start Docker containers
echo ""
echo "Starting Docker containers..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 15

# Check service health
echo ""
echo "Checking service health..."

# Check Kafka
if docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092 &> /dev/null; then
    echo -e "${GREEN}✓ Kafka is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Kafka may not be ready yet${NC}"
fi

# Check PostgreSQL
if docker exec postgres pg_isready -U ml_user -d secom_pipeline &> /dev/null; then
    echo -e "${GREEN}✓ PostgreSQL is healthy${NC}"
else
    echo -e "${YELLOW}⚠ PostgreSQL may not be ready yet${NC}"
fi

# Create Kafka topics
echo ""
echo "Creating Kafka topics..."
docker exec kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --topic secom-raw-data \
    --partitions 3 \
    --replication-factor 1

docker exec kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --topic secom-preprocessed-data \
    --partitions 3 \
    --replication-factor 1

docker exec kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --topic secom-dead-letter-queue \
    --partitions 1 \
    --replication-factor 1

echo -e "${GREEN}✓ Kafka topics created${NC}"

# Display access information
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Services are running at:"
echo "  - Kafka: localhost:9092"
echo "  - PostgreSQL: localhost:5432"
echo "  - Kafka UI: http://localhost:8080"
echo "  - pgAdmin: http://localhost:8081"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000"
echo ""
echo "To run the pipeline:"
echo "  1. Train the SDV model (if not already done):"
echo "     python data_generator/secom_raw_trainer.py"
echo ""
echo "  2. Start the producer:"
echo "     python pipeline/producer.py"
echo ""
echo "  3. Start the consumer (in another terminal):"
echo "     python pipeline/consumer.py"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To stop and remove volumes:"
echo "  docker-compose down -v"
echo "=========================================="
