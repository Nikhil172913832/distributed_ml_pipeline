# Makefile for SECOM ML Pipeline

.PHONY: help setup start stop restart clean test lint format health logs

help:
	@echo "SECOM ML Pipeline - Available Commands:"
	@echo ""
	@echo "  make setup      - Initial setup (install deps, start services)"
	@echo "  make start      - Start all Docker services"
	@echo "  make stop       - Stop all Docker services"
	@echo "  make restart    - Restart all services"
	@echo "  make clean      - Stop services and remove volumes"
	@echo "  make producer   - Run the data producer"
	@echo "  make consumer   - Run the data consumer"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code with black"
	@echo "  make health     - Check service health"
	@echo "  make logs       - View all logs"
	@echo ""

setup:
	@echo "Setting up SECOM ML Pipeline..."
	chmod +x setup.sh
	./setup.sh

start:
	@echo "Starting Docker services..."
	docker-compose up -d

stop:
	@echo "Stopping Docker services..."
	docker-compose down

restart: stop start
	@echo "Services restarted"

clean:
	@echo "Cleaning up (removing volumes)..."
	docker-compose down -v
	rm -rf logs/*

producer:
	@echo "Starting producer..."
	source .venv/bin/activate && python pipeline/producer.py

consumer:
	@echo "Starting consumer..."
	source .venv/bin/activate && python pipeline/consumer.py

test:
	@echo "Running tests..."
	source .venv/bin/activate && pytest tests/ -v --cov=pipeline

lint:
	@echo "Running linter..."
	source .venv/bin/activate && flake8 pipeline/ tests/

format:
	@echo "Formatting code..."
	source .venv/bin/activate && black pipeline/ tests/

health:
	@echo "Checking service health..."
	source .venv/bin/activate && python pipeline/health_check.py

logs:
	@echo "Viewing logs..."
	docker-compose logs -f

logs-producer:
	tail -f logs/producer_*.log

logs-consumer:
	tail -f logs/consumer_*.log

kafka-topics:
	@echo "Listing Kafka topics..."
	docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

kafka-console:
	@echo "Connecting to Kafka console consumer..."
	docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic secom-raw-data --from-beginning

db-shell:
	@echo "Connecting to PostgreSQL..."
	docker exec -it postgres psql -U ml_user -d secom_pipeline

install-dev:
	@echo "Installing development dependencies..."
	source .venv/bin/activate && pip install -r requirements.txt
	source .venv/bin/activate && pip install pytest-asyncio httpx black flake8 mypy
