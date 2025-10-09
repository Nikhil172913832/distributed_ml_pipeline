# Makefile for SECOM ML Pipeline

.PHONY: help setup start stop restart clean test lint format health logs bench bench-python bench-all

help:
	@echo "SECOM ML Pipeline - Available Commands:"
	@echo ""
	@echo "  === Service Management ==="
	@echo "  make setup          - Initial setup (install deps, start services)"
	@echo "  make start          - Start all Docker services"
	@echo "  make stop           - Stop all Docker services"
	@echo "  make restart        - Restart all services"
	@echo "  make clean          - Stop services and remove volumes"
	@echo ""
	@echo "  === Run Components ==="
	@echo "  make producer       - Run the data producer"
	@echo "  make consumer       - Run the data consumer"
	@echo "  make inference      - Run the inference engine"
	@echo "  make retrainer      - Run the retrainer service"
	@echo "  make train          - Train models manually"
	@echo ""
	@echo "  === Development ==="
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linting"
	@echo "  make format         - Format code with black"
	@echo ""
	@echo "  === Monitoring ==="
	@echo "  make health         - Check service health"
	@echo "  make logs           - View all logs"
	@echo "  make status         - Show pipeline status"
	@echo "  make metrics        - Show current metrics"
	@echo ""
	@echo "  === ML Operations ==="
	@echo "  make model-info     - Show active model info"
	@echo "  make performance    - Show model performance"
	@echo "  make drift-status   - Show drift detection status"
	@echo "  make trigger-retrain - Manually trigger retraining"
	@echo ""
	@echo "  === Benchmarking ==="
	@echo "  make bench          - Run k6 load test (recommended)"
	@echo "  make bench-python   - Run Python benchmark script"
	@echo "  make bench-all      - Run all benchmarks"
	@echo ""

setup:
	@echo "Setting up SECOM ML Pipeline..."
	chmod +x setup.sh
	./setup.sh

start:
	@echo "Starting Docker services..."
	docker-compose up -d
	@echo ""
	@echo "Service URLs:"
	@echo "  - Kafka UI:   http://localhost:8080"
	@echo "  - pgAdmin:    http://localhost:8081"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana:    http://localhost:3000 (admin/admin)"

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

inference:
	@echo "Starting inference engine..."
	source .venv/bin/activate && python pipeline/inference.py

retrainer:
	@echo "Starting retrainer service..."
	source .venv/bin/activate && python pipeline/retrainer.py

train:
	@echo "Training models..."
	source .venv/bin/activate && python pipeline/model_trainer.py --triggered-by manual --auto-deploy

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

logs-inference:
	tail -f logs/inference_*.log

logs-retrainer:
	tail -f logs/retrainer_*.log

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

status:
	@echo "=== SECOM ML Pipeline Status ==="
	@echo ""
	@docker-compose ps
	@echo ""
	@echo "Recent Activity:"
	@docker exec -i postgres psql -U ml_user -d secom_pipeline -c \
		"SELECT event_type, event_status, component, created_at FROM secom.pipeline_audit_log ORDER BY created_at DESC LIMIT 5;" 2>/dev/null || echo "Database not ready"

metrics:
	@echo "Fetching current metrics..."
	@curl -s http://localhost:9090/api/v1/query?query=secom_predictions_made_total 2>/dev/null | grep -o '"value":\[[^]]*\]' || echo "Prometheus not ready"

model-info:
	@echo "=== Active Model Information ==="
	@docker exec -i postgres psql -U ml_user -d secom_pipeline -c \
		"SELECT model_name, model_version, model_type, test_accuracy, test_f1_score, deployed_at FROM secom.model_registry WHERE is_active = TRUE;"

performance:
	@echo "=== Model Performance (Last 24h) ==="
	@docker exec -i postgres psql -U ml_user -d secom_pipeline -c \
		"SELECT window_start, window_end, accuracy, precision, recall, f1_score, total_predictions FROM secom.model_performance_metrics ORDER BY window_end DESC LIMIT 10;"

drift-status:
	@echo "=== Drift Detection Status ==="
	@docker exec -i postgres psql -U ml_user -d secom_pipeline -c \
		"SELECT * FROM secom.drift_detection_summary;"

trigger-retrain:
	@echo "Triggering manual retraining..."
	@docker exec -i postgres psql -U ml_user -d secom_pipeline -c \
		"INSERT INTO secom.retraining_triggers (trigger_type, trigger_reason, status) VALUES ('manual', 'Manual trigger via Makefile', 'pending');"
	@echo "✓ Retraining triggered! Monitor with 'make logs-retrainer'"

bench:
	@echo "=== Running k6 Load Test ==="
	@echo ""
	@if ! command -v k6 &> /dev/null; then \
		echo "Error: k6 is not installed!"; \
		echo ""; \
		echo "Install k6:"; \
		echo "  macOS:   brew install k6"; \
		echo "  Linux:   sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69"; \
		echo "           echo 'deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main' | sudo tee /etc/apt/sources.list.d/k6.list"; \
		echo "           sudo apt-get update"; \
		echo "           sudo apt-get install k6"; \
		echo ""; \
		echo "Or use: make bench-python"; \
		exit 1; \
	fi
	@echo "Checking if services are running..."
	@docker-compose ps | grep -q "Up" || (echo "Error: Services not running. Run 'make start' first." && exit 1)
	@echo ""
	k6 run benchmarks/k6-inference.js
	@echo ""
	@echo "✓ Benchmark complete! Check output above."

bench-python:
	@echo "=== Running Python Benchmark ==="
	@echo ""
	@echo "Checking if services are running..."
	@docker-compose ps | grep -q "Up" || (echo "Error: Services not running. Run 'make start' first." && exit 1)
	@echo ""
	@timestamp=$$(date +%Y-%m-%d_%H%M%S); \
	source .venv/bin/activate && \
	python benchmarks/benchmark_pipeline.py \
		--duration 60 \
		--output "benchmarks/results/benchmark_$$timestamp.json"
	@echo ""
	@echo "✓ Benchmark saved to benchmarks/results/"
	@echo "  Latest: benchmarks/results/benchmark_latest.json"

bench-all: bench-python bench
	@echo ""
	@echo "=== All Benchmarks Complete ==="
	@echo ""
	@echo "Results saved to:"
	@ls -lht benchmarks/results/ | head -n 5
