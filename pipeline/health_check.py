#!/usr/bin/env python3
"""
Health check script for SECOM ML Pipeline services
"""

import sys
import requests
from kafka import KafkaAdminClient
from kafka.errors import KafkaError
import psycopg2
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()


def check_kafka():
    """Check Kafka connectivity"""
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
            request_timeout_ms=5000
        )
        topics = admin_client.list_topics()
        admin_client.close()
        logger.info(f"Kafka is healthy. Topics: {len(topics)}")
        return True
    except KafkaError as e:
        logger.error(f"x Kafka health check failed: {e}")
        return False


def check_postgres():
    """Check PostgreSQL connectivity"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DB', 'secom_pipeline'),
            user=os.getenv('POSTGRES_USER', 'ml_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'ml_password'),
            connect_timeout=5
        )
        cur = conn.cursor()
        cur.execute('SELECT 1')
        cur.close()
        conn.close()
        logger.info("PostgreSQL is healthy")
        return True
    except Exception as e:
        logger.error(f"x PostgreSQL health check failed: {e}")
        return False


def check_prometheus():
    """Check Prometheus metrics endpoint"""
    try:
        # Check producer metrics
        response = requests.get('http://localhost:8000/metrics', timeout=5)
        if response.status_code == 200:
            logger.info("Producer metrics endpoint is healthy")
            producer_healthy = True
        else:
            logger.warning("⚠ Producer metrics endpoint returned non-200 status")
            producer_healthy = False
    except Exception as e:
        logger.warning(f"⚠ Producer metrics not available: {e}")
        producer_healthy = False
    
    try:
        # Check consumer metrics
        response = requests.get('http://localhost:8001/metrics', timeout=5)
        if response.status_code == 200:
            logger.info("Consumer metrics endpoint is healthy")
            consumer_healthy = True
        else:
            logger.warning("⚠ Consumer metrics endpoint returned non-200 status")
            consumer_healthy = False
    except Exception as e:
        logger.warning(f"⚠ Consumer metrics not available: {e}")
        consumer_healthy = False
    
    return producer_healthy or consumer_healthy


def main():
    """Run all health checks"""
    logger.info("=" * 60)
    logger.info("SECOM ML Pipeline - Health Check")
    logger.info("=" * 60)
    
    checks = {
        'Kafka': check_kafka(),
        'PostgreSQL': check_postgres(),
        'Metrics': check_prometheus()
    }
    
    logger.info("=" * 60)
    logger.info("Health Check Summary:")
    for service, status in checks.items():
        status_symbol = "✓" if status else "x"
        logger.info(f"  {status_symbol} {service}: {'Healthy' if status else 'Unhealthy'}")
    logger.info("=" * 60)
    
    # Exit with error if any check failed
    if not all(checks.values()):
        sys.exit(1)
    
    logger.info("All health checks passed!")


if __name__ == "__main__":
    main()
