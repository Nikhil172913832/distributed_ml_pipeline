/**
 * k6 Load Test for SECOM ML Pipeline - Inference Performance
 * 
 * This script measures the end-to-end performance of the inference pipeline
 * by monitoring the rate at which predictions are being made.
 * 
 * Prerequisites:
 * - All services running (make up)
 * - Producer generating data
 * - Consumer processing data
 * - Inference service running
 * 
 * Run with:
 *   k6 run benchmarks/k6-inference.js
 * 
 * Or use:
 *   make bench
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const latencyTrend = new Trend('query_latency');
const predictionsCounter = new Counter('predictions_made');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 5 },   // Ramp up to 5 VUs over 30s
    { duration: '1m', target: 5 },    // Stay at 5 VUs for 1 minute
    { duration: '30s', target: 10 },  // Ramp up to 10 VUs
    { duration: '1m', target: 10 },   // Stay at 10 VUs for 1 minute
    { duration: '30s', target: 0 },   // Ramp down to 0 VUs
  ],
  thresholds: {
    'http_req_duration': ['p(95)<1000'], // 95% of requests should be below 1s
    'errors': ['rate<0.1'],              // Error rate should be below 10%
  },
};

const PROMETHEUS_URL = 'http://localhost:9090';
const CHECK_INTERVAL = 5; // seconds

/**
 * Query Prometheus for prediction rate
 */
function queryPredictionRate() {
  const query = 'rate(secom_predictions_made_total[1m])';
  const url = `${PROMETHEUS_URL}/api/v1/query?query=${encodeURIComponent(query)}`;
  
  const startTime = new Date().getTime();
  const response = http.get(url);
  const duration = new Date().getTime() - startTime;
  
  latencyTrend.add(duration);
  
  const success = check(response, {
    'prometheus query successful': (r) => r.status === 200,
    'has prediction data': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.status === 'success' && data.data.result.length > 0;
      } catch (e) {
        return false;
      }
    },
  });
  
  if (!success) {
    errorRate.add(1);
    return 0;
  }
  
  errorRate.add(0);
  
  // Extract the prediction rate
  try {
    const data = JSON.parse(response.body);
    if (data.data.result.length > 0) {
      const rate = parseFloat(data.data.result[0].value[1]);
      predictionsCounter.add(rate * CHECK_INTERVAL);
      return rate;
    }
  } catch (e) {
    console.error(`Failed to parse response: ${e}`);
  }
  
  return 0;
}

/**
 * Query Prometheus for inference latency
 */
function queryInferenceLatency() {
  const query = 'histogram_quantile(0.95, secom_inference_duration_seconds_bucket)';
  const url = `${PROMETHEUS_URL}/api/v1/query?query=${encodeURIComponent(query)}`;
  
  const response = http.get(url);
  
  check(response, {
    'latency query successful': (r) => r.status === 200,
  });
  
  try {
    const data = JSON.parse(response.body);
    if (data.data.result.length > 0) {
      const latency = parseFloat(data.data.result[0].value[1]);
      return latency * 1000; // Convert to milliseconds
    }
  } catch (e) {
    console.error(`Failed to parse latency: ${e}`);
  }
  
  return 0;
}

/**
 * Query Prometheus for model accuracy
 */
function queryModelAccuracy() {
  const query = 'secom_model_accuracy';
  const url = `${PROMETHEUS_URL}/api/v1/query?query=${encodeURIComponent(query)}`;
  
  const response = http.get(url);
  
  check(response, {
    'accuracy query successful': (r) => r.status === 200,
  });
  
  try {
    const data = JSON.parse(response.body);
    if (data.data.result.length > 0) {
      return parseFloat(data.data.result[0].value[1]);
    }
  } catch (e) {
    console.error(`Failed to parse accuracy: ${e}`);
  }
  
  return 0;
}

/**
 * Main test scenario
 */
export default function() {
  // Query prediction rate
  const predictionRate = queryPredictionRate();
  
  // Query inference latency
  const inferenceLatency = queryInferenceLatency();
  
  // Query model accuracy
  const modelAccuracy = queryModelAccuracy();
  
  // Log results
  console.log(`Prediction Rate: ${predictionRate.toFixed(2)} predictions/sec`);
  console.log(`Inference Latency (p95): ${inferenceLatency.toFixed(2)} ms`);
  console.log(`Model Accuracy: ${(modelAccuracy * 100).toFixed(2)}%`);
  
  // Wait before next check
  sleep(CHECK_INTERVAL);
}

/**
 * Setup function - runs once before the test
 */
export function setup() {
  console.log('='.repeat(60));
  console.log('SECOM ML Pipeline - Inference Benchmark');
  console.log('='.repeat(60));
  console.log(`Prometheus URL: ${PROMETHEUS_URL}`);
  console.log(`Check Interval: ${CHECK_INTERVAL}s`);
  console.log('='.repeat(60));
  
  // Verify Prometheus is accessible
  const response = http.get(`${PROMETHEUS_URL}/-/healthy`);
  if (response.status !== 200) {
    throw new Error('Prometheus is not accessible. Make sure all services are running.');
  }
  
  console.log('✓ Prometheus is accessible');
  console.log('Starting benchmark...\n');
}

/**
 * Teardown function - runs once after the test
 */
export function teardown(data) {
  console.log('\n' + '='.repeat(60));
  console.log('Benchmark Complete!');
  console.log('='.repeat(60));
  console.log('Check the summary above for detailed metrics.');
  console.log('Results are also available in Grafana: http://localhost:3000');
  console.log('='.repeat(60));
}
