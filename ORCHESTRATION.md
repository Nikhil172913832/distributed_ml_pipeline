# Scalable Orchestration Guide

## Overview

This guide explains how to scale the ML pipeline horizontally for production workloads.

## Kubernetes Deployment (Recommended for Production)

### Current Setup

The K8s manifests in `k8s/` are already configured for horizontal scaling:

- **Inference Service**: Configured with HPA (Horizontal Pod Autoscaler)
  - Min replicas: 2
  - Max replicas: 10
  - Scales based on CPU (70%) and memory (80%) utilization
  
- **Training Jobs**: Parallel job execution with configurable parallelism

### Scaling Configuration

#### 1. Inference Service Scaling

Edit `k8s/inference-deployment.yaml` to adjust scaling parameters:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  minReplicas: 2        # Minimum pods
  maxReplicas: 10       # Maximum pods
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

#### 2. Producer/Consumer Scaling

Scale Kafka consumers for higher throughput:

```bash
kubectl scale deployment consumer --replicas=5 -n ml-pipeline
```

#### 3. Resource Limits

Set appropriate resource requests/limits based on workload:

```yaml
resources:
  limits:
    memory: "4Gi"
    cpu: "2"
  requests:
    memory: "2Gi"
    cpu: "1"
```

### Monitoring Scaling Events

```bash
# Watch HPA status
kubectl get hpa -n ml-pipeline -w

# View scaling events
kubectl describe hpa inference-hpa -n ml-pipeline

# Monitor pod status
kubectl get pods -n ml-pipeline -w
```

## Docker Compose Scaling

For development or smaller deployments, use the enhanced docker-compose:

### docker-compose.scale.yml Features

- Service replicas for load distribution
- Health checks for all services
- Volume management for shared state
- Network isolation

### Scaling Services

```bash
# Scale inference service
docker-compose -f docker-compose.scale.yml up -d --scale inference=3

# Scale consumer for higher throughput
docker-compose -f docker-compose.scale.yml up -d --scale consumer=2

# View scaled services
docker-compose -f docker-compose.scale.yml ps
```

### Load Balancing

For HTTP services, add nginx or traefik:

```yaml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf:ro
  depends_on:
    - inference
```

## Microservices Architecture

### Service Boundaries

1. **Data Ingestion** (Producer)
   - Scales based on input data rate
   - Stateless, horizontally scalable
   
2. **Data Processing** (Consumer)
   - Scales based on Kafka lag
   - Idempotent processing recommended
   
3. **Inference** (Inference Service)
   - Scales based on request rate
   - Stateless, load balanced
   
4. **Training** (Trainer)
   - Scales based on training workload
   - Can run as parallel jobs

### Service Communication

- **Kafka**: Async messaging between services
- **PostgreSQL**: Shared data store (connection pooling required)
- **REST APIs**: Sync communication for inference

### Best Practices

1. **Stateless Services**: Keep services stateless for easy scaling
2. **Health Checks**: Implement liveness and readiness probes
3. **Graceful Shutdown**: Handle SIGTERM for clean shutdowns
4. **Connection Pooling**: Use connection pools for database access
5. **Circuit Breakers**: Implement circuit breakers for external dependencies

## Production Considerations

### High Availability

- Run multiple replicas of each service
- Use pod disruption budgets in K8s
- Configure anti-affinity rules to spread pods across nodes

```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - inference
        topologyKey: kubernetes.io/hostname
```

### Resource Planning

| Service | CPU (per pod) | Memory (per pod) | Min Replicas | Max Replicas |
|---------|---------------|------------------|--------------|--------------|
| Producer | 0.5 | 1Gi | 1 | 3 |
| Consumer | 1.0 | 2Gi | 2 | 5 |
| Inference | 2.0 | 4Gi | 2 | 10 |
| Kafka | 2.0 | 4Gi | 3 | 3 |
| PostgreSQL | 4.0 | 8Gi | 1 | 1 |

### Monitoring Metrics

Track these metrics for scaling decisions:

- **Kafka lag**: Consumer group lag
- **Request rate**: Requests per second to inference
- **Response time**: P95, P99 latencies
- **Error rate**: 4xx, 5xx errors
- **Resource utilization**: CPU, memory, disk

### Auto-scaling Policies

```yaml
# CPU-based scaling
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70

# Custom metric scaling (requires metrics-server)
- type: Pods
  pods:
    metric:
      name: http_requests_per_second
    target:
      type: AverageValue
      averageValue: "1000"
```

## Troubleshooting

### Pod not scaling

1. Check HPA status: `kubectl describe hpa <name>`
2. Verify metrics server: `kubectl top nodes`
3. Check resource requests are set
4. Review scaling events: `kubectl get events`

### Service unavailable during scaling

1. Implement readiness probes correctly
2. Use rolling update strategy
3. Set appropriate `minReadySeconds`
4. Configure pod disruption budgets

### Database connection exhaustion

1. Implement connection pooling
2. Scale database connections with service replicas
3. Monitor active connections
4. Use read replicas for read-heavy workloads

## Migration Path

### Phase 1: Docker Compose (Current)
- Suitable for development and testing
- Limited horizontal scaling

### Phase 2: Docker Compose with Replicas
- Use docker-compose.scale.yml
- Add load balancer (nginx/traefik)
- Suitable for small production deployments

### Phase 3: Kubernetes
- Full horizontal scaling capabilities
- Auto-scaling based on metrics
- Production-ready with HA
- Use provided K8s manifests in `k8s/`

### Phase 4: Managed Services
- Consider managed Kafka (Confluent Cloud, AWS MSK)
- Managed PostgreSQL (RDS, Cloud SQL)
- Managed Kubernetes (EKS, GKE, AKS)
- Focus on application code, not infrastructure
