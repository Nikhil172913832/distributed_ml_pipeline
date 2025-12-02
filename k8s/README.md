# Kubernetes Deployment Guide

## Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- NVIDIA GPU operator installed (for GPU training)
- Storage provisioner configured

## Quick Start

### 1. Create Namespace and Resources

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/pvc.yaml
```

### 2. Deploy Training Job

```bash
# Update image registry in training-job.yaml
kubectl apply -f k8s/training-job.yaml

# Monitor training
kubectl logs -n ml-pipeline -l app=distributed-training --follow
```

### 3. Deploy Inference Service

```bash
# Update image registry in inference-deployment.yaml
kubectl apply -f k8s/inference-deployment.yaml

# Check service status
kubectl get svc -n ml-pipeline inference-service
```

## Configuration

### Update Secrets

```bash
# Edit secrets
kubectl edit secret ml-pipeline-secrets -n ml-pipeline

# Or create from file
kubectl create secret generic ml-pipeline-secrets \
  --from-literal=database-password=your-password \
  --from-literal=wandb-api-key=your-key \
  -n ml-pipeline
```

### Scale Inference Service

```bash
# Manual scaling
kubectl scale deployment inference-service --replicas=5 -n ml-pipeline

# HPA will auto-scale based on CPU/memory
kubectl get hpa -n ml-pipeline
```

## Monitoring

```bash
# View logs
kubectl logs -n ml-pipeline -l app=inference

# Get pod status
kubectl get pods -n ml-pipeline

# Describe resources
kubectl describe job distributed-training -n ml-pipeline
```

## Cleanup

```bash
kubectl delete namespace ml-pipeline
```

## Notes

- Update image registries in deployment files
- Adjust resource requests/limits based on your cluster
- Configure storage class for your environment
- Update node selectors for GPU nodes
