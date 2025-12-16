# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting key architectural decisions made in this project.

## Index

1. [ADR-001: Use Kafka for Event Streaming](001-kafka-event-streaming.md)
2. [ADR-002: PostgreSQL for Data Persistence](002-postgresql-data-persistence.md)
3. [ADR-003: Pydantic for Configuration Management](003-pydantic-configuration.md)
4. [ADR-004: MLflow for Experiment Tracking](004-mlflow-experiment-tracking.md)
5. [ADR-005: Redis for Feature Caching](005-redis-feature-store.md)

## ADR Template

```markdown
# ADR-XXX: [Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
[What is the issue we're seeing that is motivating this decision or change?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
**Positive:**
- [Benefit 1]
- [Benefit 2]

**Negative:**
- [Drawback 1]
- [Drawback 2]

## Alternatives Considered
- **Alternative 1:** [Description and why rejected]
- **Alternative 2:** [Description and why rejected]
```
