# ADR-001: Use Kafka for Event Streaming

## Status
Accepted

## Context
The SECOM ML pipeline requires a reliable message queue for streaming synthetic manufacturing data from the producer to the consumer service. The system needs to:
- Handle high throughput (thousands of messages per second)
- Provide durable message storage for replay capability
- Support horizontal scaling through consumer groups
- Enable decoupling of data generation and processing

## Decision
Use Apache Kafka in KRaft mode (without Zookeeper) as the event streaming platform.

**Configuration:**
- 3 partitions for parallel processing
- Replication factor of 1 (development/demo)
- Topics: `secom-raw-data`, `secom-preprocessed-data`, `secom-dlq`

## Consequences

**Positive:**
- **High Throughput:** Kafka can handle millions of messages per second
- **Durability:** Messages are persisted to disk, enabling replay and recovery
- **Scalability:** Consumer groups allow horizontal scaling of processing
- **Decoupling:** Producers and consumers operate independently
- **KRaft Mode:** Eliminates Zookeeper dependency, simplifying operations

**Negative:**
- **Operational Complexity:** Kafka requires more operational knowledge than simpler queues
- **Resource Overhead:** Higher memory and disk usage compared to lightweight alternatives
- **Learning Curve:** Team needs to understand Kafka concepts (partitions, offsets, consumer groups)

## Alternatives Considered

### RabbitMQ
- **Pros:** Simpler operations, lower resource usage, better for traditional message queuing
- **Cons:** Lower throughput (~20K msgs/sec vs millions), less suitable for event streaming
- **Rejected:** Throughput limitations and less alignment with event-driven architecture

### Redis Streams
- **Pros:** Very low latency, simple setup, already using Redis
- **Cons:** Lower durability guarantees, less mature ecosystem, limited scaling
- **Rejected:** Insufficient durability for production ML pipeline

### AWS Kinesis / Google Pub/Sub
- **Pros:** Fully managed, no operational burden
- **Cons:** Vendor lock-in, cost at scale, requires cloud deployment
- **Rejected:** Project designed for self-hosted deployment
