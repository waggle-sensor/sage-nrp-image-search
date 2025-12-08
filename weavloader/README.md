# Weavloader

**Weavloader** is a distributed image processing service that continuously monitors SAGE data streams, processes images with AI models, and stores them in Weaviate for hybrid search capabilities.

## **What is Weavloader?**

Weavloader is a job processing system that:

- **Monitors SAGE data streams** for new images from environmental sensors
- **Processes images** using AI models (Gemma3, CLIP, Florence2) for captioning and embeddings
- **Stores processed data** in Weaviate vector database for hybrid search
- **Handles failures gracefully** with automatic retries and dead letter queues
- **Scales horizontally** to process thousands of images efficiently

## **Architecture**

```
SAGE Data Stream → Weavloader → AI Processing → Weaviate Database
                      ↓
                 Redis Queue (Celery)
                      ↓
              Dead Letter Queue (DLQ)
```

### **Project Structure:**

```
weavloader/
├── job_system/           # Celery job processing system
│   ├── tasks.py         # Celery tasks (image processing, monitoring, DLQ, etc.)
│   ├── celery_config.py # Celery configuration
│   └── flower_config.py # Flower monitoring configuration
├── inference/           # AI model inference
│   ├── model.py         # Triton client functions (Gemma3, CLIP, Florence2)
│   └── model_config.py  # Model hyperparameters and configuration
├── metrics/             # Prometheus metrics collection
│   ├── metrics.py       # Metrics definitions and collection
│   ├── server.py        # HTTP metrics server (unified endpoint)
│   └── artifacts/       # Prometheus configuration files
├── processing.py        # SAGE data stream processing
├── client.py            # Weaviate client initialization
├── main.py              # Application entry point
└── supervisord.conf     # Process management
```

### **Components:**

- **Job System**: Celery-based task processing with retries and DLQ management
- **Inference Engine**: AI model inference (Gemma3, CLIP, Florence2, ColBERT)
- **Metrics System**: Prometheus metrics collection and monitoring
- **Flower Monitoring**: Real-time Celery task and worker monitoring
- **Data Monitor**: Watches SAGE streams for new images
- **Redis**: Message broker and task queue
- **Weaviate**: Vector database for hybrid search

## **Key Features**

### **Job Processing:**
- **Automatic Retries**: Failed jobs retry with exponential backoff (60s → 120s → 240s)
- **Persistent Queues**: Jobs survive system restarts using Redis
   - If deployed using docker/k8s, redis requires volumes to persist data between restarts.
- **Horizontal Scaling**: Easy to add more workers
- **Dead Letter Queue**: Failed jobs archived for manual inspection

### **AI Inference:**
- **Multi-Model Support**: Gemma3, CLIP, Florence2, ColBERT, ALIGN
- **Triton Integration**: High-performance model serving
- **Configurable Hyperparameters**: Easy model tuning and optimization

### **Monitoring & Observability:**
- **Production Monitoring**: Built-in health checks and metrics
- **Tagged Logging**: Comprehensive logging with component tags
- **Prometheus Integration**: Detailed metrics collection with unified endpoint
- **Flower Monitoring**: Real-time Celery task and worker monitoring
- **Grafana Dashboards**: Real-time visualization

### **Modular Architecture:**
- **Separation of Concerns**: Clean separation between job processing, inference, and metrics
- **Easy Maintenance**: Each component can be updated independently
- **Scalable Design**: Components can be scaled separately based on demand

## **Prerequisites**

### **Required Services:**
- **Redis Server** (message broker)
- **Weaviate Database** (vector storage)
- **Triton Inference Server** (ML model serving)
- **SAGE Data Access** (environmental sensor data)

### **Environment Variables:**
   ```bash
# SAGE Data Access
   export SAGE_USER="your_username"
   export SAGE_PASS="your_password"

# ML Inference Server
   export TRITON_HOST="triton"
   export TRITON_PORT="8001"

# Database
export WEAVIATE_HOST="weaviate"
export WEAVIATE_PORT="8080"

# Celery Configuration
   export CELERY_BROKER_URL="redis://localhost:6379/0"
   export CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# Node Filtering (Optional)
export UNALLOWED_NODES="node1,node2,node3"  # Comma-separated list of nodes to exclude

# Logging Configuration (Optional)
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### **Node Filtering:**
The `UNALLOWED_NODES` environment variable allows you to exclude images from specific SAGE nodes from being processed:

- **Format**: Comma-separated list of node IDs (e.g., `"W001,W002,W003"`)
- **Case Insensitive**: Node IDs are converted to lowercase for comparison
- **Whitespace Tolerant**: Spaces around node IDs are automatically trimmed
- **Default**: Empty string (no nodes excluded)

**Example:**
   ```bash
# Exclude specific nodes from processing
export UNALLOWED_NODES="W001,W002,test-node"
   ```

## **Docker Deployment**

### **Quick Start:**
   ```bash
# 1. Set environment variables
export SAGE_USER="your_username"
export SAGE_PASS="your_password"

# 2. Start the entire stack
docker-compose up -d

# 3. Check logs
docker-compose logs weavloader
```

### **Container Architecture:**
- **Redis Server**: Message broker (port 6379)
- **Celery Stream**: Submits SAGE data stream task (default)
- **Celery Processor**: Processes image jobs from the image_processing queue
- **Celery Moderator**: Handles SAGE data stream task and DLQ health check
- **Celery Cleaner**: Manages DLQ management, DLQ reprocessing, task cleanup, system maintenance
- **Celery Scheduler**: Schedules periodic DLQ reprocessing, health checks, scheduled maintenance
- **Metrics Server**: Prometheus metrics endpoint (port 8080)
   - This port is exposed externally
- **Flower**: Celery monitoring UI (port 5555)
- **Supervisor**: Manages all processes

## **Worker Types**

Weavloader uses specialized Celery workers for different tasks:

### **Celery Processor**
- **Role**: Processes individual image jobs
- **Queue**: `image_processing`
- **Concurrency**: 3 workers
- **Node Name**: `processor@%h`
- **Purpose**: AI model inference, image captioning, embedding generation

### **Celery Moderator**
- **Role**: Handles SAGE data stream task
- **Queue**: `data_monitoring`
- **Concurrency**: 1 worker
- **Node Name**: `moderator@%h`
- **Purpose**: submits task to processor, stream health monitoring, DLQ health check

### **Celery Cleaner**
- **Role**: Manages cleanup and maintenance tasks
- **Queue**: `cleanup`
- **Concurrency**: 2 workers
- **Node Name**: `cleaner@%h`
- **Purpose**: DLQ management, task cleanup, system maintenance

### **Celery Scheduler**
- **Role**: Schedules periodic tasks and health checks
- **Queue**: None (runs as scheduler)
- **Concurrency**: 1 (single scheduler)
- **Node Name**: `scheduler@%h`
- **Purpose**: Periodic DLQ reprocessing, health checks, scheduled maintenance

## **Configuration**

### **Retry Settings** (`celery_config.py`):
- **Max Retries**: 3 attempts per task
- **Initial Delay**: 60 seconds
- **Backoff**: Exponential (60s → 120s → 240s)
- **Max Delay**: 10 minutes

### **Queue Configuration**:
- **`image_processing`**: Individual image processing tasks (handled by Processor workers)
- **`data_monitoring`**: Data monitoring and health check tasks (handled by Moderator workers)
- **`cleanup`**: Cleanup and maintenance tasks (handled by Cleaner workers)

### **Scheduled Tasks**:
- **Every 30 minutes**: Health check and monitoring
- **Every hour**: Archive failed tasks to DLQ
- **Daily**: Reprocess archived tasks

## **Manual Deployment**

### **Option 1: Direct Commands**
```bash
# Start Stream Monitor (default)
python main.py

# Start Processor Worker
python main.py processor

# Start Moderator Worker
python main.py moderator

# Start Cleaner Worker
python main.py cleaner
```

### **Option 3: Celery Commands**
```bash
# Start Processor Worker
celery -A job_system worker --loglevel=info --queues=image_processing --concurrency=3 -n processor@%h

# Start Moderator Worker
celery -A job_system worker --loglevel=info --queues=data_monitoring --concurrency=1 -n moderator@%h

# Start Cleaner Worker
celery -A job_system worker --loglevel=info --queues=cleanup --concurrency=2 -n cleaner@%h

# Start Scheduler
celery -A job_system beat --loglevel=info 

# Start Stream Monitor
python main.py
```

## **Monitoring & Observability**

### **Log Tags:**
- **`[MAIN]`**: Main file entry point
- **`[MODERATOR]`**: SAGE data stream monitoring and DLQ health check
- **`[PROCESSOR]`**: Image processing tasks
- **`[CLEANER]`**: Cleanup and maintenance tasks
- **`[METRICS]`**: Metrics collection and server

### **Monitoring Commands:**
```bash
# View active tasks
celery -A job_system inspect active

# Check queue status
redis-cli
> LLEN image_processing
> LLEN data_monitoring
> LLEN cleanup
> KEYS dlq:*

# View health metrics
docker-compose logs weavloader | grep "\[METRICS\]"
docker-compose logs weavloader | grep "\[MODERATOR\]"
```

### **Flower Monitoring:**
Flower is automatically started with the weavloader container and provides real-time monitoring of Celery tasks and workers.

**Access Points:**
- **Local Development**: http://localhost:5555
- **Authentication**: admin:weavloader123
- **Features**: Real-time task monitoring, worker status, queue management

**Flower Configuration:**
- **Config File**: `job_system/flower_config.py`
- **Port**: 5555 (Web UI)
- **Events**: Enabled for real-time updates
- **Persistence**: Disabled by default (can be enabled)
- **Logging**: Uses `LOG_LEVEL` environment variable

## **Dead Letter Queue (DLQ)**

### **How it Works:**
1. **Failed Tasks**: After 3 retries, tasks move to DLQ
2. **Archiving**: Tasks stored with full error context for 30 days
3. **Reprocessing**: Daily automatic retry of archived tasks
4. **Cleanup**: Very old tasks (>7 days) are removed

### **DLQ Management:**
```bash
# Check DLQ size
redis-cli
> KEYS dlq:*
> LLEN dlq:*

# View archived task details
> GET dlq:task-id-here

# Manual reprocessing
python -c "from tasks import reprocess_dlq_tasks; reprocess_dlq_tasks.delay()"
```

## **Scaling**

### **Add More Workers:**
```bash
# Start additional workers
celery -A tasks worker --loglevel=info --concurrency=2 --hostname=worker2@%h
celery -A tasks worker --loglevel=info --concurrency=2 --hostname=worker3@%h
```

### **Horizontal Scaling:**
- **Workers**: Scale based on image processing load
- **Redis**: Use Redis Cluster for high availability
- **Weaviate**: Scale Weaviate cluster for storage

## **Troubleshooting**

### **Common Issues:**

1. **Redis Connection Error**:
   ```bash
   redis-cli ping  # Should return PONG
   ```

2. **Tasks Not Processing**:
   ```bash
   # Check worker status
   celery -A tasks inspect active
   
   # Check queue lengths
   redis-cli
   > LLEN image_processing
   ```

3. **High Memory Usage**:
   ```bash
   # Reduce concurrency
   celery -A tasks worker --concurrency=2
   
   # Monitor Redis memory
   redis-cli info memory
   ```

4. **DLQ Growing Large**:
   ```bash
   # Check DLQ size
   redis-cli
   > KEYS dlq:* | wc -l
   
   # Manual cleanup
   python -c "from tasks import cleanup_failed_tasks; cleanup_failed_tasks.delay()"
   ```

### **Debug Commands:**
```bash
# Check Celery status
celery -A tasks inspect active
celery -A tasks inspect registered

# Purge queues (use with caution)
celery -A tasks purge

# Check Redis status
redis-cli info
redis-cli monitor
```

## **Advanced Configuration**

### **Custom Retry Logic:**
Edit `celery_config.py` to modify:
- Retry delays
- Max retries
- Queue routing
- Task timeouts

### **Environment-Specific Settings:**
```bash
# Development
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# Production
export CELERY_BROKER_URL="redis://redis-cluster:6379/0"
export CELERY_RESULT_BACKEND="redis://redis-cluster:6379/0"
```

## **Metrics & Monitoring**

### **Prometheus Metrics:**
Weavloader exposes comprehensive metrics on port 8080 with a **unified endpoint** that includes both custom weavloader metrics and Flower Celery metrics:

- **Task Metrics**: Processing rates, success rates, duration
- **Queue Metrics**: Queue sizes, DLQ status
- **System Metrics**: Memory usage, worker count, health status
- **SAGE Metrics**: Stream health, image reception rates
- **Model Metrics**: Inference duration, error rates
- **Error Metrics**: Error rates by component
- **Celery Metrics**: Worker status, task execution, queue performance (prefixed with `weavloader_`)

### **Accessing Metrics:**
```bash
# View unified metrics endpoint (includes both custom and Flower metrics)
curl http://localhost:8080/metrics

# Health check
curl http://localhost:8080/health

# Filter for specific metric types
curl http://localhost:8080/metrics | grep weavloader_celery
curl http://localhost:8080/metrics | grep weavloader_tasks
```

### **Grafana Dashboard:**
Import the provided `grafana-dashboard.json` to visualize:
- Task processing rates and success rates
- Queue sizes and DLQ status
- System health and component status
- Memory usage and performance metrics
- SAGE data stream monitoring

### **Prometheus Configuration:**
Use the provided `prometheus.yml` to scrape metrics from:
- Weavloader (port 8080)
- Redis (port 6379)
- Weaviate (port 8080)

## **Configuration Files:**

### **Core Application:**
- **`main.py`**: Application entry point
- **`processing.py`**: SAGE data stream processing
- **`client.py`**: Weaviate client initialization
- **`supervisord.conf`**: Process management
- **`Dockerfile`**: Container configuration
- **`requirements.txt`**: Python dependencies

### **Job System (`job_system/`):**
- **`tasks.py`**: Celery tasks (image processing, monitoring, DLQ management)
- **`celery_config.py`**: Celery configuration and routing
- **`flower_config.py`**: Flower monitoring configuration

### **Inference Engine (`inference/`):**
- **`model.py`**: Triton client functions (Gemma3, CLIP, Florence2, ColBERT)
- **`model_config.py`**: Model hyperparameters and configuration

### **Metrics System (`metrics/`):**
- **`metrics.py`**: Prometheus metrics definitions and collection
- **`server.py`**: HTTP metrics server (unified endpoint)
- **`artifacts/`**: Prometheus configuration files

### **Monitoring & Deployment:**
- **`grafana-dashboard.json`**: Grafana dashboard configuration
- **`prometheus.yml`**: Prometheus scraping configuration
- **`kubernetes-complete.yaml`**: Kubernetes deployment manifests

## **Development Workflow**

### **Adding New Models:**
1. Add model functions to `inference/model.py`
2. Update hyperparameters in `inference/model_config.py`
3. Export new functions in `inference/__init__.py`
4. Update imports in `processing.py` if needed

### **Adding New Tasks:**
1. Add task functions to `job_system/tasks.py`
2. Update routing in `job_system/celery_config.py`
3. Export new tasks in `job_system/__init__.py`
4. Update supervisor configuration if needed

### **Adding New Metrics:**
1. Add metric definitions to `metrics/metrics.py`
2. Update collection logic in `metrics/server.py`
3. Export new metrics in `metrics/__init__.py`
4. Update Grafana dashboard configuration

### **Testing Components:**
```bash
# Test job system
python -c "from job_system import celery_app; print('Job system OK')"

# Test inference
python -c "from inference import gemma3_run_model; print('Inference OK')"

# Test metrics
python -c "from metrics import metrics; print('Metrics OK')"

# Test Flower configuration
python -c "from job_system.flower_config import basic_auth; print('Flower config OK')"
```

## **Flower Monitoring**

### **Accessing Flower:**
- **Local Development**: http://localhost:5555
- **Authentication**: admin:weavloader123
- **Features**: Real-time task monitoring, worker status, queue management

### **Flower Features:**
- **Real-time Monitoring**: Live updates of task execution
- **Worker Management**: View active workers and their status
- **Task History**: Track task execution and results
- **Queue Monitoring**: Monitor queue sizes and processing rates
- **Error Tracking**: View failed tasks and error details

### **Flower Configuration:**
The Flower configuration is managed through `job_system/flower_config.py` and includes:
- **Authentication**: Basic auth with configurable credentials
- **Logging**: Uses `LOG_LEVEL` environment variable
- **Events**: Real-time event monitoring enabled
- **Persistence**: Configurable state persistence
- **Monitoring**: Worker and task monitoring settings

### **Flower Metrics Integration:**
Flower metrics are automatically integrated into the unified Prometheus endpoint:
- **Metric Prefix**: All Flower metrics are prefixed with `weavloader_`
- **Unified Endpoint**: Available at `/metrics` alongside custom metrics
- **Real-time Updates**: Metrics update in real-time with task execution
