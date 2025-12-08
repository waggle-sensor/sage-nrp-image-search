'''Prometheus Metrics for Weavloader'''

import os
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import time
import logging
from prometheus_client.multiprocess import MultiProcessCollector

# Create multiprocess directory if it doesn't exist
os.makedirs(os.environ['PROMETHEUS_MULTIPROC_DIR'], exist_ok=True)

# === TASK METRICS ===
# Counters for task processing
tasks_retried_total = Counter(
    'weavloader_tasks_retried_total',
    'Total number of task retries',
    ['task', 'retry_reason'],
)

# === QUEUE METRICS ===
# Queue sizes
queue_size = Gauge(
    'weavloader_queue_size',
    'Current queue size',
    ['queue_name'],
    multiprocess_mode='mostrecent'
)

# DLQ metrics
dlq_size = Gauge(
    'weavloader_dlq_size',
    'Dead letter queue size',
    multiprocess_mode='mostrecent'
)

dlq_tasks_thrown_away_total = Counter(
    'weavloader_dlq_tasks_thrown_away_total',
    'Total tasks thrown away from DLQ',
    ['node_id', 'job', 'task', 'camera'],
)

dlq_tasks_reprocessed_total = Counter(
    'weavloader_dlq_tasks_reprocessed_total',
    'Total tasks reprocessed from DLQ',
    ['status'],  # status: success, failure
)

# === SYSTEM METRICS ===
# Memory usage
memory_usage_bytes = Gauge(
    'weavloader_memory_usage_bytes',
    'Memory usage in bytes',
    ['worker_type'],  # worker_type: processor, moderator, cleaner, metric_server
    multiprocess_mode='all'
)

# === DATA STREAM METRICS ===
# SAGE data stream metrics
sage_images_received_total = Counter(
    'weavloader_sage_images_received_total',
    'Total images received from SAGE',
    ['node_id', 'job', 'task', 'camera'],
)

sage_stream_health = Gauge(
    'weavloader_sage_stream_health',
    'SAGE stream health (1=healthy, 0=unhealthy)',
    multiprocess_mode='mostrecent'
)

# === AI MODEL METRICS ===
# Model inference metrics
model_inference_duration = Histogram(
    'weavloader_model_inference_duration_seconds',
    'Time spent on model inference',
    ['model_name', 'operation', 'status'],  # operation: caption, embedding, status: success, failure
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

model_inference_total = Counter(
    'weavloader_model_inference_total',
    'Total model inference calls',
    ['model_name', 'operation', 'status'],
)

# === DATABASE METRICS ===
# Weaviate operations
weaviate_operations_total = Counter(
    'weavloader_weaviate_operations_total',
    'Total Weaviate operations',
    ['operation', 'status'],  # operation: insert, query, delete
)

weaviate_operation_duration = Histogram(
    'weavloader_weaviate_operation_duration_seconds',
    'Time spent on Weaviate operations',
    ['operation', 'status'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# === ERROR METRICS ===
errors_total = Counter(
    'weavloader_errors_total',
    'Total number of errors',
    ['component', 'error_type'],
)

# === HEALTH METRICS ===
component_health = Gauge(
    'weavloader_component_health',
    'Component health status',
    ['component'],  # component: redis, weaviate, triton, sage
    multiprocess_mode='mostrecent'
)

class MetricsCollector:
    """Metrics collection and management for Weavloader"""
    def record_task_retry(self, task: str, retry_reason: str):
        """Record a task retry"""
        tasks_retried_total.labels(task=task, retry_reason=retry_reason).inc()
        logging.debug(f"[METRICS] Task retried: {task} - {retry_reason}") 
    
    def update_queue_size(self, queue_name: str, size: int):
        """Update queue size"""
        queue_size.labels(queue_name=queue_name).set(size)
        logging.debug(f"[METRICS] Queue size: {queue_name} - {size}")
    
    def update_dlq_size(self, size: int):
        """Update DLQ size"""
        dlq_size.set(size)
        logging.debug(f"[METRICS] DLQ size: {size}")
    
    def record_dlq_throw_away(self, node_id: str, job: str, task: str, camera: str):
        """Record DLQ throw away"""
        dlq_tasks_thrown_away_total.labels(node_id=node_id, job=job, task=task, camera=camera).inc()
        logging.debug(f"[METRICS] DLQ task was thrown away: {node_id} - {job} - {task} - {camera}")
    
    def record_dlq_reprocess(self, status: str):
        """Record DLQ reprocess"""
        dlq_tasks_reprocessed_total.labels(status=status).inc()
        logging.debug(f"[METRICS] DLQ reprocess: {status}")
    
    def record_sage_image(self, node_id: str, job: str, task: str, camera: str):
        """Record SAGE image received"""
        sage_images_received_total.labels(node_id=node_id, job=job, task=task, camera=camera).inc()
        logging.debug(f"[METRICS] SAGE image: {node_id} - {job} - {task} - {camera}")
    
    def update_sage_stream_health(self, healthy: bool):
        """Update SAGE stream health"""
        sage_stream_health.set(1 if healthy else 0)
        logging.debug(f"[METRICS] SAGE stream health: {healthy}")
    
    def record_model_inference(self, model_name: str, operation: str, duration: float, status: str):
        """Record model inference"""
        model_inference_duration.labels(model_name=model_name, operation=operation, status=status).observe(duration)
        model_inference_total.labels(model_name=model_name, operation=operation, status=status).inc()
        logging.debug(f"[METRICS] Model inference: {model_name} - {operation} - {duration:.2f}s - {status}")
    
    def record_weaviate_operation(self, operation: str, status: str, duration: float):
        """Record Weaviate operation"""
        weaviate_operations_total.labels(operation=operation, status=status).inc()
        weaviate_operation_duration.labels(operation=operation, status=status).observe(duration)
        logging.debug(f"[METRICS] Weaviate operation: {operation} - {status} - {duration:.2f}s")
    
    def record_error(self, component: str, error_type: str):
        """Record an error"""
        errors_total.labels(component=component, error_type=error_type).inc()
        logging.debug(f"[METRICS] Error: {component} - {error_type}")
    
    def update_component_health(self, component: str, healthy: bool):
        """Update component health"""
        component_health.labels(component=component).set(1 if healthy else 0)
        logging.debug(f"[METRICS] Component health: {component} - {healthy}")
    
    def update_memory_usage(self, worker_type: str, bytes_used: int):
        """Update memory usage"""
        memory_usage_bytes.labels(worker_type=worker_type).set(bytes_used)
        logging.debug(f"[METRICS] Memory usage: {worker_type} - {bytes_used} bytes")

# Global metrics collector instance
metrics = MetricsCollector()

def get_metrics():
    """Get metrics in Prometheus format with multiprocess support"""
    registry = CollectorRegistry()
    MultiProcessCollector(registry)
    return generate_latest(registry)