'''Prometheus metrics for image-search query latency.'''

import logging
import time

from prometheus_client import Counter, Gauge, Histogram, generate_latest

QUERY_DURATION_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
QUERY_RESULT_BUCKETS = (1, 5, 10, 15, 20, 25, 50, 100)

query_duration_seconds = Histogram(
    'imagesearch_query_duration_seconds',
    'Time spent on a query stage',
    ['operation', 'status'],
    buckets=QUERY_DURATION_BUCKETS,
)

queries_total = Counter(
    'imagesearch_queries_total',
    'Total clip_hybrid_query completions',
    ['status'],
)

query_results = Histogram(
    'imagesearch_query_results',
    'Hits returned after node filter',
    buckets=QUERY_RESULT_BUCKETS,
)

query_empty_total = Counter(
    'imagesearch_query_empty_total',
    'Successful queries that returned no hits',
)

errors_total = Counter(
    'imagesearch_errors_total',
    'Total number of errors',
    ['component', 'error_type'],
)

component_health = Gauge(
    'imagesearch_component_health',
    'Component health status (1=healthy, 0=unhealthy)',
    ['component'],
)

memory_usage_bytes = Gauge(
    'imagesearch_memory_usage_bytes',
    'Memory usage in bytes',
    ['worker_type'],
)


class StageTimer:
    '''Context manager that observes query-stage latency.'''

    def __init__(self, operation: str, component: str = None):
        self.operation = operation
        self.component = component
        self.status = 'success'
        self._start = 0.0

    def fail(self, error_type: str = None):
        self.status = 'failure'
        if self.component and error_type:
            metrics.record_error(self.component, error_type)

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, _tb):
        if exc_type is not None:
            self.status = 'failure'
            if self.component:
                metrics.record_error(self.component, exc_type.__name__)
        metrics.record_query_stage(
            self.operation,
            self.status,
            time.perf_counter() - self._start,
        )
        return False


class MetricsCollector:
    '''Metrics collection for image-search queries.'''

    def record_query_stage(self, operation: str, status: str, duration: float):
        query_duration_seconds.labels(operation=operation, status=status).observe(duration)
        logging.debug(
            '[METRICS] Query stage: %s - %s - %.3fs',
            operation,
            status,
            duration,
        )

    def record_query(self, status: str):
        queries_total.labels(status=status).inc()
        logging.debug('[METRICS] Query: %s', status)

    def record_query_results(self, count: int):
        query_results.observe(count)
        logging.debug('[METRICS] Query results: %s', count)

    def record_query_empty(self):
        query_empty_total.inc()
        logging.debug('[METRICS] Query empty')

    def record_error(self, component: str, error_type: str):
        errors_total.labels(component=component, error_type=error_type).inc()
        logging.debug('[METRICS] Error: %s - %s', component, error_type)

    def update_component_health(self, component: str, healthy: bool):
        component_health.labels(component=component).set(1 if healthy else 0)
        logging.debug('[METRICS] Component health: %s - %s', component, healthy)

    def update_memory_usage(self, worker_type: str, bytes_used: int):
        memory_usage_bytes.labels(worker_type=worker_type).set(bytes_used)
        logging.debug('[METRICS] Memory usage: %s - %s bytes', worker_type, bytes_used)


metrics = MetricsCollector()


def get_metrics():
    '''Prometheus text exposition of the default registry.'''
    return generate_latest()
