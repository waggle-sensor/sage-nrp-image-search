'''Prometheus Metrics Server for Weavloader'''

from flask import Flask, Response
from prometheus_client import CONTENT_TYPE_LATEST
from .metrics import get_metrics, metrics
import logging
import threading
import time
import psutil
import redis
from datetime import datetime
import requests
import re
import os
app = Flask(__name__)

CAPTION_PAUSE_KEY = "weavloader:caption_paused"
CAPTION_WAIT_KEY = "weavloader:caption_wait"
_PAUSE_TRUTHY = frozenset(("1", "true", "yes"))


def _is_caption_paused(r: redis.Redis) -> bool:
    """True when captioning is paused. Missing key means not paused."""
    val = r.get(CAPTION_PAUSE_KEY)
    if val is None:
        return False
    return str(val).strip().lower() in _PAUSE_TRUTHY


@app.route('/metrics')
def metrics_endpoint():
    """Unified Prometheus metrics endpoint for both custom and Flower metrics"""
    # Get custom weavloader metrics
    custom_metrics = get_metrics()
    
    # Get Flower metrics
    try:
        flower_metrics = requests.get('http://localhost:5555/metrics', timeout=5).text
        
        # Add weavloader_ prefix to all Flower metric names
        flower_metrics = re.sub(r'^([a-zA-Z_][a-zA-Z0-9_]*)', r'weavloader_\1', flower_metrics, flags=re.MULTILINE)
        
        # Combine both metrics
        combined_metrics = custom_metrics + b'\n' + flower_metrics.encode()
    except Exception as e:
        logging.warning(f"[METRICS] Could not fetch Flower metrics: {e}")
        # Return only custom metrics if Flower is unavailable
        combined_metrics = custom_metrics
    
    return Response(combined_metrics, mimetype=CONTENT_TYPE_LATEST)

@app.route('/health')
def health_endpoint():
    """Health check endpoint"""
    paused = False
    wait_size = 0
    try:
        redis_client = get_redis_client()
        paused = _is_caption_paused(redis_client)
        wait_size = int(redis_client.llen(CAPTION_WAIT_KEY) or 0)
    except Exception as e:
        logging.warning(f"[METRICS] Health could not read caption pause state: {e}")
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'caption_paused': paused,
        'caption_wait_size': wait_size,
    }

def count_dlq_records(r: redis.Redis):
    """Count the number of records in the DLQ"""
    total = 0
    for _ in r.scan_iter("dlq:*", count=1000):
        total += 1
    return total

def get_redis_client():
    """Get shared redis client for metrics server"""
    _redis_client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        db=int(os.environ.get("REDIS_DB", "0")),
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
    )
    return _redis_client

def collect_system_metrics():
    """Collect system metrics in background"""
    while True:
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            metrics.update_memory_usage('total_used', memory_info.used)
            
            # Process memory for current process
            process = psutil.Process()
            metrics.update_memory_usage('metric_server', process.memory_info().rss)
            
            # Redis connection health and queue sizes
            try:
                redis_client = get_redis_client()
                redis_client.ping()
                metrics.update_component_health('redis', True)
                
                # Get queue sizes
                image_queue_size = redis_client.llen('image_processing')
                monitor_queue_size = redis_client.llen('data_monitoring')
                cleanup_queue_size = redis_client.llen('cleanup')
                
                metrics.update_queue_size('image_processing', image_queue_size)
                metrics.update_queue_size('data_monitoring', monitor_queue_size)
                metrics.update_queue_size('cleanup', cleanup_queue_size)
                metrics.update_queue_size(
                    'caption_wait',
                    int(redis_client.llen(CAPTION_WAIT_KEY) or 0),
                )
                metrics.update_caption_paused(_is_caption_paused(redis_client))
                metrics.update_dlq_size(count_dlq_records(redis_client))
                
            except Exception as e:
                metrics.update_component_health('redis', False)
                logging.warning(f"[METRICS] Redis connection failed: {e}")
            
        except Exception as e:
            logging.error(f"[METRICS] Error collecting system metrics: {e}")
        
        time.sleep(30)  # Collect every 30 seconds

def start_metrics_server(port=8080):
    """Start the metrics server"""
    logging.info(f"[METRICS] Starting metrics server on port {port}")
    
    # Start background metrics collection
    metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
    metrics_thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=False)
