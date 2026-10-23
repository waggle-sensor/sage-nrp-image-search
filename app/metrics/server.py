'''Prometheus metrics HTTP server for image-search queries.'''

import logging
import threading
import time
from datetime import datetime, timezone

import psutil
from flask import Flask, Response
from prometheus_client import CONTENT_TYPE_LATEST

from .metrics import get_metrics, metrics

app = Flask(__name__)
_health_clients = {'milvus': None, 'triton': None}


@app.route('/metrics')
def metrics_endpoint():
    return Response(get_metrics(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/health')
def health_endpoint():
    milvus_ok = _ping_milvus()
    triton_ok = _ping_triton()
    return {
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'milvus': milvus_ok,
        'triton': triton_ok,
    }


def _ping_milvus() -> bool:
    client = _health_clients.get('milvus')
    try:
        if client is None:
            metrics.update_component_health('milvus', False)
            return False
        client.list_collections()
        metrics.update_component_health('milvus', True)
        return True
    except Exception as e:
        metrics.update_component_health('milvus', False)
        logging.warning('[METRICS] Milvus health check failed: %s', e)
        return False


def _ping_triton() -> bool:
    client = _health_clients.get('triton')
    try:
        if client is None or not client.is_server_live():
            metrics.update_component_health('triton', False)
            return False
        metrics.update_component_health('triton', True)
        return True
    except Exception as e:
        metrics.update_component_health('triton', False)
        logging.warning('[METRICS] Triton health check failed: %s', e)
        return False


def collect_system_metrics():
    while True:
        try:
            memory_info = psutil.virtual_memory()
            metrics.update_memory_usage('total_used', memory_info.used)
            process = psutil.Process()
            metrics.update_memory_usage('gradio', process.memory_info().rss)
            _ping_milvus()
            _ping_triton()
        except Exception as e:
            logging.error('[METRICS] Error collecting system metrics: %s', e)
        time.sleep(30)


def set_health_clients(milvus_client=None, triton_client=None):
    '''Attach clients used by /health and the background collector.'''
    if milvus_client is not None:
        _health_clients['milvus'] = milvus_client
    if triton_client is not None:
        _health_clients['triton'] = triton_client


def start_metrics_server(port=8080, milvus_client=None, triton_client=None):
    '''Start /metrics and /health on a daemon thread (non-blocking).'''
    set_health_clients(milvus_client, triton_client)
    logging.info('[METRICS] Starting metrics server on port %s', port)

    threading.Thread(target=collect_system_metrics, daemon=True, name='imagesearch-metrics-collector').start()
    threading.Thread(
        target=lambda: app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        ),
        daemon=True,
        name='imagesearch-metrics-server',
    ).start()
