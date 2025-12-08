'''Celery Configuration for Weavloader'''

import os
from kombu import Queue

# Task queues
task_queues = (
    Queue('data_monitoring'),
    Queue('image_processing'),
    Queue('cleanup')
)
task_routes = {
    'job_system.tasks.monitor_data_stream':  {'queue': 'data_monitoring'},
    'job_system.tasks.process_image_task':   {'queue': 'image_processing'},
    'job_system.tasks.process_dlq_tasks':  {'queue': 'cleanup'},
    'job_system.tasks.process_dlq_message': {'queue': 'cleanup'},
    'job_system.tasks.handle_dlq':            {'queue': 'cleanup'},
}

# Periodic tasks (Celery Beat)
monitor_interval = float(os.getenv('MONITOR_DATA_STREAM_INTERVAL', '60.0')) # in seconds
process_dlq_interval = float(os.getenv('PROCESS_DLQ_INTERVAL', '86400.0')) # Daily in seconds (86400 seconds = 24 hours)
beat_schedule = {
    'monitor-data-stream': {
        'task': 'job_system.tasks.monitor_data_stream',
        'schedule': monitor_interval,
        'options': {'queue': 'data_monitoring'},
    },
    'reprocess-dlq-tasks': {
        'task': 'job_system.tasks.process_dlq_tasks',
        'schedule': process_dlq_interval,  
        'options': {'queue': 'cleanup'},
    }
}

# Celery configuration
broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True
worker_cancel_long_running_tasks_on_connection_loss = True

# Retry settings
task_acks_late = True
worker_prefetch_multiplier = 1
task_reject_on_worker_lost = True
broker_connection_retry_on_startup = True
broker_connection_retry = True

# Retry configuration
task_default_retry_delay = 60  # 1 minute
task_max_retries = 3
task_retry_jitter = True
task_retry_backoff = True
task_retry_backoff_max = 600  # 10 minutes max delay

# Monitoring
worker_send_task_events = True
task_send_sent_event = True
