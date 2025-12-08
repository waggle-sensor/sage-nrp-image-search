'''Job System package for Weavloader'''

from celery import Celery
from celery.utils.log import get_task_logger

# Initialize Celery app
app = Celery('weavloader')
app.config_from_object('job_system.celery_config')
celery_logger = get_task_logger(__name__)

# Import tasks after app initialization
from .tasks import process_image_task, monitor_data_stream, process_dlq_message, process_dlq_tasks, handle_dlq

__all__ = ['app', 'celery_logger', 'process_image_task', 'monitor_data_stream', 'process_dlq_message', 'process_dlq_tasks', 'handle_dlq']
