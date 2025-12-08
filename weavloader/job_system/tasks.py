'''Celery Tasks for Weavloader'''
import os
import traceback
import tritonclient.grpc as TritonClient
from client import initialize_weaviate_client
from processing import process_image, parse_deny_list
from metrics import metrics
import time
import psutil
from . import app, celery_logger
from celery import Task
import redis
import json
import uuid
import pandas as pd
import sage_data_client

# Get environment variables
USER = os.environ.get("SAGE_USER")
PASS = os.environ.get("SAGE_PASS")
UNALLOWED_NODES = os.environ.get("UNALLOWED_NODES", "")
UNALLOWED_NODES = parse_deny_list(UNALLOWED_NODES)
TRITON_HOST = os.environ.get("TRITON_HOST", "triton")
TRITON_PORT = os.environ.get("TRITON_PORT", "8001")
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "weaviate")
WEAVIATE_PORT = os.environ.get("WEAVIATE_PORT", "8080")
WEAVIATE_GRPC_PORT = os.environ.get("WEAVIATE_GRPC_PORT", "50051")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
DLQ_TTL_SECONDS = int(os.environ.get("DLQ_TTL_SECONDS", str(60*24*3600))) # default 60 days
DLQ_REPROCESS_MAX_PER_RUN = int(os.environ.get("DLQ_REPROCESS_MAX_PER_RUN", str(500))) # default 500 tasks
DLQ_MAX_REPROCESS_AGE = int(os.environ.get("DLQ_MAX_REPROCESS_AGE", str(50*24*3600))) # default 50 days

class DLQTask(Task):
    '''
    Dead Letter Queue Task class for handling failed tasks
    '''
    autoretry_for = (Exception,)
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    retry_kwargs = {'max_retries': 3, 'countdown': 60}

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        if kwargs.get('_dlq_attempt', 0) > 0:
            celery_logger.debug(f"[CLEANER] Skip re-forward {self.name} {task_id} (already DLQ-ed)")
            try:
                img = args[0] if args and isinstance(args[0], dict) else {}
                metrics.record_dlq_throw_away(img.get('vsn'), img.get('job'), img.get('task'), img.get('camera'))
            except Exception as e:
                celery_logger.error(f"[CLEANER] Error recording DLQ throw away: {e}")
                metrics.record_error("cleaner", type(e).__name__)
            return
        headers = {
            'task_id': task_id,
            'failed_task': self.name,
            'exc_type': exc.__class__.__name__,
            'exc_message': str(exc),
        }
        app.send_task(
            'job_system.tasks.handle_dlq',
            args=[self.name, args, kwargs, headers],
            queue='cleanup',
        )
        celery_logger.error(f"[CLEANER] Forwarded {self.name} {task_id} to DLQ: {headers}")

# Initialize shared clients (one per worker process)
_weaviate_client = None
_triton_client = None
_redis_client = None

def get_weaviate_client():
    """Get or create shared weaviate client"""
    global _weaviate_client
    if _weaviate_client is None:
        _weaviate_client = initialize_weaviate_client(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT)
        celery_logger.info("[SHARED] Initialized shared weaviate client")
    if _weaviate_client.is_ready():
        metrics.update_component_health('weaviate', True)
    else:
        metrics.update_component_health('weaviate', False)
    return _weaviate_client

def get_triton_client():
    """Get or create shared triton client"""
    global _triton_client
    if _triton_client is None:
        channel_args = [
            ("grpc.max_metadata_size", 32 * 1024),
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ]
        _triton_client = TritonClient.InferenceServerClient(
            url=f"{TRITON_HOST}:{TRITON_PORT}",
            channel_args=channel_args,
        )
        celery_logger.info("[SHARED] Initialized shared triton client")
    if _triton_client.is_server_ready():
        metrics.update_component_health('triton', True)
    else:
        metrics.update_component_health('triton', False)
    return _triton_client

def cleanup_clients():
    """Cleanup shared clients"""
    global _weaviate_client, _triton_client
    if _weaviate_client is not None:
        try:
            _weaviate_client.close()
            celery_logger.info("[SHARED] Closed shared weaviate client")
        except Exception as e:
            celery_logger.warning(f"[SHARED] Error closing weaviate client: {e}")
        _weaviate_client = None
    
    if _triton_client is not None:
        try:
            _triton_client.close()
            celery_logger.info("[SHARED] Closed shared triton client")
        except Exception as e:
            celery_logger.warning(f"[SHARED] Error closing triton client: {e}")
        _triton_client = None

def get_redis_client():
    """Get or create shared redis client for workers"""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        celery_logger.info("[SHARED] Initialized shared redis client")
    if _redis_client.ping():
        metrics.update_component_health('redis', True)
    else:
        metrics.update_component_health('redis', False)
    return _redis_client

@app.task(bind=True, base=DLQTask, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def process_image_task(self, image_data, **meta):
    """
    Process a single image with retry logic.
    
    Args:
        image_data (dict): Dictionary containing image metadata and URL
        
    Returns:
        dict: Processing result
    """
    task = self.name  # Get task name from Celery request
    dlq_attempt = meta.get('_dlq_attempt', 0)
    
    try:
        celery_logger.info(f"[PROCESSOR] Processing image: {image_data.get('url', 'unknown')}")
        
        # Get shared clients (reused across tasks)
        weaviate_client = get_weaviate_client()
        triton_client = get_triton_client()
        
        # Process the image
        result = process_image(
            image_data, 
            USER, 
            PASS, 
            weaviate_client, 
            triton_client,
            logger=celery_logger
        )
        
        # Record successful task
        process = psutil.Process()
        metrics.update_memory_usage('processor', process.memory_info().rss)
        if dlq_attempt > 0:
            metrics.record_dlq_reprocess("success")
        
        celery_logger.info(f"[PROCESSOR] Successfully processed image: {image_data.get('url', 'unknown')}")
        return result
        
    except Exception as exc:
        # Record failed task
        metrics.record_error("processor", type(exc).__name__)
        if dlq_attempt > 0:
            metrics.record_dlq_reprocess("failure")
        
        celery_logger.error(f"[PROCESSOR] Error processing image {image_data.get('url', 'unknown')}: {str(exc)}")
        celery_logger.error(f"[PROCESSOR] Traceback: {traceback.format_exc()}")
        
        # Retry with exponential backoff
        retry_delay = 60 * (2 ** self.request.retries)  # 60s, 120s, 240s
        metrics.record_task_retry(task, type(exc).__name__)
        celery_logger.warning(f"[PROCESSOR] Retrying in {retry_delay} seconds (attempt {self.request.retries + 1}/3)")
        
        raise self.retry(countdown=retry_delay, exc=exc)

@app.task
def monitor_data_stream():
    """
    Monitor the SAGE data stream and submit image processing tasks.
    """
    celery_logger.info("[MODERATOR] Starting data stream monitoring")
    
    # Redis key to store last processed timestamp
    LAST_TIMESTAMP_KEY = "weavloader:last_processed_timestamp"
    
    # Setup filter to query specific data
    filter_config = {
        # "plugin": "registry.sagecontinuum.org/yonghokim/imagesampler.*",
        "task": "imagesampler-.*"
    }
    
    try:
        # Get Redis client to store/retrieve last timestamp
        r = get_redis_client()
        
        # Get last processed timestamp from Redis, or use last 5 minutes if not set
        last_timestamp_str = r.get(LAST_TIMESTAMP_KEY)
        if last_timestamp_str:
            try:
                start = pd.Timestamp(last_timestamp_str)
                celery_logger.info(f"[MODERATOR] Resuming from timestamp: {start}")
            except Exception as e:
                celery_logger.warning(f"[MODERATOR] Failed to parse timestamp to resume from, using last 5 minutes: {e}")
                start = pd.Timestamp.utcnow() - pd.Timedelta(minutes=5)
        else:
            # First run - query from last 5 minutes (only new data going forward)
            start = pd.Timestamp.utcnow() - pd.Timedelta(minutes=5)
            celery_logger.info("[MODERATOR] First run, querying from last 5 minutes")
        
        # Query SAGE data since last timestamp, add 1 second to the last timestamp to avoid duplicates
        query_start = start + pd.Timedelta(seconds=1)
        df = sage_data_client.query(
            start=query_start,
            filter=filter_config
        )
        
        # Update component health
        metrics.update_component_health('sage', True)
        
        # Filter out nodes not allowed to be processed
        if len(df) > 0:
            df = df[~df['meta.vsn'].apply(lambda x: x.strip().lower() in UNALLOWED_NODES)]
        
        # If no new images found, update last processed timestamp to try again later and return
        if len(df) == 0:
            celery_logger.info("[MODERATOR] No new images found")
            r.set(LAST_TIMESTAMP_KEY, start.isoformat())
            metrics.update_sage_stream_health(True)
            return {"status": "success", "images_submitted": 0}
        
        # Process the dataframe
        vsns = df['meta.vsn'].unique()
        end_time = df.timestamp.max()
        start_time = df.timestamp.min()
        
        celery_logger.info(f'[MODERATOR] Processing {len(df)} images for nodes: {vsns}')
        celery_logger.info(f'[MODERATOR] Time range: {start_time} to {end_time}')
        
        # Submit each image as a separate task
        images_submitted = 0
        for i in df.index:
            image_data = {
                'url': df.value[i],
                'timestamp': df.timestamp[i].isoformat(),
                'vsn': df["meta.vsn"][i] if "meta.vsn" in df.columns else "unknown",
                'filename': df["meta.filename"][i] if "meta.filename" in df.columns else "unknown",
                'camera': df["meta.camera"][i] if "meta.camera" in df.columns else "unknown",
                'host': df["meta.host"][i] if "meta.host" in df.columns else "unknown",
                'job': df["meta.job"][i] if "meta.job" in df.columns else "unknown",
                'node': df["meta.node"][i] if "meta.node" in df.columns else "unknown",
                'plugin': df["meta.plugin"][i] if "meta.plugin" in df.columns else "unknown",
                'task': df["meta.task"][i] if "meta.task" in df.columns else "unknown",
                'zone': df["meta.zone"][i] if "meta.zone" in df.columns else "unknown",
            }
            
            # Record SAGE image received
            metrics.record_sage_image(
                image_data['vsn'], 
                image_data['job'], 
                image_data['task'], 
                image_data['camera']
            )
            
            # Submit task to Celery queue
            process_image_task.apply_async(args=[image_data], queue="image_processing")
            celery_logger.debug(f"[MODERATOR] Submitted image task: {image_data['url']}")
            images_submitted += 1
        
        # Update last processed timestamp to the maximum timestamp in this batch
        new_last_timestamp = df.timestamp.max()
        r.set(LAST_TIMESTAMP_KEY, new_last_timestamp.isoformat())
        celery_logger.debug(f"[MODERATOR] Updated last timestamp to: {new_last_timestamp}")
        
        metrics.update_sage_stream_health(True)
        process = psutil.Process()
        metrics.update_memory_usage('moderator', process.memory_info().rss)
        
        return {
            "status": "success",
            "images_submitted": images_submitted,
            "last_timestamp": new_last_timestamp.isoformat()
        }
        
    except Exception as e:
        metrics.update_sage_stream_health(False)
        metrics.update_component_health('sage', False)
        metrics.record_error("moderator", type(e).__name__)
        
        celery_logger.error(f"[MODERATOR] Error in data stream monitoring: {str(e)}")
        celery_logger.error(f"[MODERATOR] Traceback: {traceback.format_exc()}")
        raise

@app.task
def handle_dlq(failed_task_name, args, kwargs, headers):
    """
    Handle DLQ for failed tasks.
    """
    r = get_redis_client()
    record = {
        'failed_task': failed_task_name,
        'args': args,
        'kwargs': kwargs,
        'headers': headers,
        'archived_at': time.time(),
    }
    key = f"dlq:{uuid.uuid4().hex}"
    r.setex(key, DLQ_TTL_SECONDS, json.dumps(record, default=str))
    celery_logger.info(f"[CLEANER] Stored DLQ record {key} for {failed_task_name}")

@app.task
def process_dlq_message(failed_task_name, args, kwargs):
    """
    Process a DLQ message to be reprocessed as a "process image" task again.
    """
    kwargs = dict(kwargs or {})
    kwargs['_dlq_attempt'] = kwargs.get('_dlq_attempt', 0) + 1

    if failed_task_name == 'job_system.tasks.process_image_task':
        res = process_image_task.apply_async(
            args=args, kwargs=kwargs, queue='image_processing', countdown=30
        )
        celery_logger.info(f"[CLEANER] Requeued {failed_task_name} as {res.id}")
        return res.id

    celery_logger.warning(f"[CLEANER] Unknown task {failed_task_name}; not requeued")
    return None

@app.task
def process_dlq_tasks():
    """
    Process DLQ tasks to be processed as messages.
    """
    # Get DLQ keys
    r = get_redis_client()
    dlq_keys = []
    for k in r.scan_iter("dlq:*", count=1000):
        dlq_keys.append(k)
        if len(dlq_keys) >= DLQ_REPROCESS_MAX_PER_RUN:
            break
    if not dlq_keys:
        celery_logger.info("[CLEANER] No DLQ records to reprocess")
        return

    reprocessed = failed = skipped = 0
    now = time.time()
    for key in dlq_keys:
        try:
            # Get DLQ record
            raw = r.get(key)
            if not raw:
                skipped += 1
                continue
            rec = json.loads(raw)
            failed_task = rec.get('failed_task')
            args = rec.get('args', [])
            kwargs = rec.get('kwargs', {})
            
            # Skip very old DLQ entries
            if (now - float(rec.get('archived_at', now))) > DLQ_MAX_REPROCESS_AGE:
                celery_logger.warning(f"[CLEANER] Skipping very old DLQ {key}")
                r.delete(key)
                skipped += 1
                try:
                    img = args[0] if args and isinstance(args[0], dict) else {}
                    metrics.record_dlq_throw_away(img.get('vsn'), img.get('job'), img.get('task'), img.get('camera'))
                except Exception as e:
                    celery_logger.error(f"[CLEANER] Error recording DLQ throw away: {e}")
                    metrics.record_error("cleaner", type(e).__name__)
                    continue
                continue
            # Reprocess DLQ message
            res_id = process_dlq_message.apply_async(
                args=[failed_task, args, kwargs],
                queue='cleanup',
            ).id
            # Remove DLQ record
            r.delete(key)
            reprocessed += 1
            celery_logger.info(f"[CLEANER] Scheduled reprocess {res_id} and removed {key}")
        except Exception as e:
            failed += 1
            celery_logger.error(f"[CLEANER] Error reprocessing {key}: {e}")
            metrics.record_error("cleaner", type(e).__name__)

    # Update memory usage
    process = psutil.Process()
    metrics.update_memory_usage('cleaner', process.memory_info().rss)
    celery_logger.info(f"[CLEANER] Reprocess summary: {reprocessed} requeued, {failed} failed, {skipped} skipped")
