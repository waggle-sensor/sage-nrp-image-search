'''This file contains code that adds data to Milvus using sage_data_client.
These images will be the ones with which the hybrid search will compare
the text query given by the user.'''

import os
import pandas as pd
import time
import sage_data_client
import requests
import logging
from PIL import Image
from io import BytesIO
from inference import get_clip_embeddings, run_nrp_model, run_triton_model
from urllib.parse import urljoin
from metrics import metrics
import numpy as np
from math import isfinite
from openai import OpenAI

MANIFEST_API = os.environ.get("MANIFEST_API", "https://auth.sagecontinuum.org/manifests/")
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "HybridSearchExample")
LLM_RUN_MODE = os.environ.get("LLM_RUN_MODE", "TRITON")
METRIC_REPORT_CAPTION_MODEL = f"{LLM_RUN_MODE}_unknown".lower()
if LLM_RUN_MODE == 'NRP':
    NRP_API_KEY = os.environ.get("NRP_API_KEY", "")
    NRP_API_ENDPOINT = os.environ.get("NRP_API_ENDPOINT", "")
    nrp_client = OpenAI(api_key = NRP_API_KEY,base_url = NRP_API_ENDPOINT)
    NRP_LLM_MODEL = os.environ.get("NRP_LLM_MODEL","gemma")
    METRIC_REPORT_CAPTION_MODEL = f"{LLM_RUN_MODE}_{NRP_LLM_MODEL}".lower()
elif LLM_RUN_MODE == 'TRITON':
    TRITON_LLM_MODEL = os.environ.get("TRITON_LLM_MODEL", "gemma3")
    METRIC_REPORT_CAPTION_MODEL = f"{LLM_RUN_MODE}_{TRITON_LLM_MODEL}".lower()

def watch(start=None, filter=None, logger=logging.getLogger(__name__)):
    """
    Watches for incoming data and yields dataframes as new data is available.
    Uses adaptive polling to minimize traffic:
    - Faster polling when data is found (burst detection)
    - Slower polling when no data (reduce idle traffic)
    """
    if start is None:
        start = pd.Timestamp.utcnow()
    
    # Configurable intervals (in seconds)
    min_interval = 15.0      # When data found
    max_interval = 120.0     # When no data
    current_interval = min_interval
    
    while True:
        try:
            df = sage_data_client.query(
                    start=start,
                    filter=filter
                )
            metrics.update_component_health('sage', True)
        except Exception as e:
            metrics.update_component_health('sage', False)
            logger.error(f"[PROCESSING] Error querying Sage data: {e}")
            current_interval = min(current_interval * 1.5, max_interval)
            logger.debug(f"[PROCESSING] Error querying Sage data, increasing interval to {current_interval:.1f}s")
            time.sleep(current_interval)
            continue
        
        if len(df) > 0:
            # Data found - use fast polling for burst detection
            start = df.timestamp.max()
            current_interval = min_interval
            logger.debug(f"[PROCESSING] Sage data found, resetting interval to {min_interval:.1f}s")
            yield df
        else:
            # No data - gradually increase interval to reduce idle traffic
            current_interval = min(current_interval * 1.5, max_interval)
            logger.debug(f"[PROCESSING] No data, increasing interval to {current_interval:.1f}s")

        time.sleep(current_interval)

def parse_deny_list(raw: str) -> set[str]:
    """
    Parse the deny list from the environment variable.
    Args:
        raw (str): The raw deny list string
        
    Returns:
        set[str]: The parsed deny list
    """
    return {x.strip().lower() for x in raw.split(",") if x.strip()}

def safe_coord(value, default=0.0, label="coord", logger=None):
    """
    Safe coordinate check. Treat None as <default>.
    Args:
        value (float): The value to check
        default (float): The default value to return if the value is None
        label (str): The label of the coordinate
        logger (logging.Logger): The logger to use
        
    Returns:
        float: The safe coordinate
    """
    try:
        v = float(value)
    except Exception:
        if logger:
            logger.warning(f"[PROCESSING] {label} not convertible to float: {value!r}, defaulting to {default}")
        return default

    if not isfinite(v):
        if logger:
            logger.warning(f"[PROCESSING] Non-finite {label}: {v!r}, defaulting to {default}")
        return default

    # clamp to valid ranges
    if label.startswith("lat") and not (-90 <= v <= 90):
        if logger:
            logger.warning(f"[PROCESSING] Out-of-range latitude {v!r}, defaulting to {default}")
        return default
    if label.startswith("lon") and not (-180 <= v <= 180):
        if logger:
            logger.warning(f"[PROCESSING] Out-of-range longitude {v!r}, defaulting to {default}")
        return default

    return v

def safe_str(value, default="unknown"):
    """
    Safe string check. Treat None as <default>.
    Args:
        value (str): The value to check
        default (str): The default value to return if the value is None
        
    Returns:
        str: The safe string
    """
    if value is None:
        return default
    return str(value)

def process_image(image_data, username, token, milvus_client, triton_client, logger=logging.getLogger(__name__)):
    """
    Process a single image and add it to Milvus.
    
    Args:
        image_data (dict): Dictionary containing image metadata
        username (str): SAGE username
        token (str): SAGE token
        milvus_client: MilvusClient instance
        triton_client: Triton client instance
        
    Returns:
        dict: Processing result
    """
    url = image_data['url']
    timestamp = pd.Timestamp(image_data['timestamp'])
    vsn = image_data['vsn']
    filename = image_data['filename']
    camera = image_data['camera']
    host = image_data['host']
    job = image_data['job']
    node = image_data['node']
    plugin = image_data['plugin']
    task = image_data['task']
    zone = image_data['zone']
    
    # Auth header for Sage
    auth = (username, token)
    
    logger.debug(f"[PROCESSING] Processing image: {vsn}, {timestamp}, {url}")
    
    try:
        # Get the image data
        response = requests.get(url, auth=auth)
        response.raise_for_status()
        image_content = response.content

        # Check if the response contains valid image data
        if not image_content:
            raise ValueError(f"Empty content received for URL: {url}")

        image_stream = BytesIO(image_content)
        image = Image.open(image_stream).convert("RGB")

        # Get the manifest
        response = requests.get(urljoin(MANIFEST_API, vsn.upper()))
        response.raise_for_status()
        manifest = response.json()

        # Extract fields from manifest
        project = manifest.get('project', 'unknown')
        address = manifest.get('address', 'unknown')
        lat = manifest.get('gps_lat', 0.0)
        lon = manifest.get('gps_lon', 0.0)

        # Get live lat & lon
        loc_df = sage_data_client.query(start="-5m", filter={"vsn": vsn.upper(), "name": "sys.gps.lat|sys.gps.lon"}, tail=1)
        if not loc_df.empty:
            lat = loc_df[loc_df['name'] == 'sys.gps.lat']['value'].values[0]
            lon = loc_df[loc_df['name'] == 'sys.gps.lon']['value'].values[0]

        # Generate caption
        start_time = time.perf_counter()
        try:
            if LLM_RUN_MODE == 'TRITON':
                caption = run_triton_model(triton_client, TRITON_LLM_MODEL, image)
            elif LLM_RUN_MODE == 'NRP':
                caption = run_nrp_model(nrp_client, image, NRP_LLM_MODEL)
            else:
                raise ValueError(f"Unsupported LLM mode: {LLM_RUN_MODE}")

            caption_duration = time.perf_counter() - start_time
            metrics.record_model_inference(METRIC_REPORT_CAPTION_MODEL, "caption", caption_duration, "success")
        except Exception as e:
            caption_duration = time.perf_counter() - start_time
            metrics.record_model_inference(METRIC_REPORT_CAPTION_MODEL, "caption", caption_duration, "failure")
            raise e

        # Generate clip embedding
        start_time = time.perf_counter()
        try:
            clip_embedding = get_clip_embeddings(triton_client, caption, image)
            embedding_duration = time.perf_counter() - start_time
            metrics.record_model_inference("clip", "embedding", embedding_duration, "success")
        except Exception as e:
            embedding_duration = time.perf_counter() - start_time
            metrics.record_model_inference("clip", "embedding", embedding_duration, "failure")
            raise e

        # finite checks
        lat_sanitized = safe_coord(lat, default=0.0, label="lat", logger=logger)
        lon_sanitized = safe_coord(lon, default=0.0, label="lon", logger=logger)
        if not np.all(np.isfinite(clip_embedding)):
            logger.error(f"[PROCESSING] Non-finite values in embedding vector for {url}")
            raise ValueError(f"Non-finite values in embedding vector for {url}")

        caption_s = safe_str(caption)
        camera_s = safe_str(camera)
        host_s = safe_str(host)
        job_s = safe_str(job)
        vsn_s = safe_str(vsn)
        plugin_s = safe_str(plugin)
        zone_s = safe_str(zone)
        project_s = safe_str(project)
        address_s = safe_str(address)

        # Same fields Weaviate used in query_properties for BM25
        search_text = (
            f"{caption_s} {camera_s} {host_s} {job_s} {vsn_s} "
            f"{plugin_s} {zone_s} {project_s} {address_s}"
        )

        vector = (
            clip_embedding.tolist()
            if hasattr(clip_embedding, "tolist")
            else list(clip_embedding)
        )

        row = {
            "vector": vector,
            "search_text": search_text[:65535],
            "filename": safe_str(filename),
            "timestamp": safe_str(timestamp.strftime('%y-%m-%d %H:%M Z')),
            "link": safe_str(url),
            "caption": caption_s[:65535],
            "camera": camera_s,
            "host": host_s,
            "job": job_s,
            "node": safe_str(node),
            "plugin": plugin_s,
            "task": safe_str(task),
            "vsn": vsn_s,
            "zone": zone_s,
            "project": project_s,
            "address": address_s,
            "location_lat": float(lat_sanitized),
            "location_lon": float(lon_sanitized),
        }

        # Insert into Milvus with metrics
        start_time = time.perf_counter()
        try:
            milvus_client.insert(
                collection_name=MILVUS_COLLECTION,
                data=[row],
            )
            insert_duration = time.perf_counter() - start_time
            metrics.record_milvus_operation("insert", "success", insert_duration)
        except Exception as e:
            insert_duration = time.perf_counter() - start_time
            metrics.record_milvus_operation("insert", "failure", insert_duration)
            raise e
        
        logger.debug(f'[PROCESSING] Image added: {url}')
        return {"status": "success", "url": url, "vsn": vsn}

    except requests.exceptions.HTTPError as e:
        raise e
    except requests.exceptions.RequestException as e:
        raise e
    except Exception as e:
        raise e
