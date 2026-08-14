'''Query helpers for Milvus hybrid search + Triton CLIP image-text rerank.'''
#NOTE: This will be deployed in our cloud under k8s namespace beehive-sage
#   most likely integrated with beehive-data-api. If we want to allow our
#   python client to use these queries, we will also have to update our
#   sage-data-client python lib to include these new queries.

import HyperParameters as hp
from model import get_clip_embeddings, clip_image_text_score
import logging
import requests
import os
import re
from PIL import Image
from io import BytesIO
import pandas as pd
from pymilvus import AnnSearchRequest, WeightedRanker

OUTPUT_FIELDS = [
    "filename",
    "caption",
    "vsn",
    "camera",
    "project",
    "timestamp",
    "link",
    "host",
    "job",
    "plugin",
    "task",
    "zone",
    "node",
    "address",
    "location",
]

# WKT POINT(lon lat) — x=longitude, y=latitude
_POINT_RE = re.compile(
    r"POINT\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)",
    re.IGNORECASE,
)


def parse_wkt_point(wkt) -> tuple[float, float]:
    """Return (lat, lon) from a Milvus GEOMETRY WKT POINT. Invalid → (0.0, 0.0)."""
    if not wkt:
        return 0.0, 0.0
    match = _POINT_RE.search(str(wkt))
    if not match:
        return 0.0, 0.0
    lon, lat = float(match.group(1)), float(match.group(2))
    return lat, lon


class Milvus_query:
    """
    Query Milvus with hybrid caption/image dense + BM25 search and Triton CLIP rerank.
    """

    def __init__(self, milvus_client, triton_client=None, collection_name=None):
        self.milvus_client = milvus_client
        self.triton_client = triton_client
        self.collection_name = collection_name or os.getenv(
            "MILVUS_COLLECTION", "SageImageSearch"
        )
        self.sage_query = Sage_query()

    def clip_hybrid_query(self, nearText, collection_name=None):
        """
        Hybrid CLIP caption_vector + image_vector + BM25 sparse search,
        then Triton CLIP rerank (query text vs retrieved image), matching
        logits_per_image-style scoring.

        Dense vs sparse uses ``query_alpha``. Within dense, ``clip_alpha``
        weights ``image_vector`` vs ``caption_vector``.
        """
        collection = collection_name or self.collection_name
        clip_embedding = get_clip_embeddings(self.triton_client, nearText)
        if clip_embedding is None:
            logging.error("Failed to get CLIP embedding for query")
            return pd.DataFrame()

        vector = (
            clip_embedding.tolist()
            if hasattr(clip_embedding, "tolist")
            else list(clip_embedding)
        )

        limit = hp.response_limit
        query_alpha = hp.query_alpha
        clip_alpha = hp.clip_alpha
        dense_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        image_req = AnnSearchRequest(
            data=[vector],
            anns_field="image_vector",
            param=dense_params,
            limit=limit,
        )
        caption_req = AnnSearchRequest(
            data=[vector],
            anns_field="caption_vector",
            param=dense_params,
            limit=limit,
        )
        sparse_req = AnnSearchRequest(
            data=[nearText],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=limit,
        )

        hits = self.milvus_client.hybrid_search(
            collection_name=collection,
            reqs=[image_req, caption_req, sparse_req],
            ranker=WeightedRanker(
                query_alpha * clip_alpha,
                query_alpha * (1.0 - clip_alpha),
                1.0 - query_alpha,
            ),
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )

        hit_list = hits[0] if hits else []
        objects = []

        logging.debug("============clip_hybrid_query RESULTS==================")
        for hit in hit_list:
            entity = hit.get("entity", hit)
            caption = entity.get("caption", "") or ""
            location = entity.get("location", "") or ""
            objects.append({
                "uuid": str(hit.get("id", "")),
                "filename": entity.get("filename", "") or "",
                "caption": caption,
                "score": float(hit.get("distance", 0.0) or 0.0),
                "explainScore": "",
                "vsn": entity.get("vsn", "") or "",
                "camera": entity.get("camera", "") or "",
                "project": entity.get("project", "") or "",
                "timestamp": entity.get("timestamp", "") or "",
                "link": entity.get("link", "") or "",
                "host": entity.get("host", "") or "",
                "job": entity.get("job", "") or "",
                "plugin": entity.get("plugin", "") or "",
                "task": entity.get("task", "") or "",
                "zone": entity.get("zone", "") or "",
                "node": entity.get("node", "") or "",
                "address": entity.get("address", "") or "",
                "location": location,
            })
            logging.debug("----------------%s----------------", objects[-1]["uuid"])
            logging.debug(f"Properties: {objects[-1]}")
            logging.debug(f"Score: {objects[-1]['score']}")

        if not objects:
            logging.debug("==============END========================")
            return pd.DataFrame()

        # Drop denied VSNs before rerank so we do not fetch images or call Triton
        # for nodes the caller cannot see. TODO: replace UNALLOWED_NODES with
        # per-user Sage authorization.
        allowed = [obj for obj in objects if self.sage_query.authorize(obj["vsn"])]
        skipped = len(objects) - len(allowed)
        if skipped:
            logging.debug(
                "Skipped %s unallowed-node hit(s) before rerank", skipped
            )
        objects = allowed
        if not objects:
            logging.debug("==============END========================")
            return pd.DataFrame()

        # Rerank: Triton CLIP query-text vs image (same idea as HF logits_per_image)
        for obj in objects:
            image = self.sage_query.getImage(obj["link"]) if obj["link"] else None
            if image is None:
                obj["rerank_score"] = 0.0
            else:
                obj["rerank_score"] = clip_image_text_score(
                    self.triton_client, nearText, image
                )
            logging.debug(
                f"Rerank Score for {obj['uuid']}: {obj['rerank_score']}"
            )

        objects.sort(key=lambda x: x["rerank_score"], reverse=True)
        logging.debug("==============END========================")

        return pd.DataFrame(objects)


class Sage_query:
    """
    This class is used to query Sage.
    """
    def __init__(self):
        return
    
    @staticmethod
    def _parse_deny_list(raw: str) -> set[str]:
        return {x.strip().lower() for x in raw.split(",") if x.strip()}

    def authorize(self, vsn: str, username: str = None, key: str = None) -> bool:
        """
        Default-allow. Deny only if VSN appears in UNALLOWED_NODES (case-insensitive).
        Returns False if vsn is empty/None.
        TODO: implement authorization logic using username and key from sage user
        """
        # in the meantime, we will use a static deny list from env var
        if not vsn:
            return False
        return vsn.strip().lower() not in self._parse_deny_list(os.getenv("UNALLOWED_NODES", ""))

    def getImage(self, url):
        '''
        Retrieve the Images from Sage
        '''
        #Creds
        USER = os.environ.get("SAGE_USER")
        PASS = os.environ.get("SAGE_PASS")

        # Auth header for Sage
        auth = (USER, PASS)

        try:
            # Get the image data
            response = requests.get(url, auth=auth)
            response.raise_for_status()  # Raise error for bad responses
            image_data = response.content

            # Convert the image data to a PIL Image
            image = Image.open(BytesIO(image_data))
            image = image.convert("RGB")  # Ensure it's in RGB mode if necessary

        except requests.exceptions.HTTPError as e:
            logging.debug(f"Image skipped, HTTPError for URL {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logging.debug(f"Image skipped, request failed for URL {url}: {e}")
            return None
        except Exception as e:
            logging.debug(f"Image skipped, an error occurred for URL {url}: {e}")
            return None

        return image
