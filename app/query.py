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
    "location_lat",
    "location_lon",
]


class Milvus_query:
    """
    Query Milvus with hybrid dense + BM25 search and Triton CLIP image-text rerank.
    """

    def __init__(self, milvus_client, triton_client=None, collection_name=None):
        self.milvus_client = milvus_client
        self.triton_client = triton_client
        self.collection_name = collection_name or os.getenv(
            "MILVUS_COLLECTION", "HybridSearchExample"
        )
        self.sage_query = Sage_query()

    def clip_hybrid_query(self, nearText, collection_name=None):
        """
        Hybrid CLIP dense + BM25 sparse search, then Triton CLIP rerank
        (query text vs retrieved image), matching logits_per_image-style scoring.
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
        alpha = hp.query_alpha

        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
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
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(alpha, 1.0 - alpha),
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )

        hit_list = hits[0] if hits else []
        objects = []

        logging.debug("============clip_hybrid_query RESULTS==================")
        for hit in hit_list:
            entity = hit.get("entity", hit)
            caption = entity.get("caption", "") or ""
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
                "location_lat": float(entity.get("location_lat", 0.0) or 0.0),
                "location_lon": float(entity.get("location_lon", 0.0) or 0.0),
            })
            logging.debug("----------------%s----------------", objects[-1]["uuid"])
            logging.debug(f"Properties: {objects[-1]}")
            logging.debug(f"Score: {objects[-1]['score']}")

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
