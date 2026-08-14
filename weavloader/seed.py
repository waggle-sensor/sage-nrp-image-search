#!/usr/bin/env python3
"""Seed Milvus from a portable Hugging Face init dataset (parquet).

Controlled by env:
  INIT_DATASET            HF dataset id (empty = skip)
  INIT_DATASET_REVISION   default main
  INIT_DATASET_BATCH_SIZE default 256
  HF_TOKEN                required for private datasets
  MILVUS_URI / MILVUS_TOKEN / MILVUS_DB / MILVUS_COLLECTION
  REDIS_HOST / REDIS_PORT / REDIS_DB  (distributed lock)

Idempotent: skips if collection already has entities.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("seed")


def _env_truthy_dataset() -> Optional[str]:
    raw = (os.getenv("INIT_DATASET") or "").strip()
    return raw or None


def _to_wkt_point(lon: float, lat: float) -> str:
    return f"POINT({float(lon)} {float(lat)})"


def _normalize_timestamp(ts: Any) -> str:
    """ISO-8601 UTC for Milvus TIMESTAMPTZ (Weaviate dump used ``YY-MM-DD HH:MM Z``)."""
    if ts is None:
        return ""
    s = str(ts).strip()
    if not s:
        return ""
    if s.endswith("+00:00"):
        s = s.replace("+00:00", "Z")
    import re
    m = re.match(r"^(\d{2})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?\s*Z?$", s)
    if m:
        yy, mo, dd, hh, mi, ss = m.groups()
        year = int(yy)
        year += 2000 if year < 100 else 0
        return f"{year:04d}-{mo}-{dd}T{hh}:{mi}:{ss or '00'}Z"
    if "T" not in s and " " in s and s.endswith("Z"):
        s = s.replace(" ", "T", 1)
    return s


def _vec(row: Dict[str, Any], key: str) -> List[float]:
    v = row[key]
    if hasattr(v, "tolist"):
        v = v.tolist()
    return [float(x) for x in v]


def row_to_milvus(row: Dict[str, Any]) -> Dict[str, Any]:
    lat = float(row.get("location_lat") or 0.0)
    lon = float(row.get("location_lon") or 0.0)
    caption = str(row.get("caption") or "")
    search_text = str(row.get("search_text") or "").strip()
    if not search_text:
        search_text = " ".join(
            str(row.get(k) or "")
            for k in (
                "caption",
                "camera",
                "host",
                "job",
                "vsn",
                "plugin",
                "zone",
                "project",
                "address",
            )
        ).strip()

    return {
        "caption_vector": _vec(row, "caption_vector"),
        "image_vector": _vec(row, "image_vector"),
        "search_text": search_text[:65535],
        "filename": str(row.get("filename") or ""),
        "timestamp": _normalize_timestamp(row.get("timestamp")),
        "link": str(row.get("link") or ""),
        "caption": caption[:65535],
        "camera": str(row.get("camera") or ""),
        "host": str(row.get("host") or ""),
        "job": str(row.get("job") or ""),
        "node": str(row.get("node") or ""),
        "plugin": str(row.get("plugin") or ""),
        "task": str(row.get("task") or ""),
        "vsn": str(row.get("vsn") or ""),
        "zone": str(row.get("zone") or ""),
        "project": str(row.get("project") or ""),
        "address": str(row.get("address") or ""),
        "location": _to_wkt_point(lon, lat),
    }


def _collection_num_entities(client, name: str) -> int:
    stats = client.get_collection_stats(collection_name=name)
    if isinstance(stats, dict):
        if "row_count" in stats:
            return int(stats["row_count"])
        for k, v in stats.items():
            if "row" in str(k).lower() or "count" in str(k).lower():
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
    return 0


def _redis_client():
    import redis

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    return redis.Redis(host=host, port=port, db=db)


def seed_collection() -> int:
    """Return 0 on success/skip, non-zero on hard failure."""
    dataset_id = _env_truthy_dataset()
    if not dataset_id:
        LOG.info("[SEED] INIT_DATASET unset; skipping Hub seed")
        return 0

    collection = os.getenv("MILVUS_COLLECTION", "SageImageSearch")
    revision = os.getenv("INIT_DATASET_REVISION", "main")
    batch_size = int(os.getenv("INIT_DATASET_BATCH_SIZE", "256"))
    lock_key = f"weavloader:init_dataset:{collection}"
    lock_ttl = int(os.getenv("INIT_DATASET_LOCK_TTL", str(6 * 3600)))

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        force=True,
    )

    # Wait for Redis (supervisord starts redis + seed in parallel)
    r = None
    for attempt in range(60):
        try:
            r = _redis_client()
            r.ping()
            break
        except Exception as e:
            LOG.warning("[SEED] Waiting for Redis (%s)...", e)
            time.sleep(2)
    if r is None:
        LOG.error("[SEED] Redis unavailable; cannot take seed lock")
        return 1

    acquired = r.set(lock_key, "1", nx=True, ex=lock_ttl)
    if not acquired:
        LOG.info("[SEED] Another process holds %s; skipping", lock_key)
        return 0

    try:
        from client import initialize_milvus_client

        milvus = initialize_milvus_client()
        # Wait for collection (weavmanage may still be running)
        for attempt in range(90):
            if milvus.has_collection(collection):
                break
            LOG.warning(
                "[SEED] Collection %s missing (attempt %d/90); waiting for weavmanage...",
                collection,
                attempt + 1,
            )
            time.sleep(10)
        else:
            LOG.error(
                "[SEED] Collection %s still missing after wait; not creating schema here",
                collection,
            )
            return 2

        n = _collection_num_entities(milvus, collection)
        if n > 0:
            LOG.info(
                "[SEED] Collection %s already has %d entities; skipping seed",
                collection,
                n,
            )
            return 0

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            LOG.error("[SEED] HF_TOKEN required to load private dataset %s", dataset_id)
            return 3

        from datasets import load_dataset

        LOG.info(
            "[SEED] Streaming %s@%s into %s (batch=%d)",
            dataset_id,
            revision,
            collection,
            batch_size,
        )
        ds = load_dataset(
            dataset_id,
            split="train",
            revision=revision,
            streaming=True,
            token=token,
        )

        batch: List[Dict[str, Any]] = []
        inserted = 0
        for row in ds:
            batch.append(row_to_milvus(dict(row)))
            if len(batch) >= batch_size:
                milvus.insert(collection_name=collection, data=batch)
                inserted += len(batch)
                LOG.info("[SEED] Inserted %d (total %d)", len(batch), inserted)
                # Refresh lock TTL while progress is made
                r.expire(lock_key, lock_ttl)
                batch = []

        if batch:
            milvus.insert(collection_name=collection, data=batch)
            inserted += len(batch)
            LOG.info("[SEED] Inserted final %d (total %d)", len(batch), inserted)

        LOG.info("[SEED] Done. inserted=%d into %s", inserted, collection)
        return 0
    except Exception:
        LOG.exception("[SEED] Failed")
        return 4
    finally:
        try:
            r.delete(lock_key)
        except Exception:
            pass


def main() -> int:
    return seed_collection()


if __name__ == "__main__":
    sys.exit(main())
