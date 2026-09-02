'''Redis helpers for pausing captioning during benchmarking.

While weavloader:caption_paused is set, SAGE ingest still collects image
metadata onto weavloader:caption_wait. No NRP (or Triton) caption calls
are made until the flag is cleared and drain_caption_wait moves items
onto the image_processing Celery queue.

Missing pause key means not paused (normal ingest).
'''

import json

PAUSE_KEY = "weavloader:caption_paused"
WAIT_KEY = "weavloader:caption_wait"

_PAUSE_TRUTHY = frozenset(("1", "true", "yes"))


def is_paused(r) -> bool:
    """Return True when captioning is paused. Missing key means not paused."""
    val = r.get(PAUSE_KEY)
    if val is None:
        return False
    return str(val).strip().lower() in _PAUSE_TRUTHY


def wait_length(r) -> int:
    """Return the number of parked image payloads on the wait list."""
    return int(r.llen(WAIT_KEY) or 0)


def enqueue_wait(r, image_data) -> None:
    """Park image metadata on the wait list (LPUSH; drain uses RPOP for FIFO)."""
    r.lpush(WAIT_KEY, json.dumps(image_data, default=str))


def drain_wait(r, limit: int) -> list:
    """RPOP up to `limit` image_data dicts. Stops if the pause flag is raised.

    Pause is checked before each RPOP so a flag raised mid-drain does not
    drop items. Callers should still no-op when is_paused is already True.
    """
    items = []
    for _ in range(max(0, int(limit))):
        if is_paused(r):
            break
        raw = r.rpop(WAIT_KEY)
        if raw is None:
            break
        try:
            items.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    return items
