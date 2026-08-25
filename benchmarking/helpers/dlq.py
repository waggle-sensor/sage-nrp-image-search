"""Dead-letter queue helpers for benchmark indexing failures."""

from __future__ import annotations

import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from imsearch_eval.framework.interfaces import DLQ_SOFT_KEY

DLQ_CSV_COLUMNS = [
    "item_id",
    "reason",
    "error",
    "attempts",
    "final_status",
    "last_error",
]


def soft_caption_dlq(
    item_id: Any = "",
    error: str = "empty caption from provider",
) -> dict:
    """Return a soft-DLQ sentinel (not inserted until retries succeed or force_insert)."""
    return {
        DLQ_SOFT_KEY: True,
        "reason": "caption_failed",
        "error": error,
        "item_id": "" if item_id is None else str(item_id),
    }


@dataclass
class DlqConfig:
    """DLQ retry / output settings from environment variables."""

    max_retries: int = 3
    retry_base_seconds: int = 60
    file_name: str = "dlq_records.csv"

    @classmethod
    def from_env(cls) -> "DlqConfig":
        return cls(
            max_retries=int(os.environ.get("DLQ_MAX_RETRIES", "3")),
            retry_base_seconds=int(os.environ.get("DLQ_RETRY_BASE_SECONDS", "60")),
            file_name=os.environ.get("DLQ_FILE", "dlq_records.csv"),
        )


@dataclass
class DlqTerminalRecord:
    """One terminal DLQ outcome for CSV export."""

    item_id: str
    reason: str
    error: str
    attempts: int
    final_status: str  # retried_ok | inserted_degraded | abandoned
    last_error: str = ""


def _is_soft_sentinel(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get(DLQ_SOFT_KEY))


def _is_success(result: Any) -> bool:
    return result is not None and not _is_soft_sentinel(result)


def retry_dlq_failures(
    data_loader,
    failures: List[Dict[str, Any]],
    on_success: Callable[[Dict[str, Any]], None],
    *,
    workers: int = 1,
    config: Optional[DlqConfig] = None,
) -> List[DlqTerminalRecord]:
    """
    Retry indexing DLQ entries with exponential backoff.

    Backoff matches production weavloader: ``base * 2**attempt``
    (default 60s, 120s, 240s).

    Soft caption failures that still fail after retries are force-inserted
    with an empty caption (``force_insert=True``). Hard failures that never
    succeed are abandoned (not inserted).
    """
    cfg = config or DlqConfig.from_env()
    if not failures:
        return []

    pending = [dict(f) for f in failures]
    terminal: List[DlqTerminalRecord] = []
    max_workers = max(1, workers)

    logging.info(
        "DLQ: retrying %s failed item(s) (max_retries=%s, base_delay=%ss)",
        len(pending),
        cfg.max_retries,
        cfg.retry_base_seconds,
    )

    for attempt in range(cfg.max_retries):
        if not pending:
            break
        delay = cfg.retry_base_seconds * (2**attempt)
        logging.warning(
            "DLQ: attempt %s/%s — waiting %ss before retrying %s item(s)",
            attempt + 1,
            cfg.max_retries,
            delay,
            len(pending),
        )
        time.sleep(delay)

        still_pending: List[Dict[str, Any]] = []

        def _retry_one(entry: Dict[str, Any]):
            item = entry["item"]
            result = data_loader.process_item(item, force_insert=False)
            return entry, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_retry_one, entry) for entry in pending]
            for future in as_completed(futures):
                entry, result = future.result()
                entry["attempt"] = attempt + 1
                if _is_success(result):
                    on_success(result)
                    terminal.append(
                        DlqTerminalRecord(
                            item_id=str(entry.get("item_id") or ""),
                            reason=str(entry.get("reason") or "hard_fail"),
                            error=str(entry.get("error") or ""),
                            attempts=attempt + 1,
                            final_status="retried_ok",
                            last_error="",
                        )
                    )
                elif _is_soft_sentinel(result):
                    entry["reason"] = result.get("reason", "caption_failed")
                    entry["error"] = result.get(
                        "error", "empty caption from provider"
                    )
                    if result.get("item_id"):
                        entry["item_id"] = str(result["item_id"])
                    still_pending.append(entry)
                else:
                    entry["reason"] = "hard_fail"
                    entry["error"] = (
                        str(result)
                        if result is not None
                        else "process_item returned None"
                    )
                    still_pending.append(entry)

        pending = still_pending

    # Exhausted retries
    for entry in pending:
        reason = str(entry.get("reason") or "hard_fail")
        last_error = str(entry.get("error") or "")
        attempts = int(entry.get("attempt") or cfg.max_retries)

        if reason == "caption_failed":
            try:
                result = data_loader.process_item(
                    entry["item"], force_insert=True
                )
            except Exception as exc:
                logging.error(
                    "DLQ: force_insert failed for %s: %s",
                    entry.get("item_id"),
                    exc,
                )
                terminal.append(
                    DlqTerminalRecord(
                        item_id=str(entry.get("item_id") or ""),
                        reason=reason,
                        error=last_error,
                        attempts=attempts,
                        final_status="abandoned",
                        last_error=str(exc),
                    )
                )
                continue

            if _is_success(result):
                on_success(result)
                terminal.append(
                    DlqTerminalRecord(
                        item_id=str(entry.get("item_id") or ""),
                        reason=reason,
                        error=last_error,
                        attempts=attempts,
                        final_status="inserted_degraded",
                        last_error=last_error,
                    )
                )
            else:
                terminal.append(
                    DlqTerminalRecord(
                        item_id=str(entry.get("item_id") or ""),
                        reason=reason,
                        error=last_error,
                        attempts=attempts,
                        final_status="abandoned",
                        last_error=last_error or "force_insert returned None",
                    )
                )
        else:
            terminal.append(
                DlqTerminalRecord(
                    item_id=str(entry.get("item_id") or ""),
                    reason=reason,
                    error=last_error,
                    attempts=attempts,
                    final_status="abandoned",
                    last_error=last_error,
                )
            )

    return terminal


def write_dlq_csv(
    path: Path | str,
    records: List[DlqTerminalRecord],
) -> Path:
    """Write DLQ terminal records to CSV (headers only when empty)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DLQ_CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "item_id": rec.item_id,
                    "reason": rec.reason,
                    "error": rec.error,
                    "attempts": rec.attempts,
                    "final_status": rec.final_status,
                    "last_error": rec.last_error,
                }
            )
    return out


def log_dlq_summary(records: List[DlqTerminalRecord]) -> None:
    """Log aggregate DLQ counts by reason and final_status."""
    if not records:
        logging.info("DLQ summary: no failed items")
        return

    by_status: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    for rec in records:
        by_status[rec.final_status] = by_status.get(rec.final_status, 0) + 1
        by_reason[rec.reason] = by_reason.get(rec.reason, 0) + 1

    logging.info(
        "DLQ summary: total=%s by_status=%s by_reason=%s",
        len(records),
        by_status,
        by_reason,
    )
