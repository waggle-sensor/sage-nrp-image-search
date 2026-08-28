# Troubleshooting

Common issues and fixes for Sage Image Search deployments.

## Triton fails to load models

**Symptoms:** Triton container restarts, logs show OSErrors loading CLIP or Gemma weights, models stuck in `LOADING` state, or:

`Unrecognized processing class in /models/gemma-3-4b-it` / `failed to load all models`

**Causes:**
- Missing or invalid `HF_TOKEN` (or token without access to gated `google/gemma-3-4b-it`)
- Incomplete Gemma download left in the pod `emptyDir` (entrypoint used to skip whenever the directory was non-empty)
- Missing `processor_config.json` / `preprocessor_config.json` so `AutoProcessor` fails while CLIP still loads
- Network issues downloading from Hugging Face
- Insufficient disk / ephemeral-storage for model weights

**Fixes:**

1. Verify `HF_TOKEN` is set, can access `google/gemma-3-4b-it`, and you have accepted the model license on Hugging Face (see [Authentication](authentication.md)).
2. Check Triton logs: `docker compose logs triton` or `kubectl logs -l app=triton`.
3. On Kubernetes, **delete the Triton pod** so `emptyDir` model caches are cleared and `entrypoint.sh` re-downloads a complete snapshot (it now validates processor + weight files before skipping).
4. Manually download models and copy into the container (see [Getting Started](getting-started.md#triton-model-download-workaround)):

```bash
source .env
cd triton && python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
huggingface-cli download --local-dir clip --revision "$CLIP_MODEL_VERSION" "$CLIP_HF_REPO"
huggingface-cli download --local-dir "$(basename "$GEMMA_MODEL_PATH")" --revision "$GEMMA_MODEL_VERSION" google/gemma-3-4b-it
docker cp clip/. sage-nrp-image-search-triton-1:/models/clip/
docker cp "$(basename "$GEMMA_MODEL_PATH")" sage-nrp-image-search-triton-1:/models/
```

Triton is started with `--exit-on-error=false --strict-readiness=false` so CLIP can stay up even if Gemma fails; captioning on NRP (`LLM_RUN_MODE=NRP`) does not need local Gemma.
---

## Gradio cannot connect to Milvus

**Symptoms:** UI shows connection errors, logs repeat "Failed to connect to Milvus".

**Causes:**
- NRP Milvus endpoint unreachable
- Wrong `MILVUS_URI` / `MILVUS_TOKEN`
- Missing or invalid milvus secret on Kubernetes / missing token in `.env` for Compose

**Fixes:**

1. Wait — the Gradio app retries every 10 seconds until Milvus is ready.
2. Check environment variables: `MILVUS_URI`, `MILVUS_TOKEN`, `MILVUS_COLLECTION`.
3. Confirm NRP credentials from [vector-database docs](https://nrp.ai/documentation/userdocs/ai/vector-database/).

---

## No search results

**Symptoms:** Queries return empty results or an empty metadata table.

**Causes:**
- Weavloader has not indexed any images yet (fresh collections after Milvus cutover start empty)
- All matching nodes are on the `UNALLOWED_NODES` deny list
- The Milvus collection is empty or the wrong `MILVUS_COLLECTION` is configured
- Query is too specific or uses terms not in indexed data

**Fixes:**

1. Check weavloader is running and processing: `docker compose logs weavloader` or check Flower/metrics.
2. Verify images are being indexed — look for `process_image_task` success in weavloader logs.
3. Try a broad query like `W040` or `clouds`.
4. Check `UNALLOWED_NODES` in your environment — results from those VSNs are filtered at query time.
5. Confirm your Sage credentials have access to images (images without access are skipped at index time).

---

## Images not displaying in the gallery

**Symptoms:** Metadata table has results but the image gallery is empty.

**Causes:**
- Missing Sage credentials in the Gradio container (common in local Docker Compose)
- Image download from SAGE URL failed
- Network issues reaching SAGE object storage

**Fixes:**

1. In Kubernetes, verify the Sage user secret is mounted in the gradio-ui deployment.
2. In Docker Compose, add `SAGE_USER` and `SAGE_PASS` to the gradio-ui service in `docker-compose.yml`.
3. Check Gradio logs for download errors when fetching images from `link` URLs.

---

## Weavloader not ingesting images

**Symptoms:** No new images appear in Milvus; weavloader logs show no processing activity.

**Causes:**
- Invalid Sage credentials
- Redis connection failure
- Celery workers not running
- All nodes filtered by `UNALLOWED_NODES`
- Caption pause flag is set (`weavloader:caption_paused=1`) — ingest metadata queues, but no captions run

**Fixes:**

1. Verify Sage credentials: `SAGE_USER`, `SAGE_PASS`.
2. Check Redis: `redis-cli ping` should return `PONG`.
3. Check Celery worker status via Flower (port 5555) or logs.
4. Review the DLQ for failed tasks — see [weavloader/README.md](../weavloader/README.md#dead-letter-queue-dlq).
5. If you paused captioning for a bench, confirm `/health` (`caption_paused`, `caption_wait_size`) and clear the flag when done — see [Configuration → NRP fair use](configuration.md#nrp-fair-use-when-llm_run_modenrp).

For detailed weavloader troubleshooting (DLQ, queue lengths, scaling), see the [weavloader troubleshooting section](../weavloader/README.md#troubleshooting).

---

## Caption generation failures

**Symptoms:** Images indexed without captions, or weavloader tasks failing on the caption step.

**Causes:**
- `LLM_RUN_MODE=TRITON` but Gemma model not loaded in Triton
- `LLM_RUN_MODE=NRP` but missing `NRP_API_KEY` or `NRP_API_ENDPOINT`
- VLM timeout or OOM

**Fixes:**

1. Check which mode is active: `LLM_RUN_MODE` in your environment (TRITON for Compose, NRP for K8s).
2. For TRITON mode: verify Gemma is loaded in Triton (`kubectl logs -l app=triton`).
3. For NRP mode: verify NRP secrets are configured (see [Authentication](authentication.md)).

---

## Benchmark job failures

**Symptoms:** Kubernetes benchmark Job fails or produces no results.

**Fixes:**

1. Ensure the main stack (Milvus credentials + Triton) is deployed and healthy before running benchmarks.
2. Check job logs: `make logs` from the benchmark directory.
3. See [benchmarking/kubernetes/README.md](../benchmarking/kubernetes/README.md) for deployment details.
4. For local debugging: `make run-local` port-forwards Triton (and any still-Weaviate-based benchmark adapters as documented under `benchmarking/`).

---

## Docker Compose vs Kubernetes behavior differences

**Symptoms:** Search works on NRP but not locally (or vice versa), different result quality.

**Cause:** Caption backends may differ (`LLM_RUN_MODE=TRITON` vs `NRP`), or Compose/K8s use different `MILVUS_COLLECTION` values. Collections start empty until weavloader backfills.

**Fix:** Align `MILVUS_COLLECTION` and credentials with the target env. For production-like behavior, test against the NRP deployment. See [Architecture](architecture.md#docker-compose-vs-kubernetes).

---

## Getting help

- [weavloader/README.md](../weavloader/README.md) — ingestion, Celery, DLQ, metrics
- [kubernetes/README.md](../kubernetes/README.md) — secrets, overlays, port-forwarding
- [benchmarking/README.md](../benchmarking/README.md) — benchmark setup and evaluation
