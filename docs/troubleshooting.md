# Troubleshooting

Common issues and fixes for Sage Image Search deployments.

## Triton fails to load models

**Symptoms:** Triton container restarts, logs show OSErrors loading CLIP or Gemma weights, or models stuck in `LOADING` state.

**Causes:**
- Missing or invalid `HF_TOKEN`
- Network issues downloading from Hugging Face
- Insufficient disk space in the model volume

**Fixes:**

1. Verify `HF_TOKEN` is set and has access to the required models (see [Authentication](authentication.md)).
2. Check Triton logs: `docker compose logs triton` or `kubectl logs -l app=triton`.
3. Manually download models and copy into the container (see [Getting Started](getting-started.md#triton-model-download-workaround)):

```bash
source .env
cd triton && python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
huggingface-cli download --local-dir clip --revision "$CLIP_MODEL_VERSION" "$CLIP_HF_REPO"
huggingface-cli download --local-dir "$(basename "$GEMMA_MODEL_PATH")" --revision "$GEMMA_MODEL_VERSION" google/gemma-3-4b-it
docker cp clip/. sage-nrp-image-search-triton-1:/models/clip/
docker cp "$(basename "$GEMMA_MODEL_PATH")" sage-nrp-image-search-triton-1:/models/
```

---

## Gradio cannot connect to Weaviate

**Symptoms:** UI shows connection errors, logs repeat "Failed to connect to Weaviate".

**Causes:**
- Weaviate is still starting up
- Wrong host/port configuration
- Weaviate pod crashed

**Fixes:**

1. Wait — the Gradio app retries every 10 seconds until Weaviate is ready.
2. Verify Weaviate is running: `docker compose ps weaviate` or `kubectl get pods -l app=weaviate`.
3. Check environment variables: `WEAVIATE_HOST`, `WEAVIATE_PORT`, `WEAVIATE_GRPC_PORT`.
4. Port-forward if needed: `kubectl port-forward svc/dev-weaviate 8080:8080`.

---

## No search results

**Symptoms:** Queries return empty results or an empty metadata table.

**Causes:**
- Weavloader has not indexed any images yet
- All matching nodes are on the `UNALLOWED_NODES` deny list
- The Weaviate collection is empty
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

**Symptoms:** No new images appear in Weaviate; weavloader logs show no processing activity.

**Causes:**
- Invalid Sage credentials
- Redis connection failure
- Celery workers not running
- All nodes filtered by `UNALLOWED_NODES`

**Fixes:**

1. Verify Sage credentials: `SAGE_USER`, `SAGE_PASS`.
2. Check Redis: `redis-cli ping` should return `PONG`.
3. Check Celery worker status via Flower (port 5555) or logs.
4. Review the DLQ for failed tasks — see [weavloader/README.md](../weavloader/README.md#dead-letter-queue-dlq).

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

1. Ensure the main stack (Weaviate + Triton) is deployed and healthy before running benchmarks.
2. Check job logs: `make logs` from the benchmark directory.
3. See [benchmarking/kubernetes/README.md](../benchmarking/kubernetes/README.md) for deployment details.
4. For local debugging: `make run-local` port-forwards Weaviate and Triton.

---

## Docker Compose vs Kubernetes behavior differences

**Symptoms:** Search works on NRP but not locally (or vice versa), different result quality.

**Cause:** Local Compose uses `multi2vec-bind` vectorizer; NRP K8s uses user-provided CLIP vectors with a different query path.

**Fix:** This is expected. For production-like behavior, test against the NRP deployment. See [Architecture](architecture.md#docker-compose-vs-kubernetes).

---

## Getting help

- [weavloader/README.md](../weavloader/README.md) — ingestion, Celery, DLQ, metrics
- [kubernetes/README.md](../kubernetes/README.md) — secrets, overlays, port-forwarding
- [benchmarking/README.md](../benchmarking/README.md) — benchmark setup and evaluation
