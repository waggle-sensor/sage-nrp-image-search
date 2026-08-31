# Configuration

This page consolidates the main environment variables and search tuning parameters. For per-environment deployment values, check the Kubernetes overlays in [`kubernetes/nrp-dev/`](../kubernetes/nrp-dev/) and [`kubernetes/nrp-prod/`](../kubernetes/nrp-prod/).

## Credentials

| Variable | Default | Component | Description |
|----------|---------|-----------|-------------|
| `SAGE_USER` | — | weavloader, Gradio | Sage username |
| `SAGE_PASS` | — | weavloader, Gradio | Sage password |
| `HF_TOKEN` | — | Triton | Hugging Face token for model downloads ([create a token](https://huggingface.co/docs/hub/en/security-tokens)) |
| `NRP_API_KEY` | — | weavloader | NRP AI Gateway API key |
| `NRP_API_ENDPOINT` | — | weavloader | NRP AI Gateway endpoint URL |
| `NRP_LLM_MODEL` | `gemma` (K8s) | weavloader | Model name for NRP captioning |
| `MILVUS_TOKEN` | — | app, weavloader, weavmanage | Milvus auth (`user:password`). Required for NRP Milvus: [vector-database docs](https://nrp.ai/documentation/userdocs/ai/vector-database/) |

See [Authentication](authentication.md) for setup instructions.

## Milvus

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_URI` | `https://milvus.nrp-nautilus.io:50051` | NRP-managed Milvus gRPC endpoint (use `https://` — TLS required, same as Attu “secure gRPC”) |
| `MILVUS_TOKEN` | — | Auth token `user:password` |
| `MILVUS_DB` | `image_search_svc` | NRP-provisioned database (do not create) |
| `MILVUS_COLLECTION` | `SageImageSearch` | Collection name; use `SageImageSearchDev` on nrp-dev |

### Optional Hub seed (weavloader)

| Variable | Default | Description |
|----------|---------|-------------|
| `INIT_DATASET` | empty (off) | Hugging Face dataset id to bulk-load into an **empty** collection, e.g. `sagecontinuum/init_img_search`. Unset to skip. |
| `INIT_DATASET_REVISION` | `main` | Hub revision/tag |
| `INIT_DATASET_BATCH_SIZE` | `256` | Rows per Milvus insert batch |
| `HF_TOKEN` | — | Required for the private init dataset (weavloader reads `huggingface-secret` on Kubernetes) |

Seeding runs once at weavloader startup via a supervisord oneshot (`seed.py`). It is idempotent: if the collection already has entities, seed is skipped. Do **not** enable on prod by default; nrp-dev sets `INIT_DATASET=sagecontinuum/init_img_search`.

Schema is managed by weavmanage migrations in [`weavmanage/migrations/`](../weavmanage/migrations/). Compose and Kubernetes both use NRP-managed Milvus (no Milvus service in this repo).

To inspect collections during development, use [Attu](https://milvus.io/docs/quickstart_with_attu.md). For pymilvus agent help, see [zilliztech/milvus-skill](https://github.com/zilliztech/milvus-skill). Details: [Authentication → Milvus development tools](authentication.md#milvus-development-tools).

## Triton

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_HOST` | `triton` | Triton hostname |
| `TRITON_PORT` | `8001` | Triton gRPC port |
| `MODEL_REPOSITORY` | `/app/models` | Triton model repository path |
| `CLIP_HF_REPO` | `apple/DFN5B-CLIP-ViT-H-14-378` | Hugging Face repo for CLIP |
| `CLIP_MODEL_PATH` | `/models/clip` | CLIP model directory |
| `CLIP_MODEL_VERSION` | `419d1f8f...` | CLIP model git revision |
| `GEMMA_MODEL_PATH` | `/models/gemma-3-4b-it` | Gemma VLM directory |
| `GEMMA_MODEL_VERSION` | `093f9f38...` | Gemma model git revision |

Active Triton models: `clip`, `gemma3`. Model download logic is in [`triton/entrypoint.sh`](../triton/entrypoint.sh).

Dynamic batching (NVIDIA default `dynamic_batching { }`, no preferred sizes, no queue delay):

| Model | `max_batch_size` | Notes |
|-------|------------------|-------|
| `clip` | 8 | Ragged `image` (`allow_ragged_batch`) so different `H×W` can share a GPU forward. Clients must send a leading batch dim (`text` `[1,1]`, `image` `[1,H,W,3]`). |
| `gemma3` | 2 | Same ragged-image rule; clients send `image` `[1,H,W,3]` and `prompt` `[1,1]`. |

Celery still submits one image per task. Concurrent weavloader workers are combined by Triton when CLIP is busy. Optional later: `max_queue_delay_microseconds` if batch-size histograms stay at 1 under load.

## Weavloader (ingestion)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_RUN_MODE` | `TRITON` (Compose), `NRP` (K8s) | Caption backend: `TRITON` or `NRP` |
| `NRP_LLM_MODEL` | `gemma` | NRP gateway model id when `LLM_RUN_MODE=NRP` (see [available models](https://nrp.ai/documentation/userdocs/ai/llm-managed/models/#gemma)) |
| `NRP_ENABLE_THINKING` | `false` | Keep `false` for caption latency; Gemma reasoning is on by default unless disabled |
| `LLM_IMAGE_BYTE_LIMITING` | `false` | When `true`, apply `LLM_MAX_IMAGE_SIDE`, downscale, and JPEG quality caps for caption LLMs |
| `LLM_MAX_IMAGE_BYTES` | `12582912` (12 MiB) | Max caption image payload when byte limiting is enabled |
| `LLM_MAX_IMAGE_SIDE` | `6144` | Longest-side pixel cap when byte limiting is enabled |
| `TRITON_LLM_MODEL` | `gemma3` | Triton model name for captioning |
| `MONITOR_DATA_STREAM_INTERVAL` | `60` | Seconds between SAGE stream polls |
| `MONITOR_DATA_STREAM_QUERY_DELAY_MINUTE` | `5` | Lookback window on first run (minutes) |
| `UNALLOWED_NODES` | (long list) | Comma-separated VSN deny list |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery Redis broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` | Celery result backend |
| `CAPTION_WAIT_DRAIN_INTERVAL` | `15` | Seconds between caption wait-queue drain ticks |
| `CAPTION_WAIT_DRAIN_BATCH` | `200` | Max wait-list items moved onto `image_processing` per drain tick |
| `CAPTION_PROMPT_ID` | `scientific_two_captions_v1` | Prompt catalog id from [`prompts/`](../prompts/). Shared with the benchmarking suite. |
| `CAPTION_MODEL_PROMPT` | unset | Raw prompt override; if set, `CAPTION_PROMPT_ID` is ignored |
| `LOG_LEVEL` | `INFO` | Logging level |
| `INIT_DATASET` | empty | Optional Hub dataset id for empty-collection seed (see Milvus section) |

### NRP fair use (when `LLM_RUN_MODE=NRP`)

Caption calls go through the [NRP managed LLM](https://nrp.ai/documentation/userdocs/ai/llm-managed/fair-use/) gateway and must stay within per-user limits:

| Limit | Guidance for weavloader |
|-------|-------------------------|
| Max concurrent requests | For `gemma` / `gemma-small`: **8** per user. Processor Celery concurrency is **6** (`weavloader/main.py`) with **1** replica by default → **6 ≤ 8**. |
| Combined context | Concurrent requests together should stay under **35%** of model context (~92k tokens for gemma’s 262k window). Single-image captions are far below this; do not treat large camera JPEGs as “millions of tokens” — Gemma 4 uses a soft-token budget (default **280** vision tokens/image). |
| Thinking / reasoning | Keep `NRP_ENABLE_THINKING=false` unless you need reasoning (adds latency and tokens). |
| Retries | Celery uses exponential backoff (60s → 120s → 240s), matching NRP’s advice to retry with increasing intervals. |

**Monitoring:** Check gateway health and usage analytics while tuning concurrency or debugging caption failures:

- [NRP LLM status](https://nrp.ai/llm-status/) — public endpoint availability
- [Envoy LLMs (Grafana)](https://grafana.nrp-nautilus.io/d/ad8bzhl/envoy-llms?from=now-1h&to=now&timezone=browser&var-team_id=$__all&var-model=$__all&var-token=Francisco) — request volume, latency, and errors by model

The Grafana dashboard is filtered by the API token used by Image Search (`var-token=Francisco`). Remove the **token** filter to see usage across all services on NRP’s hosted LLMs.

**Scaling rule:** total in-flight NRP captions ≈ `replicas × processor_concurrency`. Raising either so the product exceeds **8** (for gemma) violates fair use. The same NRP API key is shared across weavloader and any benchmark jobs — do not run both at high concurrency against NRP at once.

**Pause ingest during benches:** set Redis `weavloader:caption_paused` to `1` on each weavloader pod that uses this key (typically both `dev-weavloader` and `prod-weavloader`). The moderator still polls SAGE and checkpoints `weavloader:last_processed_timestamp`, but image metadata goes onto `weavloader:caption_wait` instead of calling NRP. Processor tasks already in `image_processing` are parked onto the same list. Clear the flag (`0` or delete the key) to resume; `drain_caption_wait` (every `CAPTION_WAIT_DRAIN_INTERVAL` seconds, batch `CAPTION_WAIT_DRAIN_BATCH`) moves wait items onto `image_processing`. CLIP and SAGE image download wait as well — only metadata is stored while paused.

```bash
kubectl -n sage exec deploy/prod-weavloader -- redis-cli SET weavloader:caption_paused 1
kubectl -n sage exec deploy/prod-weavloader -- redis-cli SET weavloader:caption_paused 0
```

Do not add unauthenticated HTTP pause endpoints — weavloader metrics is ingress-exposed. Inspect state via `/health` (`caption_paused`, `caption_wait_size`) or Prometheus `weavloader_caption_paused`.

Policy: [Fair use](https://nrp.ai/documentation/userdocs/ai/llm-managed/fair-use/) · Models: [gemma](https://nrp.ai/documentation/userdocs/ai/llm-managed/models/#gemma)

Caption prompt and VLM tuning: [`weavloader/inference/model_config.py`](../weavloader/inference/model_config.py)

Caption LLM image prep (NRP gateway, Triton gemma3/qwen2_5, benchmarks): [`weavloader/inference/image_utils.py`](../weavloader/inference/image_utils.py) and `imsearch_eval.framework.image_utils` — always converts to RGB; side cap and byte limits apply only when `LLM_IMAGE_BYTE_LIMITING=true`.

Supported models: [`weavloader/inference/model.py`](../weavloader/inference/model.py)

## Gradio UI

| Variable | Default | Description |
|----------|---------|-------------|
| `CLUSTER_FLAG` | `True` | K8s vs local deployment flag |

## Search hyperparameters

These are set in [`app/HyperParameters.py`](../app/HyperParameters.py) and are not environment-driven:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `response_limit` | `25` | Maximum number of results returned |
| `query_alpha` | `0.4` | Dense vs BM25: 0 = pure BM25, 1 = pure dense (`image_vector` + `caption_vector`) |
| `clip_alpha` | `0.7` | Within dense: 1 = `image_vector` only, 0 = `caption_vector` only. Combined as `WeightedRanker(query_alpha * clip_alpha, query_alpha * (1 - clip_alpha), 1 - query_alpha)` |

The active query path uses `clip_hybrid_query` with Milvus `hybrid_search` + Triton CLIP query–image rerank.

## Changing models

| What to change | Where |
|----------------|-------|
| CLIP model | [`triton/models/clip/`](../triton/models/clip/), [`kubernetes/base/triton.yaml`](../kubernetes/base/triton.yaml), `.env.example` |
| Caption VLM | [`triton/models/`](../triton/models/), [`triton/entrypoint.sh`](../triton/entrypoint.sh) |
| Caption prompt | [`weavloader/inference/model_config.py`](../weavloader/inference/model_config.py) |
| Reranker (CLIP) | Same Triton `clip` model as retrieval — [`app/query.py`](../app/query.py), [`triton/models/clip/`](../triton/models/clip/) |
| NRP caption model | `NRP_LLM_MODEL` in K8s weavloader env overlay |

After changing models, run the [benchmarking suite](benchmarking.md) to check for regressions.

## Environment-specific overrides

Kubernetes overlays patch environment variables per deployment:

- Dev: [`kubernetes/nrp-dev/`](../kubernetes/nrp-dev/) — `MILVUS_COLLECTION=SageImageSearchDev`
- Prod: [`kubernetes/nrp-prod/`](../kubernetes/nrp-prod/) — `MILVUS_COLLECTION=SageImageSearch`

Milvus URI/token come from `milvus-secret` (`kubernetes/base/milvus-secret.template.yaml`).

Image tags are pinned to the NRP GitLab registry (`latest` branch tag) in each overlay's `kustomization.yaml`.

## Next steps

- [Architecture](architecture.md) — how components connect
- [Troubleshooting](troubleshooting.md) — fix configuration issues
- [Benchmarking](benchmarking.md) — measure the impact of config changes
