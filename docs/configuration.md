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
| `MILVUS_COLLECTION` | `HybridSearchExample` | Collection name; use `HybridSearchExampleDev` on nrp-dev |

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

## Weavloader (ingestion)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_RUN_MODE` | `TRITON` (Compose), `NRP` (K8s) | Caption backend: `TRITON` or `NRP` |
| `TRITON_LLM_MODEL` | `gemma3` | Triton model name for captioning |
| `MONITOR_DATA_STREAM_INTERVAL` | `60` | Seconds between SAGE stream polls |
| `MONITOR_DATA_STREAM_QUERY_DELAY_MINUTE` | `5` | Lookback window on first run (minutes) |
| `UNALLOWED_NODES` | (long list) | Comma-separated VSN deny list |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery Redis broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` | Celery result backend |
| `LOG_LEVEL` | `INFO` | Logging level |

Caption prompt and VLM tuning: [`weavloader/inference/model_config.py`](../weavloader/inference/model_config.py)

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
| `query_alpha` | `0.4` | Hybrid blend for `WeightedRanker(alpha, 1-alpha)`: 0 = pure BM25, 1 = pure vector |
| `clip_alpha` | `0.7` | Image vs caption fusion weight at embed time |

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

- Dev: [`kubernetes/nrp-dev/`](../kubernetes/nrp-dev/) — `MILVUS_COLLECTION=HybridSearchExampleDev`
- Prod: [`kubernetes/nrp-prod/`](../kubernetes/nrp-prod/) — `MILVUS_COLLECTION=HybridSearchExample`

Milvus URI/token come from `milvus-secret` (`kubernetes/base/milvus-secret.template.yaml`).

Image tags are pinned to the NRP GitLab registry (`latest` branch tag) in each overlay's `kustomization.yaml`.

## Next steps

- [Architecture](architecture.md) — how components connect
- [Troubleshooting](troubleshooting.md) — fix configuration issues
- [Benchmarking](benchmarking.md) — measure the impact of config changes
