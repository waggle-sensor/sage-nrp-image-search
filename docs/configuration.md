# Configuration

This page consolidates the main environment variables and search tuning parameters. For per-environment deployment values, check the Kubernetes overlays in [`kubernetes/nrp-dev/`](../kubernetes/nrp-dev/) and [`kubernetes/nrp-prod/`](../kubernetes/nrp-prod/).

## Credentials

| Variable | Default | Component | Description |
|----------|---------|-----------|-------------|
| `SAGE_USER` | — | weavloader, Gradio (K8s) | Sage username |
| `SAGE_PASS` | — | weavloader, Gradio (K8s) | Sage password |
| `HF_TOKEN` | — | Triton, Weaviate | Hugging Face token for model downloads |
| `NRP_API_KEY` | — | weavloader | NRP AI Gateway API key |
| `NRP_API_ENDPOINT` | — | weavloader | NRP AI Gateway endpoint URL |
| `NRP_LLM_MODEL` | `gemma` (K8s) | weavloader | Model name for NRP captioning |

See [Authentication](authentication.md) for setup instructions.

## Weaviate

| Variable | Default | Description |
|----------|---------|-------------|
| `WEAVIATE_HOST` | `weaviate` | Weaviate hostname |
| `WEAVIATE_PORT` | `8080` | Weaviate REST port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |
| `RERANKER_INFERENCE_API` | `http://reranker-transformers:8080` | Reranker service URL |
| `QUERY_DEFAULTS_LIMIT` | `25` | Default query result limit |
| `DEFAULT_VECTORIZER_MODULE` | `multi2vec-bind` (Compose) | Vectorizer module (Compose only) |
| `ENABLE_MODULES` | `multi2vec-bind,reranker-transformers,backup-filesystem` | Enabled Weaviate modules |
| `ASYNC_INDEXING` | `true` | Async vector indexing |
| `USE_BLOCKMAX_WAND` | `true` | BM25 optimization |
| `USE_INVERTED_SEARCHABLE` | `true` | Inverted index for searchable properties |

Schema is managed by weavmanage migrations in [`weavmanage/migrations/`](../weavmanage/migrations/).

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
| `query_alpha` | `0.4` | Hybrid search blend: 0 = pure keyword, 1 = pure vector |
| `max_vector_distance` | `0.4` | Max accepted distance for vector search |
| `near_text_certainty` | `0.7` | Minimum similarity score (alternative to max_vector_distance) |
| `autocut_jumps` | `0` | Autocut limit (0 = disabled; use response_limit instead) |
| `concepts_to_avoid` | `["police", "gun"]` | Concepts to move away from in vector search |
| `avoid_concepts_force` | `0` | Strength of concept avoidance (0 = disabled) |
| `hybrid_weight` | `0.7` | Weight for hybrid component in colbert blend (experimental) |
| `colbert_weight` | `0.3` | Weight for colbert component in colbert blend (experimental) |

The active query path uses `clip_hybrid_query` with `query_alpha=0.4` and `response_limit=25`.

## Changing models

| What to change | Where |
|----------------|-------|
| CLIP model | [`triton/models/clip/`](../triton/models/clip/), [`kubernetes/base/triton.yaml`](../kubernetes/base/triton.yaml), `.env.example` |
| Caption VLM | [`triton/models/`](../triton/models/), [`triton/entrypoint.sh`](../triton/entrypoint.sh) |
| Caption prompt | [`weavloader/inference/model_config.py`](../weavloader/inference/model_config.py) |
| Reranker model | [`kubernetes/base/reranker-transformers.yaml`](../kubernetes/base/reranker-transformers.yaml) |
| NRP caption model | `NRP_LLM_MODEL` in K8s weavloader env overlay |

After changing models, run the [benchmarking suite](benchmarking.md) to check for regressions.

## Environment-specific overrides

Kubernetes overlays patch environment variables per deployment:

- Dev: [`kubernetes/nrp-dev/`](../kubernetes/nrp-dev/) — `gradio-env.yaml`, `weavloader-env.yaml`, `triton-env.yaml`
- Prod: [`kubernetes/nrp-prod/`](../kubernetes/nrp-prod/)

Image tags are pinned to the NRP GitLab registry (`latest` branch tag) in each overlay's `kustomization.yaml`.

## Next steps

- [Architecture](architecture.md) — how components connect
- [Troubleshooting](troubleshooting.md) — fix configuration issues
- [Benchmarking](benchmarking.md) — measure the impact of config changes
