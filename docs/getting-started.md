# Getting Started

This guide walks you through running Sage Image Search locally with Docker Compose or on NRP with Kubernetes.

## Prerequisites

- **Sage credentials** — `SAGE_USER` and `SAGE_PASS` with access to SAGE images
- **Hugging Face token** — `HF_TOKEN` with access to the configured models ([how to create a token](https://huggingface.co/docs/hub/en/security-tokens))
- **NRP API credentials** (Kubernetes only) — required when `LLM_RUN_MODE=NRP`

See [Authentication](authentication.md) for credential setup details.

---

## Local deployment (Docker Compose)

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set:

```
SAGE_USER=your_username
SAGE_PASS=your_password
HF_TOKEN=your_hf_token
```

### 2. Start the stack

```bash
docker compose up -d --build
```

### 3. Open the search UI

Navigate to [http://localhost:7860](http://localhost:7860).

### 4. Verify services

```bash
docker compose ps
docker compose logs weavloader   # check ingestion
docker compose logs triton       # check model loading
```

### Managing the stack

```bash
# Restart with rebuild
docker compose down && docker compose up -d --build

# Stop
docker compose down

# Stop and remove volumes
docker compose down --volumes
```

### Triton model download workaround

Triton may fail to load CLIP or Gemma model weights inside the container. If this happens, download models locally and copy them into the running container:

```bash
source .env   # assumes HF_TOKEN is set
cd triton
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
huggingface-cli download --local-dir clip --revision "$CLIP_MODEL_VERSION" "$CLIP_HF_REPO"
huggingface-cli download --local-dir "$(basename "$GEMMA_MODEL_PATH")" --revision "$GEMMA_MODEL_VERSION" google/gemma-3-4b-it

docker cp clip/. sage-nrp-image-search-triton-1:/models/clip/
docker cp "$(basename "$GEMMA_MODEL_PATH")" sage-nrp-image-search-triton-1:/models/
```

See [Troubleshooting](troubleshooting.md) for more details.

### Local limitations

- The Gradio UI container does not include Sage credentials by default, so image thumbnails may not display.
- Compose and Kubernetes both use NRP-managed Milvus at `milvus.nrp-nautilus.io:50051` — set `MILVUS_URI` / `MILVUS_TOKEN` in `.env`. See [Architecture](architecture.md).

---

## NRP deployment (Kubernetes)

### 1. Set up secrets

Create secret files from templates in `kubernetes/base/`. See [kubernetes/README.md](../kubernetes/README.md) for full instructions.

```bash
cp kubernetes/base/huggingface-secret.template.yaml kubernetes/base/._huggingface-secret.yaml
cp kubernetes/base/sage-user-secret.template.yaml kubernetes/base/._sage-user-secret.yaml
cp kubernetes/base/nrp-llm-user-secret.template.yaml kubernetes/base/._nrp-llm-user-secret.yaml
cp kubernetes/base/milvus-secret.template.yaml kubernetes/base/._milvus-secret.yaml
```

Fill `MILVUS_URI` / `MILVUS_TOKEN` in the milvus secret from [NRP vector-database docs](https://nrp.ai/documentation/userdocs/ai/vector-database/).

Fill in base64-encoded values for each secret.

### 2. Deploy

Tested with kubectl v1.29.1 and kustomize v5.0.4.

**Dev:**

```bash
kubectl apply -k kubernetes/nrp-dev
```

**Prod:**

```bash
kubectl apply -k kubernetes/nrp-prod
```

### 3. Access the UI

| Environment | URL |
|-------------|-----|
| Dev | [https://dev-sage-hybrid-search.nrp-nautilus.io](https://dev-sage-hybrid-search.nrp-nautilus.io) |
| Prod | [https://sage-hybrid-search.nrp-nautilus.io](https://sage-hybrid-search.nrp-nautilus.io) |

### 4. Monitor ingestion

Weavloader metrics and Flower are available via ingress:

| Environment | Metrics | Flower |
|-------------|---------|--------|
| Dev | [dev-weavloader-metrics.nrp-nautilus.io](https://dev-weavloader-metrics.nrp-nautilus.io) | port 5555 via port-forward |
| Prod | [weavloader-metrics.nrp-nautilus.io](https://weavloader-metrics.nrp-nautilus.io) | port 5555 via port-forward |

Port-forward for local debugging:

```bash
kubectl port-forward svc/dev-triton 8001:8001
kubectl port-forward svc/dev-gradio-ui 7860:7860
kubectl port-forward svc/dev-weavloader-metrics 8081:8080
```

### 5. Tear down

```bash
kubectl delete -k kubernetes/nrp-dev
# or
kubectl delete -k kubernetes/nrp-prod
```

### Debugging with rendered YAML

```bash
kubectl kustomize kubernetes/nrp-dev -o sage-image-search-dev.yaml
kubectl kustomize kubernetes/nrp-prod -o sage-image-search-prod.yaml
```

---

## Testing a pull request

The `kubernetes/prs/` overlay lets you deploy a PR-specific build. See [kubernetes/README.md](../kubernetes/README.md#testing-a-pull-request) for manual steps to configure the overlay and port-forward services.

---

## What happens after deployment

1. **Weavmanage** runs a one-shot Job to create the env-specific Milvus collection schema.
2. **Triton** downloads models and starts serving CLIP and Gemma.
3. **Weavloader** begins polling the SAGE data stream and indexing new images into Milvus.
4. **Gradio UI** connects to Milvus and Triton, ready for text queries (CLIP rerank via Triton).

Indexing is continuous. New images appear in search results after weavloader processes them (typically within the `MONITOR_DATA_STREAM_INTERVAL`, default 60 seconds).

---

## Next steps

- [Using the Search UI](using-the-search-ui.md) — run your first search
- [Configuration](configuration.md) — tune models, search parameters, and environment variables
- [weavloader/README.md](../weavloader/README.md) — ingestion, Celery workers, DLQ, metrics
- [kubernetes/README.md](../kubernetes/README.md) — secrets, overlays, PR testing
