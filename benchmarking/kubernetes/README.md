# Benchmarking Kubernetes Deployments

Kubernetes deployments for benchmarking using kustomize for configuration management.

## Structure

```
benchmarking/kubernetes/
├── base/                    # Base kustomization (shared across all benchmarks)
│   ├── kustomization.yaml
│   ├── benchmark-job.yaml   # Combined job (loads data and evaluates)
│   ├── s3-secret.template.yaml
│   ├── huggingface-secret.template.yaml
│   ├── nrp-secret.template.yaml
│   ├── milvus-secret.template.yaml
│   └── ._*-secret.yaml      # Local copies of secrets (gitignored)
│
└── INQUIRE/                 # INQUIRE benchmark overlay
    ├── nrp-dev/             # Dev environment overlay
    │   ├── kustomization.yaml
    │   └── env.yaml
    └── nrp-prod/            # Prod environment overlay
        ├── kustomization.yaml
        └── env.yaml
```

## Base Components

The `base/` directory contains generic resources that can be reused by any benchmark:

- **benchmark-job.yaml**: Job that runs the combined benchmark script (loads data and evaluates)
- **._s3-secret.yaml**: Secret for S3 credentials (access key and secret key)
- **._huggingface-secret.yaml**: Secret for HuggingFace token (for accessing private datasets)
- **._nrp-secret.yaml**: Secret for the NRP Envoy AI Gateway API key (`NRP_API_KEY`). Copy from `nrp-secret.template.yaml`. Required when `LLM_MODEL_PROVIDER=nrp`.
- **._milvus-secret.yaml**: Secret for NRP Milvus (`MILVUS_URI`, `MILVUS_TOKEN`). Copy from `milvus-secret.template.yaml`.

The job is **vector database and inference server agnostic**:
- Includes health checks and resource limits
- Includes base environment variables (PYTHONUNBUFFERED, PYTHONPATH)
- Includes S3 configuration (endpoint, bucket, secure flag) with defaults
- S3 credentials are loaded from the `s3-secret` secret
- HuggingFace token is loaded from the `huggingface-secret` secret (for accessing private datasets)
- NRP API key is loaded from the `nrp-secret` secret (for NRP caption models)
- Milvus URI/token are loaded from the `milvus-secret` secret
- Vector DB and inference server environment variables should be added via patches in benchmark-specific overlays (env.yaml)

## Creating a New Benchmark Overlay

To create a new benchmark (e.g., `MYBENCHMARK`):

1. **Create overlay directory**:
```bash
mkdir -p benchmarking/kubernetes/MYBENCHMARK
```

2. **Create kustomization.yaml**:
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: sage

namePrefix: mybenchmark-
commonLabels:
  benchmark: mybenchmark

resources:
  - ../base

patches:
  - path: env.yaml
    target:
      kind: Job
      labelSelector: "app=benchmark-job"

images:
  - name: PLACEHOLDER_BENCHMARK_JOB_IMAGE
    newName: gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-mybenchmark-job
    newTag: latest
```

3. **Update nrp-dev/env.yaml**:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: benchmark-job
spec:
  template:
    spec:
      containers:
        - name: benchmark-job
          env:
            # Vector DB configuration
            - name: WEAVIATE_HOST
              value: "dev-weaviate.sage.svc.cluster.local"
            - name: VECTOR_DB
              value: "milvus"
            - name: MILVUS_DB
              value: "image_search_svc"
            # Inference server configuration (Triton)
            - name: TRITON_HOST
              value: "dev-triton.sage.svc.cluster.local"
            # S3 upload configuration (override base defaults for this benchmark)
            - name: S3_PREFIX
              value: "dev-metrics/mybenchmark"
            - name: LOG_LEVEL
              value: "DEBUG"
```

4. **Update nrp-prod/** similarly with prod service names and S3 prefix.

## Environment Switching (Dev/Prod)

Benchmarks use separate overlays for dev and prod environments:
- **nrp-dev/**: Default development environment overlay
- **nrp-prod/**: Production environment overlay

>NOTE: By default, the benchmark will use the dev environment resources (`nrp-dev/`).

### Using Environment Overlays

From the benchmark directory (e.g., `benchmarking/benchmarks/INQUIRE/`):

```bash
# Run using prod environment resources
make run ENV=prod

# Run using default (dev environment) resources
make run
```

The `ENV` variable controls which kustomize overlay is used:
- `ENV=prod` → Uses `kubernetes/INQUIRE/nrp-prod/`
- No `ENV` or `ENV=dev` → Uses `kubernetes/INQUIRE/nrp-dev/`

## Usage

### Prerequisites

- `kubectl` configured with access to cluster
- `kustomize` (or `kubectl` with kustomize support)
- Images built and pushed to registry
- Secrets configured from the templates in `kubernetes/base/` (`._s3-secret.yaml`, `._huggingface-secret.yaml`, `._nrp-secret.yaml`, `._milvus-secret.yaml`)

### Run Benchmark

```bash
cd benchmarking/benchmarks/INQUIRE
make run              # Deploy and run using dev environment (default)
make run ENV=prod     # Deploy and run using prod environment resources
```

### Monitor

```bash
make status
make logs
```

### Cleanup

```bash
make down                # Remove deployments (dev environment)
make down ENV=prod       # Remove prod deployments
```

## Environment Variables

Benchmark-specific environment variables are set via patches in each overlay:

- **Job** (`env.yaml`): 
  - Vector DB connection (e.g., WEAVIATE_HOST, WEAVIATE_PORT)
  - Inference server connection (e.g., TRITON_HOST, TRITON_PORT)
  - Dataset name, collection name, query method, batch sizes
  - S3 prefix override (if different from base default)

### Base Environment Variables

The base `benchmark-job.yaml` includes:
- `S3_ENDPOINT`: S3 endpoint URL (override in env.yaml if needed)
- `S3_BUCKET`: S3 bucket name (override in env.yaml if needed)
- `S3_SECURE`: Use TLS for S3 (default: "true")
- `S3_PREFIX`: S3 prefix for uploaded files (default: "benchmark-results")
- `UPLOAD_TO_S3`: Enable S3 upload (default: "false")

Secrets are loaded from Kubernetes secrets:
- **S3 credentials** from `s3-secret`:
  - `S3_ACCESS_KEY`: From secret
  - `S3_SECRET_KEY`: From secret
- **HuggingFace token** from `huggingface-secret`:
  - `HF_TOKEN`: From secret (for accessing private datasets)
- **NRP API key** from `nrp-secret`:
  - `NRP_API_KEY`: From secret (required when `LLM_MODEL_PROVIDER=nrp`)
- **Milvus** from `milvus-secret`:
  - `MILVUS_URI`: From secret
  - `MILVUS_TOKEN`: From secret

## Secrets Configuration

### Setting Up S3 Secret

Create `kubernetes/base/._s3-secret.yaml` using the template file:
```bash
cp benchmarking/kubernetes/base/s3-secret.template.yaml benchmarking/kubernetes/base/._s3-secret.yaml
```

To generate base64 values:
```bash
echo -n "your-access-key" | base64
echo -n "your-secret-key" | base64
```

### Setting Up HuggingFace Secret

Create `kubernetes/base/._huggingface-secret.yaml` using the template file:
```bash
cp benchmarking/kubernetes/base/huggingface-secret.template.yaml benchmarking/kubernetes/base/._huggingface-secret.yaml
```

To generate base64 value for HuggingFace token:
```bash
echo -n "your-huggingface-token" | base64
```

### Setting Up NRP Secret

Create `kubernetes/base/._nrp-secret.yaml` using the template file:
```bash
cp benchmarking/kubernetes/base/nrp-secret.template.yaml benchmarking/kubernetes/base/._nrp-secret.yaml
```

To generate the base64 value for the NRP Envoy AI Gateway API key:
```bash
echo -n "your-nrp-api-key" | base64
```

This secret provides `NRP_API_KEY` to the benchmark job. It is required when captioning uses the NRP gateway (`LLM_MODEL_PROVIDER=nrp`).

### Setting Up Milvus Secret

Create `kubernetes/base/._milvus-secret.yaml` using the template file:
```bash
cp benchmarking/kubernetes/base/milvus-secret.template.yaml benchmarking/kubernetes/base/._milvus-secret.yaml
```

To generate base64 values:
```bash
echo -n "https://milvus.nrp-nautilus.io:50051" | base64
echo -n "username:password" | base64
```

> **Important:** 
> All secret files you actually use must be named with leading `._` per `.gitignore` and not checked into version control! Only commit the `*.template.yaml` files.

### Overriding S3 Configuration

To override base S3 settings for a specific benchmark, add to `env.yaml`:

```yaml
- name: S3_ENDPOINT
  value: "your-custom-endpoint:9000"
- name: S3_BUCKET
  value: "your-bucket"
- name: S3_PREFIX
  value: "custom-prefix/benchmark-name"
- name: UPLOAD_TO_S3
  value: "true"
```

## Image Registry

Images should be built and pushed to:
- `gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-{name}-job:latest`

Update the registry in `kustomization.yaml` if using a different registry.

## Local Development

For local development, use port-forwarding:

```bash
make run-local
```

This will:
1. Start port-forwarding for Triton (and Weaviate if `VECTOR_DB=weaviate`)
2. Run the benchmark locally against NRP Milvus by default (`VECTOR_DB=milvus`)
3. Stop port-forwarding when done

Results are saved locally in the current directory.
