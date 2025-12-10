# Benchmarking Kubernetes Deployments

Kubernetes deployments for benchmarking using kustomize for configuration management.

## Structure

```
benchmarking/kubernetes/
├── base/                    # Base kustomization (shared across all benchmarks)
│   ├── kustomization.yaml
│   ├── benchmark-evaluator.yaml
│   └── benchmark-data-loader.yaml
│
└── INQUIRE_WEAV/                 # INQUIRE_WEAV benchmark overlay
    ├── kustomization.yaml   # Extends base with INQUIRE_WEAV-specific config
    ├── env.yaml             # Environment variables for evaluator
    ├── data-loader-env.yaml # Environment variables for data loader
    ├── gpus.yaml            # GPU configuration for data loader
    ├── results-pvc.yaml     # PVC for storing results
    └── results-pvc-patch.yaml # Patch to mount results PVC
```

## Base Components

The `base/` directory contains generic deployments that can be reused by any benchmark:

- **benchmark-evaluator.yaml**: Deployment for running benchmark evaluations
- **benchmark-data-loader.yaml**: Deployment for loading data into vector databases

Both deployments:
- Reference Weaviate and Triton services via kustomize vars
- Mount HuggingFace cache PVC
- Include health checks and resource limits

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
  - results-pvc.yaml

patches:
  - path: env.yaml
    target:
      kind: Deployment
      labelSelector: "app=benchmark-evaluator"
  - path: data-loader-env.yaml
    target:
      kind: Deployment
      labelSelector: "app=benchmark-data-loader"

images:
  - name: PLACEHOLDER_BENCHMARK_EVALUATOR_IMAGE
    newName: gitlab-registry.nrp-nautilus.io/ndp/sage/hybrid-search/benchmark-mybenchmark-evaluator
    newTag: latest
  - name: PLACEHOLDER_BENCHMARK_DATA_LOADER_IMAGE
    newName: gitlab-registry.nrp-nautilus.io/ndp/sage/hybrid-search/benchmark-mybenchmark-data-loader
    newTag: latest
```

3. **Create environment patches**:
   - `env.yaml`: Benchmark-specific environment variables for evaluator
   - `data-loader-env.yaml`: Environment variables for data loader
   - `results-pvc.yaml`: PVC for storing results
   - `results-pvc-patch.yaml`: Patch to mount results PVC

4. **Update Makefile** in your benchmark directory to use the new overlay

## Usage

### Prerequisites

- `kubectl` configured with access to cluster
- `kustomize` (or `kubectl` with kustomize support)
- Images built and pushed to registry

### Deploy

```bash
cd benchmarking/benchmarks/INQUIRE_WEAV
make deploy
```

### Load Data

```bash
make load
```

### Run Evaluation

```bash
make calculate
```

### Get Results

```bash
make get
```

### Monitor

```bash
make status
make logs-evaluator
make logs-data-loader
```

### Cleanup

```bash
make down
make clean  # Also removes PVCs
```

## Environment Variables

Benchmark-specific environment variables are set via patches in each overlay:

- **Evaluator**: Dataset name, collection name, query method, batch sizes
- **Data Loader**: Dataset name, collection name, batch sizes, workers

## Image Building

Images should be built and pushed to the registry before deployment:

```bash
make build
docker push <registry>/benchmark-<name>-evaluator:latest
docker push <registry>/benchmark-<name>-data-loader:latest
```

## Dependencies

The benchmark deployments depend on:
- **Weaviate**: Vector database (from main `kubernetes/base/`)
- **Triton**: Inference server (from main `kubernetes/base/`)
- **HF PVC**: HuggingFace cache (from main `kubernetes/base/`)

These should be deployed separately using the main kubernetes configuration.

