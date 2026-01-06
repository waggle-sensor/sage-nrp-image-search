# Benchmarking Kubernetes Deployments

Kubernetes deployments for benchmarking using kustomize for configuration management.

## Structure

```
benchmarking/kubernetes/
├── base/                    # Base kustomization (shared across all benchmarks)
│   ├── kustomization.yaml
│   ├── benchmark-evaluator.yaml
│   ├── benchmark-data-loader.yaml
│   ├── hf_pvc.yaml
│   └── results-pvc.yaml
│
└── INQUIRE/                 # INQUIRE benchmark overlay
    ├── kustomization.yaml   # Extends base with INQUIRE-specific config
    ├── env.yaml             # Environment variables for evaluator
    ├── data-loader-env.yaml # Environment variables for data loader
    └── nrp-prod/            # Prod environment overlay
```

## Base Components

The `base/` directory contains generic deployments that can be reused by any benchmark:

- **benchmark-evaluator.yaml**: Deployment for running benchmark evaluations
- **benchmark-data-loader.yaml**: Deployment for loading data into vector databases
- **hf_pvc.yaml**: Persistent volume claim for HuggingFace cache
- **results-pvc.yaml**: Persistent volume claim for storing results

Both deployments are **vector database and inference server agnostic**:
- Include health checks and resource limits
- Only include generic environment variables (PYTHONUNBUFFERED, PYTHONPATH)
- Vector DB and inference server environment variables should be added via patches in benchmark-specific overlays (env.yaml and data-loader-env.yaml)

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
      kind: Deployment
      labelSelector: "app=benchmark-evaluator"
  - path: data-loader-env.yaml
    target:
      kind: Deployment
      labelSelector: "app=benchmark-data-loader"

images:
  - name: PLACEHOLDER_BENCHMARK_EVALUATOR_IMAGE
    newName: gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-mybenchmark-evaluator
    newTag: latest
  - name: PLACEHOLDER_BENCHMARK_DATA_LOADER_IMAGE
    newName: gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-mybenchmark-data-loader
    newTag: latest
```

3. **Create environment patches**:
   - `env.yaml`: Benchmark-specific environment variables for evaluator (including vector DB and inference server config)
   - `data-loader-env.yaml`: Environment variables for data loader (including vector DB and inference server config)

   Example `env.yaml` for Weaviate + Triton:
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: benchmark-evaluator
   spec:
     template:
       spec:
         containers:
           - name: benchmark-evaluator
             env:
               - name: WEAVIATE_HOST
                 value: "weaviate.sage.svc.cluster.local"
               - name: WEAVIATE_PORT
                 value: "8080"
               - name: WEAVIATE_GRPC_PORT
                 value: "50051"
               - name: TRITON_HOST
                 value: "triton.sage.svc.cluster.local"
               - name: TRITON_PORT
                 value: "8001"
               - name: COLLECTION_NAME
                 value: "MYBENCHMARK"
               - name: QUERY_METHOD
                 value: "clip_hybrid_query"
   ```

4. **Update Makefile** in your benchmark directory to use the new overlay

## Environment Switching (Dev/Prod)

Benchmarks can be deployed to use either **dev** or **prod** environment resources. Each benchmark can have a `nrp-prod/` overlay that patch service names and PVC references to match the prod environment.
>NOTE: By default, the benchmark will use the dev environment resources.

### Using Environment Overlays

From the benchmark directory (e.g., `benchmarking/benchmarks/INQUIRE/`):

```bash
# Deploy to prod environment  
make deploy ENV=prod

# Deploy to default (dev environment)
make deploy
```

The `ENV` variable controls which kustomize overlay is used:
- `ENV=prod` → Uses `kubernetes/INQUIRE/nrp-prod/`
- No `ENV` → Uses `kubernetes/INQUIRE/` (base overlay using dev environment resources)

## Usage

### Prerequisites

- `kubectl` configured with access to cluster
- `kustomize` (or `kubectl` with kustomize support)
- Images built and pushed to registry

### Deploy

```bash
cd benchmarking/benchmarks/INQUIRE
make deploy              # Default deployment (dev environment)
make deploy ENV=prod     # Deploy to prod environment
```

### Load Data

```bash
make load                # Default deployment (dev environment)
make load ENV=prod       # Load using prod resources
```

### Run Evaluation

```bash
make calculate           # Default deployment (dev environment)
make calculate ENV=prod  # Evaluate using prod resources
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
make down                # Default deployment (dev environment)
make down ENV=prod       # Clean up prod deployment
make clean               # Also removes PVCs
```

## Environment Variables

Benchmark-specific environment variables are set via patches in each overlay:

- **Evaluator** (`env.yaml`): 
  - Vector DB connection (e.g., WEAVIATE_HOST, WEAVIATE_PORT, or PINECONE_API_KEY, etc.)
  - Inference server connection (e.g., TRITON_HOST, TRITON_PORT, or OPENAI_API_KEY, etc.)
  - Dataset name, collection name, query method, batch sizes
- **Data Loader** (`data-loader-env.yaml`):
  - Vector DB connection (same as evaluator)
  - Inference server connection (same as evaluator)
  - Dataset name, collection name, batch sizes, workers

The base deployments are agnostic to the specific vector DB and inference server used. Each benchmark overlay should add the appropriate environment variables for its chosen stack.

## Image Building

Images should be built and pushed to the registry before deployment:

```bash
make build
docker push <registry>/benchmark-<name>-evaluator:latest
docker push <registry>/benchmark-<name>-data-loader:latest
```

## Dependencies

The benchmark deployments depend on:
- **Vector Database**: Any vector database service (Weaviate, Pinecone, Qdrant, etc.)
- **Inference Server**: Any inference server or model API (Triton, OpenAI, HuggingFace, etc.)
- **HF PVC**: HuggingFace cache (from main `kubernetes/base/`)

The base deployments are agnostic to the specific services used. Each benchmark overlay should:
1. Configure environment variables pointing to the vector DB and inference server services
2. Ensure the required services are deployed in the cluster
3. Use the appropriate service names/endpoints in the environment variable patches

For example, INQUIRE uses Weaviate and Triton, but other benchmarks could use different stacks.

