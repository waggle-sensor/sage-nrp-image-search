# INQUIRE Benchmark Kubernetes Deployment

Kubernetes deployment for the INQUIRE benchmark using kustomize.

## Structure

This overlay extends `../base/` with INQUIRE-specific configuration:

- **env.yaml**: Environment variables for benchmark evaluator
- **data-loader-env.yaml**: Environment variables for data loader
- **gpus.yaml**: GPU configuration for data loader (requires GPU for model inference)
- **results-pvc.yaml**: Persistent volume claim for storing results
- **results-pvc-patch.yaml**: Patch to mount results PVC in evaluator

## Usage

### Prerequisites

- Kubernetes cluster with access to Weaviate and Triton services
- Images built and pushed to registry
- `kubectl` configured with appropriate context

### Deploy

```bash
cd benchmarking/benchmarks/INQUIRE
make deploy
```

### Load Data

```bash
make load
```

Monitor with:
```bash
make logs-data-loader
```

### Run Evaluation

```bash
make calculate
```

Monitor with:
```bash
make logs-evaluator
```

### Get Results

```bash
make get
```

Results are also stored in the `inquire-results-pvc` PVC.

### Status

```bash
make status
```

### Cleanup

```bash
make down      # Remove deployments
make clean     # Remove deployments and PVCs
```

## Environment Variables

### Evaluator
- `INQUIRE_DATASET`: HuggingFace dataset name
- `COLLECTION_NAME`: Weaviate collection name
- `QUERY_METHOD`: Query method to use
- `QUERY_BATCH_SIZE`: Batch size for parallel queries

### Data Loader
- `INQUIRE_DATASET`: HuggingFace dataset name
- `COLLECTION_NAME`: Weaviate collection name
- `IMAGE_BATCH_SIZE`: Batch size for processing
- `SAMPLE_SIZE`: Number of samples (0 = all)
- `WORKERS`: Number of parallel workers

## Image Registry

Images should be built and pushed to:
- `gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-inquire-evaluator:latest`
- `gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-inquire-data-loader:latest`

Update the registry in `kustomization.yaml` if using a different registry.

