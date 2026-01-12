# INQUIRE Benchmark Kubernetes Deployment

Kubernetes deployment for the INQUIRE benchmark using kustomize.

## Structure

This overlay extends `../base/` with INQUIRE-specific configuration:

- **env.yaml**: Environment variables for benchmark job

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

### Run Benchmark Job

```bash
make run-job
```

Monitor with:
```bash
make logs
```

### Status

```bash
make status
```

### Cleanup

```bash
make down      # Remove deployments
make clean     # Remove all resources
```

## Environment Variables

### Job Configuration
- `INQUIRE_DATASET`: HuggingFace dataset name
- `COLLECTION_NAME`: Weaviate collection name
- `QUERY_METHOD`: Query method to use
- `QUERY_BATCH_SIZE`: Batch size for parallel queries
- `IMAGE_BATCH_SIZE`: Batch size for processing
- `SAMPLE_SIZE`: Number of samples (0 = all)
- `WORKERS`: Number of parallel workers
- `S3_PREFIX`: S3 prefix for uploaded results (dev: "dev-metrics/inquire", prod: "prod-metrics/inquire")

## Image Registry

Images should be built and pushed to:
- `gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-inquire-job:latest`

Update the registry in `kustomization.yaml` if using a different registry.
