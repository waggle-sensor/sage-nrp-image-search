# INQUIRE Benchmark Kubernetes Deployment

Kubernetes deployment for the INQUIRE benchmark using kustomize.

## Structure

This overlay extends `../base/` with INQUIRE-specific configuration:

- **env.yaml**: Environment variables for benchmark job

## Usage

### Prerequisites

- Kubernetes cluster with access to Triton and NRP Milvus (or in-cluster Weaviate if `VECTOR_DB=weaviate`)
- Images built and pushed to registry
- `kubectl` configured with appropriate context

### Run Benchmark

```bash
cd benchmarking/benchmarks/INQUIRE
make run        # Deploys and runs the benchmark job (dev environment by default)
make run ENV=prod  # Deploys and runs using prod environment resources
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
make down          # Remove deployments (dev environment)
make down ENV=prod # Remove prod deployments
```

## Environment Variables

### Job Configuration

The following environment variables are set in `nrp-dev/env.yaml` and `nrp-prod/env.yaml`:

**Vector DB Configuration:**
- `VECTOR_DB`: `milvus` (default) or `weaviate`
- `MILVUS_DB`: NRP Milvus database (default: `image_search_svc`)
- `MILVUS_URI` / `MILVUS_TOKEN`: loaded from `milvus-secret`
- `WEAVIATE_HOST`: Weaviate service host (used when `VECTOR_DB=weaviate`)

**Inference Server Configuration:**
- `TRITON_HOST`: Triton service host (dev: `dev-triton.sage.svc.cluster.local`, prod: `prod-triton.sage.svc.cluster.local`)

**Benchmark-Specific Configuration:**
- `INQUIRE_DATASET`: HuggingFace dataset name (default: `sagecontinuum/INQUIRE-Benchmark-small`)
- `COLLECTION_NAME`: collection name (default: `INQUIRE`)
- `QUERY_METHOD`: Query method (Milvus default: `clip_hybrid_query_dual_index`)
- `QUERY_BATCH_SIZE`: Batch size for parallel queries
- `IMAGE_BATCH_SIZE`: Batch size for processing images
- `SAMPLE_SIZE`: Number of samples (0 = all)
- `WORKERS`: Number of parallel workers
- `LOG_LEVEL`: Logging level (dev: `DEBUG`, prod: `INFO`)

**S3 Configuration:**
- `S3_PREFIX`: S3 prefix for uploaded results (dev: `dev-metrics/inquire`, prod: `prod-metrics/inquire`)

Additional environment variables (S3 endpoint, bucket, credentials, HuggingFace token) are configured in the base Kubernetes resources and loaded from secrets.

## Image Registry

Images should be built and pushed to:
- `gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-inquire-job:latest`

Update the registry in `kustomization.yaml` if using a different registry.
