# Kubernetes Template for Benchmarks

This directory contains Kubernetes/kustomize templates for benchmark deployments.

## Quick Start

1. Copy this directory to your benchmark's kubernetes folder:
   ```bash
   cd benchmarking/kubernetes
   cp -r ../benchmarks/template/kubernetes MYBENCHMARK
   cd MYBENCHMARK
   ```

2. Replace `MYBENCHMARK` with your benchmark name in all files:
   ```bash
   # On macOS/Linux
   find . -type f -exec sed -i '' 's/MYBENCHMARK/mybenchmark/g' {} +
   find . -type f -exec sed -i '' 's/mybenchmark/mybenchmark/g' {} +
   ```

3. Update the image name in `kustomization.yaml`

4. Customize environment variables in `env.yaml`

## Files Overview

### `kustomization.yaml`
Main kustomize configuration that:
- Sets the name prefix and labels
- References the base deployment
- Applies patches for environment variables
- Defines image replacement

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name
- Update image name in the `images` section

### `env.yaml`
Environment variables for the benchmark job.

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name
- Update environment variable names and values
- Add/remove variables as needed

### `nrp-prod/` (Optional)
Production environment overlay that:
- Extends the base overlay
- Patches service names for prod environment
- Can override S3 prefix for prod

## Step-by-Step Setup

### 1. Copy Template

```bash
cd benchmarking/kubernetes
cp -r ../benchmarks/template/kubernetes MYBENCHMARK
cd MYBENCHMARK
```

### 2. Replace Placeholders

Replace `MYBENCHMARK` with your benchmark name (lowercase) in all files:

```bash
# Example: Replace MYBENCHMARK with "mybenchmark"
find . -type f -name "*.yaml" -exec sed -i '' 's/MYBENCHMARK/mybenchmark/g' {} +
```

Also replace in `kustomization.yaml`:
- `namePrefix: MYBENCHMARK-` → `namePrefix: mybenchmark-`
- `benchmark: MYBENCHMARK` → `benchmark: mybenchmark`

### 3. Update Image Name

Edit `kustomization.yaml` and update the image name:

```yaml
images:
  - name: PLACEHOLDER_BENCHMARK_JOB_IMAGE
    newName: gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-MYBENCHMARK-job
    newTag: latest
```

### 4. Customize Environment Variables

#### Job (`env.yaml`)

Update environment variables that your benchmark needs:

```yaml
env:
  # Vector DB configuration (Weaviate)
  - name: WEAVIATE_HOST
    value: "dev-weaviate.sage.svc.cluster.local"
  - name: WEAVIATE_PORT
    value: "8080"
  - name: WEAVIATE_GRPC_PORT
    value: "50051"
  # Inference server configuration (Triton)
  - name: TRITON_HOST
    value: "dev-triton.sage.svc.cluster.local"
  - name: TRITON_PORT
    value: "8001"
  # Benchmark-specific configuration
  - name: MYBENCHMARK_DATASET
    value: "your-dataset/name"
  - name: COLLECTION_NAME
    value: "MYBENCHMARK"
  - name: QUERY_METHOD
    value: "clip_hybrid_query"
  # S3 upload configuration (override base defaults if needed)
  - name: S3_PREFIX
    value: "dev-metrics/MYBENCHMARK"
```

### 5. Update Makefile

Ensure your benchmark's Makefile points to the correct kustomize directory:

```makefile
KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK
```

## Testing

After setting up, test the configuration:

```bash
# Preview the generated manifests
kubectl kustomize . | less

# Run benchmark (from benchmark directory)
make run
```

## Common Customizations

### Different Namespace

Update `kustomization.yaml`:

```yaml
namespace: your-namespace
```

### Additional Environment Variables

Add to `env.yaml`:

```yaml
env:
  - name: NEW_VARIABLE
    value: "value"
```

### S3 Configuration

S3 endpoint, bucket, and credentials are configured in the base. To override:

```yaml
env:
  - name: S3_PREFIX
    value: "custom-prefix/benchmark-name"
  - name: UPLOAD_TO_S3
    value: "true"  # Enable S3 upload
```

## Integration with Makefile

The Makefile should reference this directory:

```makefile
KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK
```

Then use:

```bash
make build     # Build Docker image
make run       # Deploy and run benchmark job
make logs      # View logs
make down      # Removes deployment
```

## Environment Switching (Dev/Prod)

Benchmarks can be deployed to use either **dev** or **prod** environment resources. Each benchmark can have a `nrp-prod/` overlay that patches service names to match the prod environment.
>NOTE: By default, the benchmark will use the dev environment resources.

### Using Environment Overlays

From the benchmark directory (e.g., `benchmarking/benchmarks/MYBENCHMARK/`):

```bash
# Run using prod environment resources
make run ENV=prod

# Run using default (dev environment) resources
make run
```

The `ENV` variable controls which kustomize overlay is used:
- `ENV=prod` → Uses `kubernetes/MYBENCHMARK/nrp-prod/`
- No `ENV` → Uses `kubernetes/MYBENCHMARK/` (base overlay using dev environment resources)

### Creating Production Overlay

1. Copy `env.yaml` to `nrp-prod/env.yaml`
2. Update service names to prod:
   ```yaml
   - name: WEAVIATE_HOST
     value: "prod-weaviate.sage.svc.cluster.local"
   - name: TRITON_HOST
     value: "prod-triton.sage.svc.cluster.local"
   ```
3. Update S3 prefix if needed:
   ```yaml
   - name: S3_PREFIX
     value: "prod-metrics/MYBENCHMARK"
   ```

## Troubleshooting

### Error: "no matches for kind"

Make sure you're referencing the base correctly:
```yaml
resources:
  - ../base
```

### Error: "image not found"

Check that image name in `kustomization.yaml` matches your registry and image name.

### Job not starting

Check logs:
```bash
make logs
```

## See Also

- `../../kubernetes/README.md` - Kubernetes overview
- `../../kubernetes/base/` - Base deployment definitions
- `../../../benchmarks/INQUIRE/` - Complete example
