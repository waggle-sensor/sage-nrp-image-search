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

### `nrp-dev/` (Default)
Development environment overlay that:
- Sets the name prefix (`dev-MYBENCHMARK-`)
- References the base deployment
- Applies patches for environment variables
- Defines image replacement

**Files:**
- `kustomization.yaml` - Main kustomize configuration
- `env.yaml` - Environment variables for dev environment

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name in both files
- Update image name in `kustomization.yaml`
- Update S3_PREFIX in `env.yaml` if needed

### `nrp-prod/` (Optional)
Production environment overlay that:
- Sets the name prefix (`prod-MYBENCHMARK-`)
- Extends the base overlay
- Patches service names for prod environment
- Can override S3 prefix for prod

**Files:**
- `kustomization.yaml` - Main kustomize configuration for prod
- `env.yaml` - Environment variables for prod environment

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name in both files
- Update image name and tag in `kustomization.yaml`
- Update S3_PREFIX in `env.yaml` for prod path

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

Also replace in both `nrp-dev/kustomization.yaml` and `nrp-prod/kustomization.yaml`:
- `namePrefix: dev-MYBENCHMARK-` → `namePrefix: dev-mybenchmark-`
- `namePrefix: prod-MYBENCHMARK-` → `namePrefix: prod-mybenchmark-`
- `benchmark: MYBENCHMARK` → `benchmark: mybenchmark`

### 3. Update Image Name

Edit both `nrp-dev/kustomization.yaml` and `nrp-prod/kustomization.yaml` and update the image name:

```yaml
images:
  - name: PLACEHOLDER_BENCHMARK_JOB_IMAGE
    newName: gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-MYBENCHMARK-job
    newTag: latest  # Use specific tag for prod (e.g., pr-1)
```

### 4. Customize Environment Variables

#### Dev Environment (`nrp-dev/env.yaml`)

Update environment variables for dev environment:

```yaml
env:
  # Vector DB configuration (Weaviate)
  - name: WEAVIATE_HOST
    value: "dev-weaviate.sage.svc.cluster.local"
  # Inference server configuration (Triton)
  - name: TRITON_HOST
    value: "dev-triton.sage.svc.cluster.local"
  # S3 upload configuration (override base defaults for this benchmark)
  - name: S3_PREFIX
    value: "dev-metrics/MYBENCHMARK"
  - name: LOG_LEVEL
    value: "DEBUG"
```

#### Prod Environment (`nrp-prod/env.yaml`)

Update environment variables for prod environment:

```yaml
env:
  # Vector DB configuration (Weaviate) - prod environment
  - name: WEAVIATE_HOST
    value: "prod-weaviate.sage.svc.cluster.local"
  # Inference server configuration (Triton) - prod environment
  - name: TRITON_HOST
    value: "prod-triton.sage.svc.cluster.local"
  # S3 upload configuration (override base defaults for this benchmark)
  - name: S3_PREFIX
    value: "prod-metrics/MYBENCHMARK"
  - name: LOG_LEVEL
    value: "INFO"
```

### 5. Update Makefile

Ensure your benchmark's Makefile uses conditional logic to select the correct kustomize directory:

```makefile
ENV ?= dev
ifeq ($(ENV),prod)
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-prod
else
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-dev
endif
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

The Makefile should use conditional logic to select the correct kustomize directory:

```makefile
ENV ?= dev
ifeq ($(ENV),prod)
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-prod
else
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-dev
endif
```

Then use:

```bash
make build     # Build Docker image
make run       # Deploy and run benchmark job (dev environment by default)
make run ENV=prod  # Deploy and run using prod environment resources
make logs      # View logs
make down      # Removes deployment
```

## Environment Switching (Dev/Prod)

Benchmarks can be deployed to use either **dev** or **prod** environment resources. The template includes both `nrp-dev/` and `nrp-prod/` overlays.

>NOTE: By default, the benchmark will use the dev environment resources (`nrp-dev/`).

### Using Environment Overlays

From the benchmark directory (e.g., `benchmarking/benchmarks/MYBENCHMARK/`):

```bash
# Run using default (dev environment) resources
make run

# Run using prod environment resources
make run ENV=prod
```

The `ENV` variable controls which kustomize overlay is used:
- `ENV=prod` → Uses `kubernetes/MYBENCHMARK/nrp-prod/`
- No `ENV` or `ENV=dev` → Uses `kubernetes/MYBENCHMARK/nrp-dev/`

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
