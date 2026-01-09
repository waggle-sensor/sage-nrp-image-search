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

3. Update the image names in `kustomization.yaml`

4. Customize environment variables in `env.yaml` and `data-loader-env.yaml`

## Files Overview

### `kustomization.yaml`
Main kustomize configuration that:
- Sets the name prefix and labels
- References the base deployment
- Applies patches for environment variables and GPU config
- Defines image replacements

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name
- Update image names in the `images` section

### `env.yaml`
Environment variables for the benchmark evaluator.

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name
- Update environment variable names and values
- Add/remove variables as needed

### `data-loader-env.yaml`
Environment variables for the data loader.

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name
- Update environment variable names and values
- Add/remove variables as needed

### `results-pvc.yaml`
PersistentVolumeClaim for storing benchmark results.

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name
- Adjust storage size if needed

### `results-pvc-patch.yaml`
Patch to mount the results PVC in the evaluator deployment.

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name

### `gpus.yaml` (Optional)
GPU configuration for the data loader.

**Required changes:**
- Replace `MYBENCHMARK` with your benchmark name
- Adjust GPU requirements if needed
- Remove this file if GPUs are not required

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

### 3. Update Image Names

Edit `kustomization.yaml` and update the image names:

```yaml
images:
  - name: PLACEHOLDER_BENCHMARK_EVALUATOR_IMAGE
    newName: gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-MYBENCHMARK-evaluator
    newTag: latest
  - name: PLACEHOLDER_BENCHMARK_DATA_LOADER_IMAGE
    newName: gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search/benchmark-MYBENCHMARK-data-loader
    newTag: latest
```

### 4. Customize Environment Variables

#### Evaluator (`env.yaml`)

Update environment variables that your evaluator needs:

```yaml
env:
  - name: MYBENCHMARK_DATASET
    value: "your-dataset/name"
  - name: COLLECTION_NAME
    value: "MYBENCHMARK"
  - name: RESULTS_FILE
    value: "results.csv"
  - name: METRICS_FILE
    value: "metrics.csv"
```

#### Data Loader (`data-loader-env.yaml`)

Update environment variables that your data loader needs:

```yaml
env:
  - name: MYBENCHMARK_DATASET
    value: "your-dataset/name"
  - name: COLLECTION_NAME
    value: "MYBENCHMARK"
  - name: BATCH_SIZE
    value: "25"
  - name: WORKERS
    value: "5"
```

### 5. Update Makefile

Ensure your benchmark's Makefile points to the correct kustomize directory:

```makefile
KUSTOMIZE_DIR := ../kubernetes/MYBENCHMARK
```

## Testing

After setting up, test the configuration:

```bash
# Preview the generated manifests
kubectl kustomize . | less

# Deploy (from benchmark directory)
make deploy
```

## Common Customizations

### Different Namespace

Update `kustomization.yaml`:

```yaml
namespace: your-namespace
```

### Different Storage Class

Update `results-pvc.yaml`:

```yaml
storageClassName: your-storage-class
```

### No GPU Support

1. Delete `gpus.yaml`
2. Remove the GPU patch from `kustomization.yaml`

### Additional Environment Variables

Add to `env.yaml` or `data-loader-env.yaml`:

```yaml
env:
  - name: NEW_VARIABLE
    value: "value"
```

## Integration with Makefile

The Makefile should reference this directory:

```makefile
KUSTOMIZE_DIR := ../kubernetes/MYBENCHMARK
```

Then use:

```bash
make deploy    # Deploys using kustomize
make down      # Removes deployment
```

## Environment Switching (Dev/Prod)

Benchmarks can be deployed to use either **dev** or **prod** environment resources. Each benchmark can have a `nrp-prod/` overlay that patch service names and PVC references to match the prod environment.
>NOTE: By default, the benchmark will use the dev environment resources.

### Using Environment Overlays

From the benchmark directory (e.g., `benchmarking/benchmarks/MYBENCHMARK/`):

```bash
# Deploy to prod environment  
make deploy ENV=prod

# Deploy to default (dev environment)
make deploy
```

The `ENV` variable controls which kustomize overlay is used:
- `ENV=prod` → Uses `kubernetes/MYBENCHMARK/nrp-prod/`
- No `ENV` → Uses `kubernetes/MYBENCHMARK/` (base overlay using dev environment resources)

## Troubleshooting

### Error: "no matches for kind"

Make sure you're referencing the base correctly:
```yaml
resources:
  - ../base
```

### Error: "image not found"

Check that image names in `kustomization.yaml` match your registry and image names.

### PVC not mounting

Verify PVC is created before the deployment

## See Also

- `../../kubernetes/README.md` - Kubernetes overview
- `../../kubernetes/base/` - Base deployment definitions
- `../../../benchmarks/INQUIRE/` - Complete example

