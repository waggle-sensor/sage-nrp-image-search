# Benchmarking Makefile Guide

## Overview

The benchmarking framework provides a reusable Makefile system that allows any benchmark instance to leverage common build, deployment, and execution commands while customizing benchmark-specific settings.

## Structure

```
benchmarking/
├── Makefile              # Base Makefile with generic commands
benchmarks/
├── Makefile                  # Base Makefile (shared by all benchmarks)
├── MAKEFILE.md              # Makefile documentation
├── Dockerfile.template       # Base Dockerfile template
├── DOCKER.md                # Dockerfile documentation
    ├── INQUIRE/
│   └── Makefile             # INQUIRE-specific variables + includes base
    └── template/
    └── Makefile             # Template Makefile
```

## Base Makefile

The base `benchmarks/Makefile` contains all generic commands that work for any benchmark:

- **Build**: `make build` - Build Docker images
- **Deploy**: `make deploy` - Deploy to Kubernetes
- **Load**: `make load` - Start data loader
- **Calculate**: `make calculate` - Run benchmark evaluation
- **Get Results**: `make get` - Copy results from pod
- **Status**: `make status` - Show deployment status
- **Logs**: `make logs-evaluator` / `make logs-data-loader` - View logs
- **Clean**: `make clean` - Remove deployments and PVCs

## Creating a New Benchmark Makefile

To create a Makefile for a new benchmark (e.g., `MYBENCHMARK`):

### 1. Create the Makefile

Create `benchmarking/MYBENCHMARK/Makefile`:

```makefile
# MYBENCHMARK Benchmark Makefile
# This file sets MYBENCHMARK-specific variables and includes the base benchmarking Makefile

# ============================================================================
# Required Variables (must be set for base Makefile)
# ============================================================================
BENCHMARK_NAME := mybenchmark
KUSTOMIZE_DIR := ../kubernetes/MYBENCHMARK
DOCKERFILE_EVALUATOR := Dockerfile.benchmark
DOCKERFILE_DATA_LOADER := Dockerfile.data_loader
RESULTS_PVC_NAME := $(BENCHMARK_NAME)-benchmark-results-pvc
RESULTS_FILES := results.csv metrics.csv

# ============================================================================
# Optional Variables (can be overridden)
# ============================================================================
KUBECTL_NAMESPACE := sage
KUBECTL_CONTEXT ?= nrp-dev
REGISTRY := gitlab-registry.nrp-nautilus.io/ndp/sage/hybrid-search
EVALUATOR_TAG ?= latest
DATA_LOADER_TAG ?= latest

# ============================================================================
# MYBENCHMARK-Specific Environment Variables
# ============================================================================
# These are used by the Kubernetes deployments via environment variable patches
MYBENCHMARK_DATASET ?= mydataset/benchmark
BATCH_SIZE ?= 10

# Include the base Makefile (after setting variables)
include ../Makefile
```

### 2. Required Variables

Each benchmark Makefile **must** define:

- `BENCHMARK_NAME`: Unique identifier for the benchmark (used in labels, names, etc.)
- `KUSTOMIZE_DIR`: Path to the kustomize directory (e.g., `../kubernetes/MYBENCHMARK`)
- `DOCKERFILE_EVALUATOR`: Name of the evaluator Dockerfile
- `DOCKERFILE_DATA_LOADER`: Name of the data loader Dockerfile
- `RESULTS_PVC_NAME`: Name of the PVC for storing results (typically `$(BENCHMARK_NAME)-benchmark-results-pvc`)
- `RESULTS_FILES`: Space-separated list of result files to copy (e.g., `results.csv metrics.csv`)

### 3. Optional Variables

These can be overridden but have defaults:

- `KUBECTL_NAMESPACE`: Kubernetes namespace (default: `sage`)
- `KUBECTL_CONTEXT`: kubectl context (default: `nrp-dev`)
- `REGISTRY`: Docker registry (default: `gitlab-registry.nrp-nautilus.io/ndp/sage/hybrid-search`)
- `EVALUATOR_TAG`: Evaluator image tag (default: `latest`)
- `DATA_LOADER_TAG`: Data loader image tag (default: `latest`)

### 4. Benchmark-Specific Variables

Add any benchmark-specific environment variables that will be used by your Kubernetes deployments. These should match the environment variables expected by your benchmark code.

## Example: INQUIRE Makefile

See `benchmarks/INQUIRE/Makefile` for a complete example:

```makefile
BENCHMARK_NAME := inquire
KUSTOMIZE_DIR := ../../kubernetes/INQUIRE
DOCKERFILE_EVALUATOR := Dockerfile.benchmark
DOCKERFILE_DATA_LOADER := Dockerfile.data_loader
RESULTS_PVC_NAME := $(BENCHMARK_NAME)-benchmark-results-pvc
RESULTS_FILES := image_search_results.csv query_eval_metrics.csv

# INQUIRE-specific env vars
INQUIRE_DATASET ?= sagecontinuum/INQUIRE-Benchmark-small
IMAGE_BATCH_SIZE ?= 25
QUERY_BATCH_SIZE ?= 5

# Include the base Makefile (after setting variables)
include ../Makefile
```

## Usage

Once your Makefile is set up, use it from your benchmark directory:

```bash
cd benchmarking/MYBENCHMARK

# Build images
make build

# Deploy to Kubernetes
make deploy

# Load data
make load

# Run evaluation
make calculate

# Get results
make get

# View status
make status

# View logs
make logs-evaluator
make logs-data-loader

# Clean up
make clean
```

## How It Works

1. The benchmark-specific Makefile sets required variables
2. It includes the base Makefile using `include ../Makefile`
3. The base Makefile uses these variables to execute commands
4. All commands are generic and work with any benchmark that sets the required variables

## Benefits

- **DRY Principle**: No code duplication across benchmarks
- **Consistency**: All benchmarks use the same commands
- **Maintainability**: Fix bugs or add features once in the base Makefile
- **Flexibility**: Each benchmark can customize variables and add benchmark-specific logic

## Adding New Commands

To add a new command that all benchmarks can use:

1. Add it to `benchmarks/Makefile` (the base)
2. Use the standard variables (`BENCHMARK_NAME`, `KUBECTL_NAMESPACE`, etc.)
3. All benchmarks will automatically inherit the new command

## Overriding Commands

If a benchmark needs to override a base command, it can define its own version after the `include` statement:

```makefile
include ../Makefile

# ... variables ...

# Override the build command
build:
	@echo "Custom build for MYBENCHMARK"
	@# Custom build logic here
```

However, this should be rare - most customization should be done via variables.

