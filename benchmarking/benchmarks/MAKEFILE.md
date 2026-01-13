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

- **Build**: `make build` - Build Docker job image
- **Run**: `make run` - Deploy and run benchmark job (loads data and evaluates)
- **Run Local**: `make run-local` - Run benchmark locally with port-forwarding
- **Status**: `make status` - Show deployment status
- **Logs**: `make logs` - View job logs
- **Port Forward**: `make port-forward-start` / `make port-forward-stop` - Manage port-forwarding
- **Down**: `make down` - Remove deployments

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
DOCKERFILE_JOB := Dockerfile.job
RESULTS_FILES := image_search_results.csv query_eval_metrics.csv
ENV ?= dev
ifeq ($(ENV),prod)
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-prod
else
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK
endif

# ============================================================================
# Optional Variables (can be overridden)
# ============================================================================
KUBECTL_NAMESPACE := sage
KUBECTL_CONTEXT ?= nautilus
REGISTRY := gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search
JOB_TAG ?= latest

# Local run script
RUN_SCRIPT := run_benchmark.py

# Include the base Makefile (after setting variables)
include ../Makefile
```

### 2. Required Variables

Each benchmark Makefile **must** define:

- `BENCHMARK_NAME`: Unique identifier for the benchmark (used in labels, names, etc.)
- `DOCKERFILE_JOB`: Name of the job Dockerfile (typically `Dockerfile.job`)
- `RESULTS_FILES`: Space-separated list of result files to copy (e.g., `image_search_results.csv query_eval_metrics.csv`)
- `KUSTOMIZE_DIR`: Path to the kustomize directory (can be conditional based on `ENV`)

### 3. Optional Variables

These can be overridden but have defaults:

- `KUBECTL_NAMESPACE`: Kubernetes namespace (default: `sage`)
- `KUBECTL_CONTEXT`: kubectl context (default: `nautilus`)
- `REGISTRY`: Docker registry (default: `gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search`)
- `JOB_TAG`: Job image tag (default: `latest`)
- `RUN_SCRIPT`: Script to run locally (default: `run_benchmark.py`)

### 4. Environment Switching

The Makefile supports switching between dev and prod environments:

```makefile
ENV ?= dev
ifeq ($(ENV),prod)
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-prod
else
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK
endif
```

Then use:
```bash
make run ENV=prod    # Run using prod resources
make run            # Run using dev resources (default)
```

## Example: INQUIRE Makefile

See `benchmarks/INQUIRE/Makefile` for a complete example:

```makefile
BENCHMARK_NAME := inquire
DOCKERFILE_JOB := Dockerfile.job
RESULTS_FILES := image_search_results.csv query_eval_metrics.csv
ENV ?= dev
ifeq ($(ENV),prod)
  KUSTOMIZE_DIR := ../../kubernetes/INQUIRE/nrp-prod
else
  KUSTOMIZE_DIR := ../../kubernetes/INQUIRE
endif

KUBECTL_NAMESPACE := sage
KUBECTL_CONTEXT ?= nautilus
REGISTRY := gitlab-registry.nrp-nautilus.io/ndp/sage/nrp-image-search
JOB_TAG ?= latest

RUN_SCRIPT := run_benchmark.py

include ../Makefile
```

## Usage

Once your Makefile is set up, use it from your benchmark directory:

```bash
cd benchmarking/MYBENCHMARK

# Build image
make build

# Run benchmark (deploys and runs the job)
make run

# Monitor logs
make logs

# View status
make status

# Run locally (with port-forwarding)
make run-local

# Clean up
make down
```

## Local Development

For local development, use port-forwarding:

```bash
# Start port-forwarding manually
make port-forward-start

# Run your benchmark script locally
python run_benchmark.py

# Stop port-forwarding
make port-forward-stop

# Or use the convenience command (does all of the above)
make run-local
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
