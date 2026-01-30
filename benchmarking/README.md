# Sage Image Search on NRP Benchmarks

This repository contains benchmark implementations for evaluating vector databases and models using the [`imsearch_eval`](https://github.com/waggle-sensor/imsearch_eval) framework.

## What's in This Repository

This repository provides:
- **Benchmark implementations** (e.g., INQUIRE) that use the `imsearch_eval` framework
- **Template system** for creating new benchmarks
- **Makefile system** for building, deploying, and managing benchmarks on NRP
- **Dockerfile templates** for containerizing benchmarks for NRP
- **Kubernetes configurations** for deploying benchmarks on NRP

The framework code itself (interfaces, adapters, evaluator) is in the separate [`imsearch_eval`](https://github.com/waggle-sensor/imsearch_eval) package.

## Quick Start: Creating a New Benchmark

```bash
cd benchmarking
cp -r benchmarks/template benchmarks/MYBENCHMARK
cd benchmarks/MYBENCHMARK
# Follow instructions in README.md
```

The `benchmarks/template/` directory contains everything you need:
- ✅ Ready-to-use Makefile and Dockerfile.job
- ✅ Python templates for `run_benchmark.py`, `config.py`, `benchmark_dataset.py`
- ✅ Comprehensive documentation and quick start guide

See `benchmarks/template/README.md` for detailed setup instructions, or `benchmarks/template/QUICKSTART.md` for a 5-minute guide.

## Repository Structure

```
benchmarking/
├── benchmarks/                   # Benchmark implementations
│   ├── template/                # Template for creating new benchmarks
│   ├── INQUIRE/                # INQUIRE benchmark implementation
│   ├── Makefile                # Base Makefile (included by benchmarks)
│   ├── MAKEFILE.md             # Makefile documentation
│   ├── Dockerfile.template      # Base Dockerfile template
│   └── DOCKER.md               # Dockerfile documentation
└── kubernetes/                  # Kubernetes deployment configurations
    ├── base/                   # Base Kubernetes resources
    └── INQUIRE/                # INQUIRE-specific Kubernetes configs
```

## Existing Benchmarks

### INQUIRE

- **Location**: `benchmarks/INQUIRE/`
- **Dataset**: INQUIRE benchmark for natural world image retrieval
- **Vector DB**: Weaviate
- **Models**: CLIP, ColBERT, ALIGN (embeddings); Gemma3, Qwen2.5-VL (captions)
- **Usage**: See `benchmarks/INQUIRE/Readme.md`

## Creating a New Benchmark

### Step 1: Copy Template

```bash
cd benchmarking/benchmarks
cp -r template MYBENCHMARK
cd MYBENCHMARK
```

### Step 2: Implement BenchmarkDataset

Create `benchmark_dataset.py` implementing the `BenchmarkDataset` interface from `imsearch_eval`:

```python
from imsearch_eval.framework.interfaces import BenchmarkDataset
import pandas as pd

class MyBenchmarkDataset(BenchmarkDataset):
    def load(self, split="test", **kwargs) -> pd.DataFrame:
        # Load your dataset
        return dataset_df
    
    def get_query_column(self) -> str:
        return "query"
    
    def get_query_id_column(self) -> str:
        return "query_id"
    
    def get_relevance_column(self) -> str:
        return "relevant"
    
    def get_metadata_columns(self) -> list:
        return ["category", "type"]
```
>NOTE: You can also implement new adapters for other vector databases and models. See the `imsearch_eval` repository for more information.

### Step 3: Create config.py

Create a `config.py` that implements the `Config` interface and loads all environment variables:

```python
import os
from imsearch_eval.framework.interfaces import Config

class MyConfig(Config):
    def __init__(self):
        self.MYBENCHMARK_DATASET = os.environ.get("MYBENCHMARK_DATASET", "your-dataset/name")
        self.WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "127.0.0.1")
        # ... add more environment variables
```

See `benchmarks/template/config.py` and `benchmarks/INQUIRE/config.py` for examples.

### Step 4: Create run_benchmark.py

Create `run_benchmark.py` that combines data loading and evaluation:

```python
from config import MyConfig
config = MyConfig()

def load_data(vector_db: VectorDBAdapter, model_provider: ModelProvider):
    # Implement data loading logic
    pass

def run_evaluation(vector_db: VectorDBAdapter, model_provider: ModelProvider):
    # Implement evaluation logic
    pass

def main():
    # Step 0: Set up clients and adapters
    # Step 1: Call load_data(vector_db, model_provider)
    # Step 2: Call run_evaluation(vector_db, model_provider)
    # Step 3: Save results
    # Step 4: Upload to S3 (optional)
    pass
```

See `benchmarks/INQUIRE/run_benchmark.py` for a complete example.

### Step 5: Update Makefile

Edit `Makefile` and set:
- `BENCHMARK_NAME`
- `DOCKERFILE_JOB`
- `KUSTOMIZE_DIR`
- `RESULTS_FILES`

### Step 6: Update requirements.txt

Add the required packages:

```txt
imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main
minio>=7.2.0
# Add other dependencies as needed
```

### Step 7: Create Kubernetes Config

```bash
cd ../../kubernetes
cp -r ../benchmarks/template/kubernetes MYBENCHMARK
cd MYBENCHMARK
# Update kustomization.yaml, env.yaml, etc.
```

See `benchmarks/template/README.md` for complete instructions.

## Makefile System

The Makefile system provides consistent commands across all benchmarks.

### Base Makefile

Located at `benchmarks/Makefile`, this contains reusable commands that all benchmarks inherit.

### Benchmark Makefiles

Each benchmark has its own `Makefile` that:
1. Sets benchmark-specific variables
2. Includes the base Makefile: `include ../Makefile`

### Common Commands

All benchmarks support:
- `make build` - Build Docker job image
- `make run` - Deploy and run benchmark job (loads data and evaluates)
- `make run-local` - Run benchmark locally with port-forwarding
- `make status` - Show deployment status
- `make logs` - View job logs
- `make down` - Remove deployments

See `benchmarks/MAKEFILE.md` for detailed documentation.

## Dockerfile System

The Dockerfile system provides templates for consistent container builds.

### Template Files

- `benchmarks/Dockerfile.template` - Base template
- `benchmarks/template/Dockerfile.job` - Combined job template

### Creating Benchmark Dockerfiles

1. Copy from template: `cp benchmarks/template/Dockerfile.job benchmarks/MYBENCHMARK/`
2. Verify `CMD` line runs `run_benchmark.py`
3. Ensure `requirements.txt` includes `imsearch_eval` and `minio` packages

See `benchmarks/DOCKER.md` for detailed documentation.

## Kubernetes Deployment

### Base Resources

Located in `kubernetes/base/`, these provide common Kubernetes resources:
- `benchmark-job.yaml` - Combined job template (loads data and evaluates)
- `._s3-secret.yaml` - S3 credentials secret (use the template file as a guide)
- `kustomization.yaml` - Base kustomization config

> **Important:** 
> All secret files you actually use must be named with leading `._` per `.gitignore` and not checked into version control! Only commit the `*.template.yaml` files.

### Benchmark-Specific Configs

Each benchmark has its own directory under `kubernetes/` (e.g., `kubernetes/INQUIRE/`) with `nrp-dev/` and `nrp-prod/` overlays:
- `nrp-dev/` - Development environment overlay (default)
  - `kustomization.yaml` - Extends base, sets images, patches
  - `env.yaml` - Environment variables for dev environment
- `nrp-prod/` - Production environment overlay (optional)
  - `kustomization.yaml` - Extends base, sets images, patches
  - `env.yaml` - Environment variables for prod environment

### Deployment Workflow

1. **Build image**: `make build` (in benchmark directory)
2. **Run benchmark**: `make run` (deploys and runs the benchmark job)
4. **Monitor**: `make logs`
5. **Status**: `make status`

See `kubernetes/README.md` for detailed Kubernetes documentation.

## Template Directory

The `benchmarks/template/` directory provides a complete starting point for new benchmarks:

- **README.md**: Comprehensive guide for creating new benchmarks
- **QUICKSTART.md**: 5-minute quick start guide
- **Makefile**: Template with all required variables
- **Dockerfile.job**: Ready-to-use combined job Dockerfile
- **Python Templates**: Template files for `run_benchmark.py`, `load_data.py`, `benchmark_dataset.py`
- **requirements.txt**: Base dependencies including `imsearch_eval` and `minio`
- **kubernetes/**: Complete Kubernetes template

## Dependencies

All benchmarks depend on the [`imsearch_eval`](https://github.com/waggle-sensor/imsearch_eval) package, which provides:
- Abstract interfaces (`VectorDBAdapter`, `ModelProvider`, `Query`, `BenchmarkDataset`, etc.)
- Evaluation logic (`BenchmarkEvaluator`)
- Shared adapters (`WeaviateAdapter`, `TritonModelProvider`, etc.)

Install it via:
```bash
pip install imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main
```

See the [`imsearch_eval` README](https://github.com/waggle-sensor/imsearch_eval) for framework documentation.

## Documentation

- **Framework Documentation**: See [`imsearch_eval` repository](https://github.com/waggle-sensor/imsearch_eval)
- **Makefile System**: `benchmarks/MAKEFILE.md`
- **Dockerfile System**: `benchmarks/DOCKER.md`
- **Kubernetes**: `kubernetes/README.md`
- **Template Guide**: `benchmarks/template/README.md`
- **Quick Start**: `benchmarks/template/QUICKSTART.md`
- **INQUIRE Benchmark**: `benchmarks/INQUIRE/Readme.md`
