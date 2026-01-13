# Benchmark Template

This directory contains templates and documentation for creating new benchmark instances.

## Quick Start

To create a new benchmark:

```bash
cd benchmarking/benchmarks
cp -r template MYBENCHMARK
cd MYBENCHMARK
# Customize the files as described below
```

## Directory Structure

A new benchmark should have the following structure:

```
MYBENCHMARK/
├── Makefile                    # Benchmark-specific Makefile (from template)
├── Dockerfile.job              # Combined job container (from template)
├── requirements.txt            # Python dependencies
├── run_benchmark.py           # Combined benchmark script (loads data and evaluates)
├── benchmark_dataset.py        # BenchmarkDataset implementation
├── data_loader.py              # DataLoader implementation (optional)
├── config.py                   # Config implementation (recommended)
└── README.md                   # Benchmark-specific documentation
```

## Step-by-Step Setup

### 1. Create Benchmark Directory

```bash
cd benchmarking/benchmarks
cp -r template MYBENCHMARK
cd MYBENCHMARK
```

### 2. Update Makefile

Edit `Makefile` and set the required variables:

```makefile
BENCHMARK_NAME := mybenchmark
DOCKERFILE_JOB := Dockerfile.job
RESULTS_FILES := image_search_results.csv query_eval_metrics.csv
ENV ?= dev
ifeq ($(ENV),prod)
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-prod
else
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-dev
endif
```

### 3. Update Dockerfile

The `Dockerfile.job` is already set up to run `run_benchmark.py`. Verify the CMD line is correct.

### 4. Create Python Files

#### `config.py` - Configuration Class (Recommended)

Create a Config class that extends the `Config` interface and loads all environment variables:

```python
import os
from imsearch_eval.framework.interfaces import Config

class MyConfig(Config):
    def __init__(self):
        # Environment variables
        self.MYBENCHMARK_DATASET = os.environ.get("MYBENCHMARK_DATASET", "your-dataset/name")
        self.WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "127.0.0.1")
        self.WEAVIATE_PORT = os.environ.get("WEAVIATE_PORT", "8080")
        self.WEAVIATE_GRPC_PORT = os.environ.get("WEAVIATE_GRPC_PORT", "50051")
        self.TRITON_HOST = os.environ.get("TRITON_HOST", "triton")
        self.TRITON_PORT = os.environ.get("TRITON_PORT", "8001")
        self.COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "MYBENCHMARK")
        # ... add more as needed
```

See `config.py` template and `../INQUIRE/config.py` for complete examples.

#### `benchmark_dataset.py` - Implement BenchmarkDataset

```python
import os
import pandas as pd
from datasets import load_dataset

from imsearch_eval.framework.interfaces import BenchmarkDataset

class MyBenchmarkDataset(BenchmarkDataset):
    def load(self, split="test", **kwargs) -> pd.DataFrame:
        """Load your dataset."""
        dataset = load_dataset("your-dataset/name", split=split)
        return dataset.to_pandas()
    
    def get_query_column(self) -> str:
        return "query"  # Column name with query text
    
    def get_query_id_column(self) -> str:
        return "query_id"  # Column name with query IDs
    
    def get_relevance_column(self) -> str:
        return "relevant"  # Column name with relevance labels (1/0)
    
    def get_metadata_columns(self) -> list:
        return []  # Optional metadata columns
```

#### `run_benchmark.py` - Benchmark Script

This script should:
1. Create a config instance at the top
2. Define a `load_data(vector_db, model_provider)` function that loads data into the vector database
3. Define a `run_evaluation(vector_db, model_provider)` function that runs the evaluation
4. In `main()`, set up clients/adapters, then call both functions sequentially
5. Save results locally
6. Optionally upload to S3

The structure should be:
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

See `../INQUIRE/run_benchmark.py` for a complete example.

### 5. Create Kubernetes Configuration

Use the Kubernetes template from this directory:

```bash
cd ../../kubernetes
cp -r ../benchmarks/template/kubernetes MYBENCHMARK
cd MYBENCHMARK
# Replace MYBENCHMARK with your benchmark name in all files
find . -type f -name "*.yaml" -exec sed -i '' 's/MYBENCHMARK/mybenchmark/g' {} +
```

Then customize:
- `kustomization.yaml`: Update image name
- `env.yaml`: Set benchmark-specific environment variables

See `../../kubernetes/README.md` for detailed instructions.

### 6. Create requirements.txt

Create a `requirements.txt` with your dependencies:

```txt
# Core benchmarking framework
imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main

# S3 upload support (MinIO)
minio>=7.2.0

# Add other dependencies as needed
# datasets>=2.14.0
# huggingface-hub>=0.16.0
# Pillow>=10.0.0
```

## Required Components

### Must Implement

1. **BenchmarkDataset** (`benchmark_dataset.py`): Loads your dataset and defines column mappings
2. **Config** (`config.py`): Configuration class that loads all environment variables (recommended pattern)
3. **run_benchmark.py**: Script that includes:
   - Config instance creation
   - `load_data(vector_db, model_provider)` function: Loads data into vector database
   - `run_evaluation(vector_db, model_provider)` function: Runs the evaluation
   - `main()` function: Sets up environment, then orchestrates the complete benchmark run

### Optional Components

1. **DataLoader** (`data_loader.py`): Custom data processing/insertion logic
2. Additional hyperparameters in `config.py` (e.g., Weaviate HNSW settings, model parameters)

## Using Shared Adapters

The `imsearch-eval` package provides shared adapters you can use:

**Triton adapters**:
- **TritonModelProvider**: For Triton inference server (implements `ModelProvider`)
- **TritonModelUtils**: Triton implementation of `ModelUtils` interface

**Weaviate adapters**:
- **WeaviateAdapter**: For Weaviate vector database (implements `VectorDBAdapter`)
- **WeaviateQuery**: Weaviate query implementation (implements `Query` interface)

Import them:

```python
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider, WeaviateQuery, TritonModelUtils
```

**Note**: Install the package with the `[weaviate]` extra to get all adapters:
```bash
pip install imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main
```

## Deployment

Once everything is set up:

1. **Build and run benchmark**:
   ```bash
   make build    # Build Docker image
   make run      # Deploy and run benchmark job
   ```

3. **Monitor logs**:
   ```bash
   make logs
   ```

4. **Run locally (with port-forwarding)**:
   ```bash
   make run-local
   ```

## S3 Upload Configuration

Results can be automatically uploaded to S3-compatible storage (MinIO). Configuration is done via:

- **Base Kubernetes config**: S3 endpoint, bucket, and secure flag are set in `benchmarking/kubernetes/base/benchmark-job.yaml`
- **S3 Secret**: Access key and secret key are stored in `benchmarking/kubernetes/base/s3-secret.yaml`
- **Benchmark-specific**: Override `S3_PREFIX` in your benchmark's `nrp-dev/env.yaml` or `nrp-prod/env.yaml` if needed

To enable S3 upload, set `UPLOAD_TO_S3=true` in the base config (already enabled by default).

## Framework Structure

The benchmarking framework is now provided as a Python package (`imsearch-eval`) installed from GitHub:

```
benchmarking/
└── benchmarks/            # Benchmark instances
    ├── template/         # Template for new benchmarks
    └── INQUIRE/         # Example benchmark implementation
```

The framework code (`framework/` and `adapters/`) is now in a separate repository:
- **Repository**: https://github.com/waggle-sensor/imsearch_eval
- **Package name**: `imsearch-eval`
- **Installation**: `pip install imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main`

## Next Steps

- Review `../README.md` for framework overview
- Review `../MAKEFILE.md` for Makefile details (same directory level)
- Review `../DOCKER.md` for Dockerfile details (same directory level)
- Review `../../kubernetes/README.md` for Kubernetes setup
- Look at `../INQUIRE/` as a complete example (same directory level)

## Getting Help

- Check existing benchmarks (e.g., `../INQUIRE/`) for examples
- Review framework documentation: https://github.com/waggle-sensor/imsearch_eval
- Review adapter documentation in the `imsearch-eval` package
