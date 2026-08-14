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

Extend the `HuggingFaceDataset` adapter for HuggingFace Hub datasets:

```python
from imsearch_eval.adapters.huggingface import HuggingFaceDataset

class MyBenchmarkDataset(HuggingFaceDataset):
    """Benchmark dataset class for MYBENCHMARK."""
    
    def get_query_column(self) -> str:
        """Return the column name containing query text."""
        return "query"  # TODO: Update with your column name
    
    def get_query_id_column(self) -> str:
        """Return the column name containing query IDs."""
        return "query_id"  # TODO: Update with your column name
    
    def get_relevance_column(self) -> str:
        """Return the column name containing relevance labels (1 for relevant, 0 for not)."""
        return "relevant"  # TODO: Update with your column name
    
    def get_metadata_columns(self) -> list:
        """Return optional metadata columns to include in evaluation stats."""
        return []  # TODO: Add metadata columns if available (e.g., ["category", "type"])
```

The `HuggingFaceDataset` adapter handles loading datasets from HuggingFace Hub. You only need to implement the column mapping methods. The dataset is loaded using `benchmark_dataset.load_as_dataset(split="test", sample_size=0, seed=42, token=config._hf_token)`.

#### `run_benchmark.py` - Benchmark Script

This script should:
1. Create a config instance at the top
2. Define a `load_data(data_loader, vector_db, hf_dataset)` function that loads data into the vector database
3. Define a `run_evaluation(evaluator, hf_dataset)` function that runs the evaluation
4. Define an `upload_to_s3(local_file_path, s3_key)` function for S3 uploads (optional)
5. In `main()`, set up clients/adapters, then call both functions sequentially
6. Save results locally (three CSV files: `image_search_results.csv`, `query_eval_metrics.csv`, `config_values.csv`)
7. Optionally upload results to S3

The structure should be:
```python
from config import MyConfig
from imsearch_eval import BenchmarkEvaluator, VectorDBAdapter
from imsearch_eval.adapters import TritonModelProvider
from helpers.backend import init_vector_db
from benchmark_dataset import MyBenchmarkDataset
from data_loader import MyDataLoader  # Optional

config = MyConfig()

def load_data(data_loader, vector_db: VectorDBAdapter, hf_dataset):
    """Load dataset into vector database."""
    # Create collection schema
    schema_config = data_loader.get_schema_config()
    vector_db.create_collection(schema_config)
    
    # Process and stream-insert data (worker pool stays full; inserts as chunks complete)
    inserted = 0
    def on_batch(chunk):
        nonlocal inserted
        inserted += vector_db.insert_data(
            config._collection_name, chunk, batch_size=len(chunk), flush=False
        )
    data_loader.process_batch(
        batch_size=config._image_batch_size,
        dataset=hf_dataset,
        workers=config._workers,
        on_batch=on_batch,
    )
    flush = getattr(vector_db, "flush_collection", None)
    if callable(flush):
        flush(config._collection_name)

def run_evaluation(evaluator: BenchmarkEvaluator, hf_dataset):
    """Run the benchmark evaluation."""
    image_results, query_evaluation = evaluator.evaluate_queries(
        query_batch_size=config._query_batch_size,
        dataset=hf_dataset,
        workers=config._workers
    )
    return image_results, query_evaluation

def main():
    # Step 0: Set up clients and adapters
    # Step 1: Call load_data(data_loader, vector_db, hf_dataset)
    # Step 2: Call run_evaluation(evaluator, hf_dataset)
    # Step 3: Save results (image_search_results.csv, query_eval_metrics.csv, config_values.csv)
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
# Core benchmarking framework (install with all extras needed)
imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.1.0
imsearch_eval[triton] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.1.0
imsearch_eval[huggingface] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.1.0

# S3 upload support (MinIO)
minio>=7.2.0

# Add other dependencies as needed
# Pillow>=10.0.0
# python-dateutil>=2.8.0
```

## Required Components

### Must Implement

1. **BenchmarkDataset** (`benchmark_dataset.py`): Extends `HuggingFaceDataset` and defines column mappings
2. **Config** (`config.py`): Configuration class that loads all environment variables and implements `to_csv()` method
3. **run_benchmark.py**: Script that includes:
   - Config instance creation
   - `load_data(data_loader, vector_db, hf_dataset)` function: Loads data into vector database
   - `run_evaluation(evaluator, hf_dataset)` function: Runs the evaluation
   - `upload_to_s3(local_file_path, s3_key)` function: Uploads results to S3 (optional)
   - `main()` function: Sets up environment, then orchestrates the complete benchmark run

### Optional Components

1. **DataLoader** (`data_loader.py`): Custom data processing/insertion logic
2. Additional hyperparameters in `config.py` (e.g., Weaviate HNSW settings, model parameters)

## Using Shared Adapters

The `imsearch-eval` package provides shared adapters you can use:

**Triton adapters**:
- **TritonModelProvider**: For Triton inference server (implements `ModelProvider`)
- **TritonModelUtils**: Triton implementation of `ModelUtils` interface

**Milvus adapters** (default):
- **MilvusAdapter**: For Milvus vector database (implements `VectorDBAdapter`)
- **MilvusQuery**: Dual-dense + BM25 hybrid with CLIP rerank on stored `image_vector`s

**Weaviate adapters**:
- **WeaviateAdapter**: For Weaviate vector database (implements `VectorDBAdapter`)
- **WeaviateQuery**: Weaviate query implementation (implements `Query` interface)

Import them:

```python
from imsearch_eval.adapters import MilvusAdapter, WeaviateAdapter, TritonModelProvider, WeaviateQuery, TritonModelUtils
```

**Note**: Install the package with all extras needed:
```bash
pip install "imsearch_eval[weaviate,milvus,triton,huggingface] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.2.0"
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

## Results

The benchmark generates three CSV files:

1. **`image_search_results.csv`**: Metadata of all images returned by the vector database for each query
2. **`query_eval_metrics.csv`**: Calculated evaluation metrics (NDCG, precision, recall, etc.) for each query
3. **`config_values.csv`**: Configuration values used for the benchmark run (generated via `config.to_csv()`)

Results are saved to `/app/results` if the directory exists (when running in Kubernetes with volume mount), otherwise to the current directory.

### Leaderboard

The template benchmark leaderboard is added in `benchmarks/template/results/leaderboard.ipynb`. Check out the notebook for more details and to see how to add a leaderboard to your benchmark.

## S3 Upload Configuration

Results can be automatically uploaded to S3-compatible storage (MinIO). Configuration is done via:

- **Base Kubernetes config**: S3 endpoint, bucket, and secure flag are set in `benchmarking/kubernetes/base/benchmark-job.yaml`
- **S3 Secret**: Access key and secret key are stored in `benchmarking/kubernetes/base/._s3-secret.yaml`
- **NRP Secret**: `NRP_API_KEY` is stored in `benchmarking/kubernetes/base/._nrp-secret.yaml` (copy from `nrp-secret.template.yaml`; required when `LLM_MODEL_PROVIDER=nrp`)
- **Benchmark-specific**: Override `S3_PREFIX` in your benchmark's `nrp-dev/env.yaml` or `nrp-prod/env.yaml` if needed

To enable S3 upload, set `UPLOAD_TO_S3=true` in the base config (already enabled by default). Results are uploaded with timestamps: `{S3_PREFIX}/{timestamp}/{filename}`.

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
