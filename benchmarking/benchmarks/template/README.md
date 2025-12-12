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
├── Dockerfile.benchmark        # Evaluator container (from template)
├── Dockerfile.data_loader      # Data loader container (from template)
├── requirements.txt            # Python dependencies
├── main.py                     # Benchmark evaluator entry point
├── load_data.py                # Data loading script
├── dataset_loader.py           # DatasetLoader implementation
├── data_loader.py              # DataLoader implementation (optional)
├── config.py                   # Config implementation (optional)
└── README.md                   # Benchmark-specific documentation
```

## Step-by-Step Setup

### 1. Create Benchmark Directory

```bash
cd benchmarking
cp -r template MYBENCHMARK
cd MYBENCHMARK
```

### 2. Update Makefile

Edit `Makefile` and set the required variables:

```makefile
BENCHMARK_NAME := mybenchmark
KUSTOMIZE_DIR := ../kubernetes/MYBENCHMARK
DOCKERFILE_EVALUATOR := Dockerfile.benchmark
DOCKERFILE_DATA_LOADER := Dockerfile.data_loader
RESULTS_PVC_NAME := $(BENCHMARK_NAME)-benchmark-results-pvc
RESULTS_FILES := results.csv metrics.csv
```

### 3. Update Dockerfiles

The Dockerfiles are already set up, but verify:
- `Dockerfile.benchmark`: CMD should run `main.py`
- `Dockerfile.data_loader`: CMD should run `load_data.py`

### 4. Create Python Files

#### `dataset_loader.py` - Implement DatasetLoader

```python
import os
import pandas as pd
from datasets import load_dataset

from imsearch_eval.framework.interfaces import DatasetLoader

class MyDatasetLoader(DatasetLoader):
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

#### `main.py` - Benchmark Entry Point

```python
import os
import logging
from tritonclient.grpc import InferenceServerClient as TritonClient

from imsearch_eval import BenchmarkEvaluator
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider
from dataset_loader import MyDatasetLoader

# Environment variables
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
TRITON_HOST = os.getenv("TRITON_HOST", "triton")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "MYBENCHMARK")
QUERY_METHOD = os.getenv("QUERY_METHOD", "clip_hybrid_query")

def main():
    """Run the benchmark evaluation."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize clients
    weaviate_client = WeaviateAdapter.init_client(host=WEAVIATE_HOST)
    triton_client = TritonClient(url=TRITON_HOST)
    
    # Create adapters
    vector_db = WeaviateAdapter(weaviate_client=weaviate_client, triton_client=triton_client)
    model_provider = TritonModelProvider(triton_client=triton_client)
    
    # Create dataset loader
    dataset_loader = MyDatasetLoader()
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(
        vector_db=vector_db,
        model_provider=model_provider,
        dataset_loader=dataset_loader,
        collection_name=COLLECTION_NAME,
        query_method=QUERY_METHOD
    )
    
    # Run evaluation
    try:
        results, stats = evaluator.evaluate_queries(split="test")
    finally:
        vector_db.close()
    
    # Save results
    results.to_csv("/app/results/results.csv", index=False)
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
```

#### `load_data.py` - Data Loading Script

```python
import os
from tritonclient.grpc import InferenceServerClient as TritonClient

from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider
from data_loader import MyDataLoader  # If you have a DataLoader
from config import MyConfig  # If you have a Config

def main():
    """Load data into vector database."""
    # Initialize clients and adapters
    # Create collection
    # Load data
    pass

if __name__ == "__main__":
    main()
```

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
- `kustomization.yaml`: Update image names
- `env.yaml`: Set evaluator environment variables
- `data-loader-env.yaml`: Set data loader environment variables
- `results-pvc.yaml`: Adjust storage size if needed
- `gpus.yaml`: Remove if GPUs not needed

See `kubernetes/README.md` for detailed instructions.

### 6. Create requirements.txt

Create a `requirements.txt` with your dependencies:

```txt
# Core benchmarking framework
imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main

# Add other dependencies as needed
# datasets>=2.14.0
# huggingface-hub>=0.16.0
# Pillow>=10.0.0
```

## Required Components

### Must Implement

1. **DatasetLoader** (`dataset_loader.py`): Loads your dataset and defines column mappings
2. **main.py**: Entry point that wires everything together

### Optional Components

1. **DataLoader** (`data_loader.py`): Custom data processing/insertion logic
2. **Config** (`config.py`): Benchmark-specific hyperparameters
3. **load_data.py**: Script to load data (if not using default DataLoader)

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

1. **Deploy to Kubernetes**:
   ```bash
   make deploy
   ```

2. **Load data**:
   ```bash
   make load
   ```

3. **Run evaluation**:
   ```bash
   make calculate
   ```

4. **Get results**:
   ```bash
   make get
   ```

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

