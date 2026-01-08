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
- ✅ Ready-to-use Makefile and Dockerfiles
- ✅ Python templates for `main.py`, `load_data.py`, `benchmark_dataset.py`
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

### Step 3: Update main.py

Edit `main.py` to use your benchmark dataset class and configure your benchmark:

```python
from imsearch_eval import BenchmarkEvaluator
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider
from benchmark_dataset import MyBenchmarkDataset

# Your benchmark-specific code here
```

### Step 4: Update Makefile

Edit `Makefile` and set:
- `BENCHMARK_NAME`
- `KUSTOMIZE_DIR`
- `RESULTS_FILES`

### Step 5: Update requirements.txt

Add the `imsearch_eval` package:

```txt
imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main
# Add other dependencies as needed
```

### Step 6: Create Kubernetes Config

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
- `make build` - Build Docker images
- `make deploy` - Deploy to Kubernetes
- `make load` - Start data loader
- `make calculate` - Run benchmark evaluation
- `make get` - Copy results from pod
- `make status` - Show deployment status
- `make logs-evaluator` / `make logs-data-loader` - View logs
- `make clean` - Remove deployments and PVCs

See `benchmarks/MAKEFILE.md` for detailed documentation.

## Dockerfile System

The Dockerfile system provides templates for consistent container builds.

### Template Files

- `benchmarks/Dockerfile.template` - Base template
- `benchmarks/template/Dockerfile.benchmark` - Evaluator template
- `benchmarks/template/Dockerfile.data_loader` - Data loader template

### Creating Benchmark Dockerfiles

1. Copy from template: `cp benchmarks/template/Dockerfile.benchmark benchmarks/MYBENCHMARK/`
2. Update `CMD` line for your entrypoint script
3. Ensure `requirements.txt` includes `imsearch_eval` package

See `benchmarks/DOCKER.md` for detailed documentation.

## Kubernetes Deployment

### Base Resources

Located in `kubernetes/base/`, these provide common Kubernetes resources:
- `benchmark-evaluator.yaml` - Evaluator deployment template
- `benchmark-data-loader.yaml` - Data loader job template
- `kustomization.yaml` - Base kustomization config

### Benchmark-Specific Configs

Each benchmark has its own directory under `kubernetes/` (e.g., `kubernetes/INQUIRE/`) with:
- `kustomization.yaml` - Extends base, sets images, patches
- `env.yaml` - Environment variables for evaluator
- `data-loader-env.yaml` - Environment variables for data loader
- `results-pvc.yaml` - Persistent volume for results
- `gpus.yaml` - GPU configuration (optional)

### Deployment Workflow

1. **Build images**: `make build` (in benchmark directory)
2. **Deploy**: `make deploy`
3. **Load data**: `make load`
4. **Run evaluation**: `make calculate`
5. **Get results**: `make get`

See `kubernetes/README.md` for detailed Kubernetes documentation.

## Template Directory

The `benchmarks/template/` directory provides a complete starting point for new benchmarks:

- **README.md**: Comprehensive guide for creating new benchmarks
- **QUICKSTART.md**: 5-minute quick start guide
- **Makefile**: Template with all required variables
- **Dockerfile.benchmark** & **Dockerfile.data_loader**: Ready-to-use Dockerfiles
- **Python Templates**: Template files for `main.py`, `load_data.py`, `benchmark_dataset.py`
- **requirements.txt**: Base dependencies including `imsearch_eval`
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
