# Sage Image Search on NRP Benchmarks
>TODO: Try out https://duckdb.org/ for analyzing the results. It might make some of the analysis easier.
This repository contains benchmark implementations for evaluating vector databases and models using the [`imsearch_eval`](https://github.com/waggle-sensor/imsearch_eval) framework.

## What's in This Repository

This repository provides:
- **Benchmark implementations** (e.g., INQUIRE) that use the `imsearch_eval` framework
- **Template system** for creating new benchmarks
- **Makefile system** for building, deploying, and managing benchmarks on NRP
- **Dockerfile templates** for containerizing benchmarks for NRP
- **Kubernetes configurations** for deploying benchmarks on NRP

The framework code itself (interfaces, adapters, evaluator) is in the separate [`imsearch_eval`](https://github.com/waggle-sensor/imsearch_eval) package.

New runs default to **Milvus** on the NRP-managed cluster (`VECTOR_DB=milvus`, `MILVUS_DB=image_search_svc`) with per-benchmark collection names. Set `VECTOR_DB=weaviate` to use the in-cluster Weaviate path. Historical Weaviate result folders are unchanged.

CLIP rerank matches production: the query text tower runs once, then hits are scored with vectorized `cosine(image_vector, query_text) * logit_scale`. There is no per-hit image download or extra Triton CLIP call.

Index and query workers stay continuously filled (`WORKERS`, default 16). Concurrent CLIP calls are combined by Triton’s dynamic batcher when the server has `max_batch_size > 0`. Set `QUERY_BATCH_SIZE` / `IMAGE_BATCH_SIZE` at least as large as `WORKERS` (defaults 16 / 32); `IMAGE_BATCH_SIZE` is the streamed insert chunk size.

The existing benchmarks are in the [imsearch_benchmarks](https://github.com/waggle-sensor/imsearch_benchmarks) repository. Some of them have been implemented here in this repository.

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

## Leaderboards

Cross-version leaderboards are supported for benchmark results in `benchmarks/*/results/baseline`, `ablation_*`, and `v10+`.

### Leaderboard Notebooks

- **Per-benchmark leaderboard notebook(s)**: `benchmarks/*/results/leaderboard.ipynb`
- **Overall leaderboard notebook**: `benchmarks/overall/leaderboard.ipynb`
- **Reusable leaderboard helpers**: `helpers/plot.py`

### Ranking Modes

- **Primary**: `MRR` + `Success@25` (where `Success@25 = mean(hit)`)
- **Primary + Diversity**: `MRR` + `Success@25` + `Diversity@25`

Metric and interpretation details are documented in `benchmarks/METRICS.md`.

### Benchmark Auto-Discovery

Leaderboard logic does not hardcode benchmark names.

- Benchmarks are discovered dynamically from directories under `benchmarking/benchmarks/`
- Excludes non-benchmark folders such as `template` and `overall`
- New benchmark folders are automatically included (if they follow expected results layout)

### GitHub Action: Rerun Leaderboards

Workflow: `.github/workflows/leaderboards.yml`

This workflow automatically reruns leaderboard notebooks:

- Triggered on:
  - pushes/PRs that touch benchmark results (`benchmarking/benchmarks/**/results/**`)
  - leaderboard helper/workflow changes
  - manual dispatch (`workflow_dispatch`)
- Executes:
  - all `benchmarking/benchmarks/*/results/leaderboard.ipynb`
  - `benchmarking/benchmarks/overall/leaderboard.ipynb`

The workflow is configured to rerun/update notebooks only. It does **not** generate or upload leaderboard artifacts.

## Ablation Studies

Benchmark runs support env-driven ablations for index-time captioning, embedding modality, and query-time BM25 keyword search. Settings are recorded in each run's `config_values.csv`.

### Ablation Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `VECTOR_DB` | `milvus` | Backend: `milvus` (default) or `weaviate` |
| `WORKERS` | `16` | Thread-pool size for indexing and query eval (keep the pool full; fills Triton’s CLIP dynamic-batcher queue) |
| `IMAGE_BATCH_SIZE` | `32` | Streamed Milvus/Weaviate insert chunk size during indexing |
| `QUERY_BATCH_SIZE` | `16` | Kept for API compatibility; in-flight query concurrency is `WORKERS` |
| `ENABLE_CAPTION_GENERATION` | `true` | When `false`, skips the caption LLM entirely and stores an empty caption |
| `EMBED_IMAGE` | `true` | Milvus: omit the `image_vector` hybrid leg. Weaviate: index-time CLIP fusion uses image only when caption is also disabled |
| `EMBED_CAPTION` | `false` when caption generation is disabled | Milvus: omit the `caption_vector` hybrid leg. Weaviate: index-time CLIP fusion |
| `INDEX_CLIP_ALPHA` | `0.7` | Weaviate-only index-time fusion weight for image vs caption. Unused on Milvus (fusion is query-time via `QUERY_CLIP_ALPHA`) |
| `QUERY_CLIP_ALPHA` | `0.7` | Within dense retrieval, weight for `image_vector` vs `caption_vector` (Milvus) or fused query embedding (Weaviate) |
| `ENABLE_BM25` | `true` | When `false`, omits the BM25/keyword leg (Milvus) or sets hybrid `alpha=1.0` (Weaviate) |
| `QUERY_ALPHA` | `0.4` | Hybrid vector/keyword blend when `ENABLE_BM25=true`. A higher value means more weight is given to the vector modality |

`ENABLE_BM25=false` disables the BM25 keyword leg of hybrid search. CLIP rerank still runs against stored `image_vector`s (same `logits_per_image` math as production) unless you change `QUERY_METHOD` or set `rerank` to false.

Keep `IMAGE_BATCH_SIZE` and `QUERY_BATCH_SIZE` ≥ `WORKERS` so batch knobs do not under-subscribe the pool. Indexing streams inserts as items complete (no “process all → insert all” barrier).

Use a distinct `COLLECTION_NAME` for each ablation condition so indexed vectors do not mix across experiments. New Milvus result folders should use a new version name (e.g. `v13`) so leaderboards can compare against historical Weaviate runs.

### Indexing DLQ

Hard failures (`process_item` returns `None` / raises) and soft caption failures (LLM returned empty; **no** dataset-summary fallback) are held in an in-memory DLQ, retried with production-style exponential backoff, then written to CSV and uploaded with other metrics when `UPLOAD_TO_S3=true`.

| Variable | Default | Effect |
|----------|---------|--------|
| `DLQ_MAX_RETRIES` | `3` | Retry rounds after the first failure |
| `DLQ_RETRY_BASE_SECONDS` | `60` | Backoff base; delay = `base * 2^attempt` (60s → 120s → 240s) |
| `DLQ_FILE` | `dlq_records.csv` | Local/S3 filename for terminal DLQ outcomes |

Semantics:

- Soft caption failure: do **not** insert on first empty caption; retry captioning; after retries are exhausted, force-insert with an empty caption (`final_status=inserted_degraded`).
- Hard failure: never insert unless a retry succeeds (`retried_ok`); otherwise `abandoned`.
- CSV columns: `item_id`, `reason`, `error`, `attempts`, `final_status`, `last_error` (no image bytes).

### Example Ablation Runs

```bash
cd benchmarking/benchmarks/INQUIRE

# Baseline (current behavior)
COLLECTION_NAME=inquire-baseline make run

# No caption generation, image-only embedding
ENABLE_CAPTION_GENERATION=false EMBED_CAPTION=false \
COLLECTION_NAME=inquire-no-caption-img-only make run

# Caption-only embedding, vector-only retrieval
EMBED_IMAGE=false ENABLE_BM25=false \
COLLECTION_NAME=inquire-caption-only-vector make run

# Full pipeline but disable BM25 keyword matching
ENABLE_BM25=false COLLECTION_NAME=inquire-no-bm25 make run
```

Shared ablation logic lives in [`helpers/ablation.py`](helpers/ablation.py). Index-time CLIP fusion delegates to `imsearch_eval`'s `TritonModelUtils.get_clip_embeddings()`.

## Creating a New Benchmark

### Step 1: Copy Template

```bash
cd benchmarking/benchmarks
cp -r template MYBENCHMARK
cd MYBENCHMARK
```

### Step 2: Implement BenchmarkDataset

Create `benchmark_dataset.py` extending the `HuggingFaceDataset` adapter from `imsearch_eval`:

```python
from imsearch_eval.adapters.huggingface import HuggingFaceDataset

class MyBenchmarkDataset(HuggingFaceDataset):
    """Benchmark dataset class for MYBENCHMARK."""
    
    def get_query_column(self) -> str:
        """Return the column name containing query text."""
        return "query"
    
    def get_query_id_column(self) -> str:
        """Return the column name containing query IDs."""
        return "query_id"
    
    def get_relevance_column(self) -> str:
        """Return the column name containing relevance labels (1 for relevant, 0 for not)."""
        return "relevant"
    
    def get_metadata_columns(self) -> list:
        """Return optional metadata columns to include in evaluation stats."""
        return ["category", "type"]
```

The `HuggingFaceDataset` adapter handles loading datasets from HuggingFace Hub. You only need to implement the column mapping methods. The dataset is loaded using `benchmark_dataset.load_as_dataset(split="test", sample_size=0, seed=42, token=config._hf_token)`.

>NOTE: You can also implement new adapters for other vector databases and models. See the `imsearch_eval` repository for more information.

### Step 3: Create config.py

Create a `config.py` that implements the `Config` interface and loads all environment variables:

```python
import os
from imsearch_eval.framework.interfaces import Config

class MyConfig(Config):
    def __init__(self):
        self.MYBENCHMARK_DATASET = os.environ.get("MYBENCHMARK_DATASET", "your-dataset/name")
        # VECTOR_DB, MILVUS_*, WEAVIATE_* are loaded via helpers.backend.apply_vector_db_config
```

See `benchmarks/template/config.py` and `benchmarks/INQUIRE/config.py` for examples.

### Step 4: Create run_benchmark.py

Create `run_benchmark.py` that combines data loading and evaluation. The script should have:

1. A `load_data()` function that loads data into the vector database
2. A `run_evaluation()` function that runs the benchmark evaluation
3. An `upload_to_s3()` function for S3 uploads (optional)
4. A `main()` function that orchestrates the complete benchmark run

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
    
    # Process and insert data
    results = data_loader.process_batch(batch_size=config._image_batch_size, 
                                        dataset=hf_dataset, 
                                        workers=config._workers)
    inserted = vector_db.insert_data(config._collection_name, results, 
                                     batch_size=config._image_batch_size)

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
# Core benchmarking framework (install with all extras needed)
imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.2.0
imsearch_eval[milvus] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.2.0
imsearch_eval[triton] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.2.0
imsearch_eval[huggingface] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.2.0
```

# S3 upload support (MinIO)
minio>=7.2.0

# Add other dependencies as needed (e.g., Pillow, python-dateutil)
```

### Step 7: Create Kubernetes Config

```bash
cd ../../kubernetes
cp -r ../benchmarks/template/kubernetes MYBENCHMARK
cd MYBENCHMARK
# Update kustomization.yaml, env.yaml, etc.
```

See `benchmarks/template/README.md` for complete instructions.

### Step 8: Upload Results

Once you have successfully run the benchmark and have the results, you can upload them into a folder called `results` under the benchmark folder.
'''
benchmarking/benchmarks/MYBENCHMARK/results/v10/
benchmarking/benchmarks/MYBENCHMARK/results/v11/
...
'''

### Step 9: Leaderboard Checklist for New Benchmarks

When you add a new benchmark folder under `benchmarking/benchmarks/`, use this checklist:

1. Create benchmark results version folders in the standard shape:
   - `benchmarking/benchmarks/MYBENCHMARK/results/v10/`
2. Ensure each version folder includes:
   - `query_eval_metrics.csv`
3. Include the expected metric columns in `query_eval_metrics.csv`:
   - `hit` (for `Success@25`)
   - `rerank_score_reciprocal_rank` (for `MRR`)
   - `diversity` (for Primary + Diversity mode)
4. Add `benchmarking/benchmarks/MYBENCHMARK/results/leaderboard.ipynb` if you want a benchmark-specific leaderboard notebook.
5. Commit results/notebook changes; the leaderboard workflow will auto-discover the benchmark and rerun notebooks.

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
- `._huggingface-secret.yaml` - HuggingFace token secret (use the template file as a guide)
- `._nrp-secret.yaml` - NRP Envoy AI Gateway API key (`NRP_API_KEY`; use the template file as a guide)
- `._milvus-secret.yaml` - NRP Milvus URI and token (use the template file as a guide)
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
- Shared adapters (`MilvusAdapter`, `WeaviateAdapter`, `TritonModelProvider`, etc.)

Install it via:
```bash
# Install with all extras needed for benchmarks
pip install "imsearch_eval[weaviate,milvus,triton,huggingface,nrp] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.2.0"
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
