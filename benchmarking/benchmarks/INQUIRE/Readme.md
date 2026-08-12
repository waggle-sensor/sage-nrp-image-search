# INQUIRE Benchmark

Here we use [INQUIRE](https://github.com/inquire-benchmark/INQUIRE) with Milvus (default) or Weaviate as the vector database for benchmarking. Different models were used to generate captions and keywords for the images. Also different models were used to generate the embeddings for the images.

## Usage

This benchmark is supposed to be used in conjunction with [Sage Image Search](../../../kubernetes/base/). The Makefile references components that are deployed in [Sage Image Search](../../../kubernetes/base/) and deploys additional containers that are used to run the INQUIRE Benchmark.

### Leaderboard

INQUIRE leaderboard is supported for benchmark results in `benchmarks/INQUIRE/results/leaderboard.ipynb`. Check out the notebook for more details.
>NOTE: INQUIRE authors have their own leaderboard for the INQUIRE benchmark. This leaderboard is not the same as the one we have here. This leaderboard is modified to fit our use case and compatible with our benchmarking framework. Although, you can still use the author's leaderboard for more information.

## Running the Example

### Prerequisites
To run this example, you'll need:
- **Kubernetes cluster** access with `kubectl` configured
- **kustomize** (or kubectl with kustomize support)
- **Docker** for building images
- **Triton** deployed (from `kubernetes/nrp-dev` or `kubernetes/nrp-prod`) and NRP Milvus credentials (`MILVUS_TOKEN`). Weaviate is optional (`VECTOR_DB=weaviate`).

### Step-by-Step Setup

1. **Deploy Sage Image Search Infrastructure**:
   - Navigate to the main [kubernetes](../../../kubernetes) directory and deploy base services:
     ```bash
     kubectl apply -k nrp-dev  # or nrp-prod
     ```

2. **Build and Push Images**:
   - Build the benchmark image:
     ```bash
     cd benchmarking/benchmarks/INQUIRE
     make build
     ```
   - Push to registry (update registry in Makefile if needed):
     ```bash
     docker push <registry>/benchmark-inquire-job:latest
     ```
>NOTE: You can also use the GitHub Actions to build and push the images to the registry. See `.github/workflows/benchmarking.yml` for more details.

3. **Run INQUIRE Benchmark**:
   - Deploy and run the complete benchmark (loads data and evaluates):
     ```bash
     make run  # defaults to dev environment
     ```
   - Monitor progress:
     ```bash
     make logs
     ```
   >NOTE: This loads [INQUIRE-Benchmark-small](https://huggingface.co/datasets/sagecontinuum/INQUIRE-Benchmark-small) into the configured vector database, runs the evaluation, and saves results.

5. **Run Locally (Development)**:
   - For local development with port-forwarding:
     ```bash
     make run-local
     ```
   - This will automatically set up port-forwarding and run the benchmark locally.

### Results

Once the benchmark is run, three CSV files will be generated:
- **`image_search_results.csv`**: Metadata of all images returned when different queries were being run
- **`query_eval_metrics.csv`**: Calculated evaluation metrics (NDCG, precision, recall, etc.) based on images returned by different queries
- **`config_values.csv`**: Configuration values used for the benchmark run (generated via `config.to_csv()`)

Results are saved to `/app/results` when running in Kubernetes (with volume mount), or to the current directory when running locally with `make run-local`. Results can also be automatically uploaded to S3 if configured (with timestamps: `{S3_PREFIX}/{timestamp}/{filename}`).

## References
- [Weaviate Blog: NDCG](https://weaviate.io/blog/retrieval-evaluation-metrics#normalized-discounted-cumulative-gain-ndcg)
- [RAG Evaluation](https://weaviate.io/blog/rag-evaluation)
- [Scikit-Learn NDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [A Guide on NDCG](https://www.aporia.com/learn/a-practical-guide-to-normalized-discounted-cumulative-gain-ndcg/)
- [Weaviate: Batch import](https://weaviate.io/developers/weaviate/manage-data/import)
- [Weaviate: Imports in Detail](https://weaviate.io/developers/weaviate/tutorials/import#data-import---best-practices)
- [INQUIRE](https://inquire-benchmark.github.io/)
- [Huggingface: Fine-tuning Florence2](https://huggingface.co/blog/finetune-florence2)
- [Medium: Fine-tuning Florence2](https://medium.com/@amit25173/fine-tuning-florence-2-aa9c99b2a83d)

## Citation
```
@article{vendrow2024inquire,
  title={INQUIRE: A Natural World Text-to-Image Retrieval Benchmark},
  author={Vendrow, Edward and Pantazis, Omiros and Shepard, Alexander and Brostow, Gabriel and Jones, Kate E and Mac Aodha, Oisin and Beery, Sara and Van Horn, Grant},
  journal={NeurIPS},
  year={2024},
}
```

# INQUIRE Benchmark Structure

## Overview

INQUIRE is a benchmark instance that uses the abstract benchmarking framework provided by the `imsearch-eval` Python package. The framework is installed from GitHub: https://github.com/waggle-sensor/imsearch_eval. This instance uses the INQUIRE dataset with Milvus by default (`VECTOR_DB=milvus`) and can still run against Weaviate.

## Directory Structure

```
benchmarking/
└── benchmarks/
    └── INQUIRE/                      # INQUIRE benchmark instance
        ├── benchmark_dataset.py        # INQUIRE-specific benchmark dataset (BenchmarkDataset)
        ├── data_loader.py             # INQUIRE-specific data loader (DataLoader)
        ├── config.py                  # INQUIRE-specific configuration (Config)
        ├── run_benchmark.py           # Main script loads data and evaluates)
        ├── requirements.txt           # Dependencies including imsearch-eval package
        ├── Dockerfile.job             # Dockerfile for the combined job
        ├── Makefile                   # Makefile for building and deploying
        └── Readme.md                  # INQUIRE-specific instructions

The framework and adapters are provided by the imsearch-eval package:
- Repository: https://github.com/waggle-sensor/imsearch_eval
- Package: imsearch_eval[weaviate]
- Installation: pip install imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@main
```

## Key Components

### 1. Config Class (`config.py`)

Implements `Config` interface for INQUIRE benchmark:
- Loads all environment variables (dataset, collection, S3 settings, etc.)
- Defines Weaviate HNSW hyperparameters
- Defines model and query hyperparameters
- Provides caption prompts for different models

### 2. Benchmark Dataset Class (`benchmark_dataset.py`)

Extends `HuggingFaceDataset` adapter for INQUIRE dataset:
- Extends `HuggingFaceDataset` from `imsearch_eval.adapters.huggingface`
- Loads from HuggingFace: `sagecontinuum/INQUIRE-Benchmark-small` (via `load_as_dataset()`)
- Defines column mappings: `query`, `query_id`, `relevant`
- Provides metadata columns: `category`, `supercategory`, `iconic_group`

### 3. Data Loader Class (`data_loader.py`)

Implements `DataLoader` interface for INQUIRE dataset:
- Processes INQUIRE dataset items
- Generates captions using model provider
- Generates CLIP embeddings
- Returns formatted data for Weaviate insertion
- Provides schema configuration for Weaviate collection

### 4. Main Script (`run_benchmark.py`)

Combined script that:
1. **Step 0**: Sets up benchmark environment (initializes clients and adapters)
2. **Step 1**: Loads data into vector database (calls `load_data()` function)
3. **Step 2**: Runs evaluation (calls `run_evaluation()` function)
4. **Step 3**: Saves results locally
5. **Step 4**: Optionally uploads results to S3

The script uses a `config` object (instance of `INQUIREConfig`) to access all configuration values.

### 5. Shared Adapters (from `imsearch-eval` package)

**MilvusAdapter and MilvusQuery** (default):
- Provided by `imsearch_eval.adapters.milvus`
- Dual-dense hybrid (`image_vector` + `caption_vector` + BM25) plus CLIP image-text rerank
- Import: `from imsearch_eval.adapters import MilvusAdapter, MilvusQuery`
- Selected via `helpers.backend.init_vector_db` when `VECTOR_DB=milvus`

**WeaviateAdapter and WeaviateQuery**:
- Provided by `imsearch_eval.adapters.weaviate`
- Used when `VECTOR_DB=weaviate`
- Import: `from imsearch_eval.adapters import WeaviateAdapter, WeaviateQuery`

**TritonModelProvider and TritonModelUtils**:
- Provided by `imsearch_eval.adapters.triton`
- `TritonModelUtils`: Implements `ModelUtils` interface
- `TritonModelProvider`: Uses `TritonModelUtils` for model operations
- Supports: CLIP, ColBERT, ALIGN embeddings
- Supports: Gemma3, Qwen2.5-VL captioning
- Import: `from imsearch_eval.adapters import TritonModelProvider, TritonModelUtils`

## Usage

### Running on Kubernetes

```bash
cd benchmarking/benchmarks/INQUIRE
make build      # Build Docker image
make run        # Deploy and run benchmark job
make logs       # Monitor logs
```

### Running Locally

```bash
cd benchmarking/benchmarks/INQUIRE
make run-local  # Runs with automatic port-forwarding
```

## Environment Variables

All environment variables are loaded through the `INQUIREConfig` class in `config.py`:

**Dataset Parameters:**
- `INQUIRE_DATASET`: HuggingFace dataset name (default: `sagecontinuum/INQUIRE-Benchmark-small`)
- `SAMPLE_SIZE`: Number of samples to use (0 = all, default: 0)
- `SEED`: Random seed for sampling (default: 42)
- `HF_TOKEN`: HuggingFace token (from secret, optional)

**Vector DB Parameters:**
- `VECTOR_DB`: `milvus` (default) or `weaviate`
- `MILVUS_URI`: NRP Milvus URI (default: `https://milvus.nrp-nautilus.io:50051`)
- `MILVUS_TOKEN`: NRP Milvus token (`username:password`)
- `MILVUS_DB`: database name (default: `image_search_svc`)
- `WEAVIATE_HOST`: Weaviate host (default: 127.0.0.1)
- `WEAVIATE_PORT`: Weaviate HTTP port (default: 8080)
- `WEAVIATE_GRPC_PORT`: Weaviate gRPC port (default: 50051)
- `COLLECTION_NAME`: collection name (default: INQUIRE)

**Inference Server Parameters:**
- `TRITON_HOST`: Triton host (default: triton)
- `TRITON_PORT`: Triton port (default: 8001)

**Processing Parameters:**
- `WORKERS`: Number of parallel workers (default: 5)
- `IMAGE_BATCH_SIZE`: Batch size for processing images (default: 25)
- `QUERY_BATCH_SIZE`: Batch size for parallel queries (default: 5)

**Query Parameters:**
- `QUERY_METHOD`: Query method (Milvus default: `clip_hybrid_query_dual_index`)
- `TARGET_VECTOR`: Target vector name (Milvus default: `image_vector`)
- `RESPONSE_LIMIT`: Maximum number of results to return (default: 50)
- `QUERY_ALPHA`: Hybrid query alpha parameter (default: 0.4)
- `QUERY_CLIP_ALPHA`: CLIP fusion weight at query time (default: 0.7)
- `AUTOCUT_JUMPS`: Autocut jumps (default: 0)
- `RERANK_PROP`: Property to use for reranking (default: caption)

**Ablation Parameters:**
- `ENABLE_CAPTION_GENERATION`: Run caption LLM during indexing (default: true)
- `EMBED_IMAGE`: Include image in index-time CLIP embedding (default: true)
- `EMBED_CAPTION`: Include caption in index-time CLIP embedding (default: true)
- `INDEX_CLIP_ALPHA`: Image/caption fusion weight at index time (default: 0.7)
- `ENABLE_BM25`: Enable BM25 keyword leg in hybrid search (default: true)

See `benchmarking/README.md` for ablation run examples.

**HNSW Hyperparameters:**
- `HNSW_DIST_METRIC`: Distance metric (default: COSINE)
- `HNSW_EF`: EF parameter (default: -1)
- `HNSW_EF_CONSTRUCTION`: EF construction (default: 100)
- `HNSW_MAX_CONNECTIONS`: Max connections (default: 50)
- And more... (see `config.py` for full list)

**S3 Upload Parameters:**
- `UPLOAD_TO_S3`: Enable S3 upload (default: false)
- `S3_BUCKET`: S3 bucket name (default: sage_imsearch)
- `S3_PREFIX`: S3 prefix for uploaded files (default: dev-metrics)
- `S3_ENDPOINT`: S3 endpoint URL (default: http://rook-ceph-rgw-nautiluss3.rook)
- `S3_ACCESS_KEY`: S3 access key (from secret)
- `S3_SECRET_KEY`: S3 secret key (from secret)
- `S3_SECURE`: Use TLS for S3 (default: false)

**Results Files:**
- `IMAGE_RESULTS_FILE`: Image results filename (default: image_search_results.csv)
- `QUERY_EVAL_METRICS_FILE`: Query metrics filename (default: query_eval_metrics.csv)
- `CONFIG_VALUES_FILE`: Config values filename (default: config_values.csv)

**Logging:**
- `LOG_LEVEL`: Logging level (default: INFO)

## Extending INQUIRE

To add new components to INQUIRE:

1. **New Vector DB**: Add adapter to the `imsearch-eval` package (contribute to the repository)
2. **New Model**: Add provider to the `imsearch-eval` package (contribute to the repository)
3. **New Query Method**: Add method to `WeaviateQuery` in the `imsearch-eval` package
4. **New Model Function**: Add method to `TritonModelUtils` in the `imsearch-eval` package or implement `ModelUtils` interface
5. **New Dataset**: Create benchmark dataset class in INQUIRE directory (benchmark-specific)

## Framework Package

The abstract framework and adapters are provided by the `imsearch-eval` Python package:

- **Repository**: https://github.com/waggle-sensor/imsearch_eval
- **Package**: `imsearch_eval[weaviate,triton,huggingface]`
- **Installation**: 
  ```bash
  pip install "imsearch_eval[weaviate,triton,huggingface] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.1.0"
  ```

This allows:
- Multiple benchmark instances to share framework and adapter code
- Framework and adapter updates to benefit all benchmarks
- Clear separation between shared code and instance-specific code
- Easy reuse of adapters across different benchmarks
- **Independence from `app/`**: All functions are in the framework package, won't break when `app/` changes
- **Easy distribution**: Benchmarks can be used in any environment by installing the package
