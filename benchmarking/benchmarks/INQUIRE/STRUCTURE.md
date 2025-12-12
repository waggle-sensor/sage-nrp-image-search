# INQUIRE Benchmark Structure

## Overview

INQUIRE is a benchmark instance that uses the abstract benchmarking framework provided by the `imsearch-eval` Python package. The framework is installed from GitHub: https://github.com/waggle-sensor/imsearch_eval. This instance uses the INQUIRE dataset with Weaviate as the vector database.

## Directory Structure

```
benchmarking/
└── benchmarks/
    └── INQUIRE/                      # INQUIRE benchmark instance
        ├── dataset_loader.py        # INQUIRE-specific dataset loader (DatasetLoader)
        ├── data_loader.py           # INQUIRE-specific data loader (DataLoader)
        ├── config.py                # INQUIRE-specific configuration (Config)
        ├── load_data.py              # Script to load data into vector DB
        ├── main.py                  # Entry point for benchmarking
        ├── requirements.txt          # Dependencies including imsearch-eval package
        └── README.md                # INQUIRE-specific instructions

The framework and adapters are provided by the imsearch-eval package:
- Repository: https://github.com/waggle-sensor/imsearch_eval
- Package: imsearch_eval[weaviate]
- Installation: pip install imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.1.0
```

## Key Components

### 1. Dataset Loader (`dataset_loader.py`)

Implements `DatasetLoader` interface for INQUIRE dataset:
- Loads from HuggingFace: `sagecontinuum/INQUIRE-Benchmark-small`
- Defines column mappings: `query`, `query_id`, `relevant`
- Provides metadata columns: `category`, `supercategory`, `iconic_group`

### 2. Shared Adapters (from `imsearch-eval` package)

**WeaviateAdapter and WeaviateQuery**:
- Provided by `imsearch_eval.adapters.weaviate`
- `WeaviateQuery`: Independent implementation of Weaviate query methods
- `WeaviateAdapter`: Uses `WeaviateQuery` for search operations
- Supports query methods: `clip_hybrid_query`, `hybrid_query`, `colbert_query`, etc.
- Implements `init_client()` class method for client initialization
- Import: `from imsearch_eval.adapters import WeaviateAdapter, WeaviateQuery`

**TritonModelProvider and TritonModelUtils**:
- Provided by `imsearch_eval.adapters.triton`
- `TritonModelUtils`: Implements `ModelUtils` interface
- `TritonModelProvider`: Uses `TritonModelUtils` for model operations
- Supports: CLIP, ColBERT, ALIGN embeddings
- Supports: Gemma3, Qwen2.5-VL captioning
- Import: `from imsearch_eval.adapters import TritonModelProvider, TritonModelUtils`

### 3. Main Entry Point (`main.py`)

Wires everything together:
1. Initializes clients (Weaviate, Triton)
2. Creates adapters
3. Creates dataset loader
4. Creates evaluator with all components
5. Runs evaluation
6. Saves results

## Usage

```bash
cd benchmarking/benchmarks/INQUIRE
python main.py
```

## Environment Variables

- `INQUIRE_DATASET`: HuggingFace dataset name
- `WEAVIATE_HOST`: Weaviate host (default: 127.0.0.1)
- `TRITON_HOST`: Triton host (default: triton)
- `COLLECTION_NAME`: Weaviate collection (default: INQUIRE)
- `QUERY_METHOD`: Query method to use (default: clip_hybrid_query)

## Extending INQUIRE

To add new components to INQUIRE:

1. **New Vector DB**: Add adapter to the `imsearch-eval` package (contribute to the repository)
2. **New Model**: Add provider to the `imsearch-eval` package (contribute to the repository)
3. **New Query Method**: Add method to `WeaviateQuery` in the `imsearch-eval` package
4. **New Model Function**: Add method to `TritonModelUtils` in the `imsearch-eval` package or implement `ModelUtils` interface
5. **New Dataset**: Create dataset loader in INQUIRE directory (benchmark-specific)

## Framework Package

The abstract framework and adapters are provided by the `imsearch-eval` Python package:

- **Repository**: https://github.com/waggle-sensor/imsearch_eval
- **Package**: `imsearch_eval[weaviate]`
- **Installation**: `pip install imsearch_eval[weaviate] @ git+https://github.com/waggle-sensor/imsearch_eval.git@0.1.0`

This allows:
- Multiple benchmark instances to share framework and adapter code
- Framework and adapter updates to benefit all benchmarks
- Clear separation between shared code and instance-specific code
- Easy reuse of adapters across different benchmarks
- **Independence from `app/`**: All functions are in the framework package, won't break when `app/` changes
- **Easy distribution**: Benchmarks can be used in any environment by installing the package

