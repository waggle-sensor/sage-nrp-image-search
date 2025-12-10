# INQUIRE Benchmark Structure

## Overview

INQUIRE is a benchmark instance that uses the abstract benchmarking framework located at `../../framework/`. This structure allows multiple benchmark instances to share the same framework code. This instance uses the INQUIRE dataset with Weaviate as the vector database.

## Directory Structure

```
benchmarking/
├── framework/                    # Abstract framework (shared across all benchmarks)
│   ├── __init__.py
│   ├── interfaces.py            # VectorDBAdapter, ModelProvider, DatasetLoader, DataLoader, Config
│   └── evaluator.py             # BenchmarkEvaluator (dataset-agnostic)
│
├── adapters/                     # Shared concrete implementations
│   ├── __init__.py
│   ├── weaviate.py              # WeaviateAdapter and WeaviateQuery (implements Query interface)
│   └── triton.py                 # TritonModelProvider and TritonModelUtils (implements ModelUtils interface)
│
└── INQUIRE/                      # INQUIRE benchmark instance
    ├── dataset_loader.py        # INQUIRE-specific dataset loader (DatasetLoader)
    ├── data_loader.py           # INQUIRE-specific data loader (DataLoader)
    ├── config.py                # INQUIRE-specific configuration (Config)
    ├── load_data.py              # Script to load data into vector DB
    ├── main.py                  # Entry point for benchmarking
    └── README.md                # INQUIRE-specific instructions
```

## Key Components

### 1. Dataset Loader (`dataset_loader.py`)

Implements `DatasetLoader` interface for INQUIRE dataset:
- Loads from HuggingFace: `sagecontinuum/INQUIRE-Benchmark-small`
- Defines column mappings: `query`, `query_id`, `relevant`
- Provides metadata columns: `category`, `supercategory`, `iconic_group`

### 2. Shared Adapters (`../adapters/`)

**WeaviateAdapter and WeaviateQuery**:
- Located in `benchmarking/adapters/weaviate.py`
- `WeaviateQuery`: Independent implementation of Weaviate query methods (no dependency on `app/query.py`)
- `WeaviateAdapter`: Uses `WeaviateQuery` for search operations
- Supports query methods: `clip_hybrid_query`, `hybrid_query`, `colbert_query`, etc.
- Implements `init_client()` class method for client initialization

**TritonModelProvider and TritonModelUtils**:
- Located in `benchmarking/adapters/triton.py`
- `TritonModelUtils`: Implements `ModelUtils` interface (independent of `app/model.py`)
- `TritonModelProvider`: Uses `TritonModelUtils` for model operations
- Supports: CLIP, ColBERT, ALIGN embeddings
- Supports: Gemma3, Qwen2.5-VL captioning

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

1. **New Vector DB**: Add adapter in `../../adapters/` (shared across benchmarks)
2. **New Model**: Add provider in `../../adapters/` (shared across benchmarks)
3. **New Query Method**: Add method to `WeaviateQuery` in `../../adapters/weaviate.py`
4. **New Model Function**: Add method to `TritonModelUtils` in `../../adapters/triton.py` or implement `ModelUtils` interface
5. **New Dataset**: Create dataset loader in INQUIRE directory

## Shared Components Location

The abstract framework and adapters are shared at the `benchmarking/` level:

- **Framework** (`../../framework/`): Abstract interfaces, evaluation logic, and model utilities
- **Adapters** (`../../adapters/`): Concrete implementations for vector DBs and models

This allows:
- Multiple benchmark instances to share framework and adapter code
- Framework and adapter updates to benefit all benchmarks
- Clear separation between shared code and instance-specific code
- Easy reuse of adapters across different benchmarks
- **Independence from `app/`**: All functions are in the framework, won't break when `app/` changes

