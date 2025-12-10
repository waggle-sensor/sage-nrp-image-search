# Benchmarking Framework Migration Summary

## Overview

The benchmarking code has been refactored to eliminate code duplication and provide an abstract, extensible framework for benchmarking different vector databases and models.

## What Changed

### ✅ New Structure

```
benchmarking/
├── framework/              # NEW: Abstract interfaces and evaluation logic (dataset-agnostic)
│   ├── __init__.py
│   ├── interfaces.py      # VectorDBAdapter, ModelProvider, DatasetLoader interfaces
│   └── evaluator.py       # BenchmarkEvaluator class
└── INQUIRE/               # INQUIRE benchmark instance
    ├── dataset_loader.py  # INQUIRE-specific dataset loader
    ├── adapters/          # Concrete implementations
    │   ├── __init__.py
    │   ├── weaviate_adapter.py    # Uses Weav_query from app/query.py
    │   └── triton_model_provider.py  # Uses functions from app/model.py
    ├── main.py            # Entry point using framework
    ├── README_FRAMEWORK.md    # INQUIRE-specific documentation
    └── app/               # DEPRECATED: Old duplicated code
```

### ✅ Key Improvements

1. **No Code Duplication**
   - Reuses `Weav_query` from `app/query.py`
   - Reuses model functions from `app/model.py`
   - Reuses client initialization from `weavloader/client.py`

2. **Abstract Interfaces**
   - `VectorDBAdapter`: Interface for any vector database
   - `ModelProvider`: Interface for any model provider
   - `DatasetLoader`: Interface for any dataset
   - Easy to add new implementations

3. **Extensibility**
   - Add new vector DBs (Pinecone, Qdrant, etc.) by implementing `VectorDBAdapter`
   - Add new models by implementing `ModelProvider`
   - Add new datasets by implementing `DatasetLoader`
   - Create new benchmark instances by combining adapters and dataset loaders
   - No changes needed to evaluation logic

## Migration Guide

### For Users

**Old way** (deprecated):
```bash
cd benchmarking/INQUIRE/app
python main.py
```

**New way**:
```bash
cd benchmarking/INQUIRE
python main.py
```

### For Developers

**Adding a new vector database:**

1. Create adapter in `adapters/`:
```python
from framework.interfaces import VectorDBAdapter, QueryResult

class MyVectorDBAdapter(VectorDBAdapter):
    def search(self, query, collection_name, limit=25, **kwargs):
        # Your implementation
        return QueryResult(results)
    
    def close(self):
        # Cleanup
        pass
```

2. Use in `main.py`:
```python
vector_db = MyVectorDBAdapter(...)
evaluator = BenchmarkEvaluator(vector_db=vector_db, ...)
```

**Adding a new model:**

1. Create provider in `adapters/`:
```python
from framework.interfaces import ModelProvider

class MyModelProvider(ModelProvider):
    def get_embedding(self, text, image=None, model_name="default"):
        # Your implementation
        return embedding
    
    def generate_caption(self, image, model_name="default"):
        # Your implementation
        return caption
```

2. Use in `main.py`:
```python
model_provider = MyModelProvider(...)
evaluator = BenchmarkEvaluator(
    vector_db=vector_db,
    model_provider=model_provider,
    dataset_loader=dataset_loader,
    ...
)
```

**Adding a new dataset:**

1. Create loader implementing `DatasetLoader`:
```python
from framework.interfaces import DatasetLoader

class MyDatasetLoader(DatasetLoader):
    def load(self, split="test", **kwargs):
        # Load your dataset
        return dataset_df
    
    def get_query_column(self) -> str:
        return "query"
    
    def get_query_id_column(self) -> str:
        return "query_id"
    
    def get_relevance_column(self) -> str:
        return "relevant"
```

2. Use in `main.py`:
```python
dataset_loader = MyDatasetLoader()
evaluator = BenchmarkEvaluator(
    vector_db=vector_db,
    model_provider=model_provider,
    dataset_loader=dataset_loader,
    ...
)
```

## Files Removed/Deprecated

- `benchmarking/INQUIRE/app/main.py` → Use `benchmarking/INQUIRE/main.py`
- `benchmarking/INQUIRE/app/inquire_eval.py` → Logic moved to `benchmarking/framework/evaluator.py`
- `benchmarking/INQUIRE/app/query.py` → Now imports from `app/query.py`
- `benchmarking/INQUIRE/app/model.py` → Now imports from `app/model.py`
- `benchmarking/INQUIRE/app/client.py` → Now imports from `weavloader/client.py`
- `benchmarking/INQUIRE/framework/` → Moved to `benchmarking/framework/` (shared across all benchmarks)

## Backward Compatibility

The old code in `app/` is kept for reference but marked as deprecated. The new framework maintains the same evaluation metrics and output format, so existing analysis scripts should continue to work.

## Testing

To test the new framework:

```bash
# Set environment variables
export WEAVIATE_HOST=127.0.0.1
export TRITON_HOST=triton
export COLLECTION_NAME=INQUIRE

# Run benchmark
python main.py
```

## Benefits

1. **Maintainability**: Single source of truth for query and model code
2. **Consistency**: Same code used in app and benchmarks
3. **Flexibility**: Easy to benchmark different combinations
4. **Extensibility**: Simple to add new vector DBs/models
5. **Testability**: Clear interfaces make testing easier

