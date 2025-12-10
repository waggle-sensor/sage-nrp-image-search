# Benchmarking Framework

Abstract, extensible framework for benchmarking vector databases and models across different datasets.

## 🚀 Quick Start: Creating a New Benchmark

**New to the framework?** Start here:

```bash
cd benchmarking
cp -r benchmarks/template benchmarks/MYBENCHMARK
cd benchmarks/MYBENCHMARK
# Follow instructions in README.md
```

The `benchmarks/template/` directory contains everything you need:
- ✅ Ready-to-use Makefile and Dockerfiles
- ✅ Python templates for `main.py`, `load_data.py`, `dataset_loader.py`
- ✅ Comprehensive documentation and quick start guide

See `benchmarks/template/README.md` for detailed setup instructions, or `benchmarks/template/QUICKSTART.md` for a 5-minute guide.

## Architecture

The framework is organized into three main components:

1. **Framework** (`framework/`): Abstract interfaces and evaluation logic (dataset-agnostic)
2. **Adapters** (`adapters/`): Shared concrete implementations for vector databases and models
3. **Benchmark Instances** (e.g., `benchmarks/INQUIRE/`): Specific benchmark implementations using the framework

### File Structure

```
benchmarking/
├── framework/                    # Abstract interfaces and evaluation logic
│   ├── interfaces.py            # VectorDBAdapter, ModelProvider, Query, DatasetLoader, etc.
│   ├── model_utils.py           # ModelUtils abstract interface
│   └── evaluator.py            # BenchmarkEvaluator class
│
├── adapters/                     # Shared concrete implementations
│   ├── __init__.py             # Exports all adapters
│   ├── triton.py               # TritonModelProvider, TritonModelUtils
│   └── weaviate.py             # WeaviateAdapter, WeaviateQuery
│
└── benchmarks/                   # Benchmark instances
    ├── template/                # Template for creating new benchmarks
    └── INQUIRE/            # Example benchmark implementation
```

## Key Features

- **Dataset-Agnostic**: Works with any dataset by implementing `DatasetLoader`
- **Independent**: No dependency on `app/` - all functions implemented in framework
- **Extensible**: Easy to add new vector databases, models, and datasets
- **Abstract Interfaces**: Clean separation between evaluation logic and implementations
- **Stable**: Functions preserved in framework won't break when `app/` changes

## Framework Components

### Interfaces (`framework/interfaces.py`)

- **`VectorDBAdapter`**: Abstract interface for vector databases
  - Methods: `init_client()`, `search()`, `create_collection()`, `delete_collection()`, `insert_data()`, `close()`
- **`ModelProvider`**: Abstract interface for model providers
  - Methods: `get_embedding()`, `generate_caption()`
- **`Query`**: Abstract interface for query classes (used by vector DB adapters)
  - Method: `query(near_text, collection_name, limit, query_method, **kwargs)` - Generic query method
  - Each vector DB implementation can define its own query types via `query_method` parameter
- **`ModelUtils`**: Abstract interface for model utilities (in `framework/model_utils.py`)
  - Methods: `calculate_embedding()`, `generate_caption()`
- **`DatasetLoader`**: Abstract interface for dataset loaders
- **`DataLoader`**: Abstract interface for loading data into vector DBs
- **`Config`**: Abstract interface for configuration/hyperparameters
- **`QueryResult`**: Container for query results

### Shared Adapters (`adapters/`)

**Triton adapters** (`adapters/triton.py`):
- `TritonModelUtils`: Triton-based implementation of `ModelUtils` interface
- `TritonModelProvider`: Triton inference server model provider

**Weaviate adapters** (`adapters/weaviate.py`):
- `WeaviateQuery`: Implements `Query` interface for Weaviate
  - Generic `query()` method routes to specific methods based on `query_method` parameter
  - Also provides Weaviate-specific methods: `hybrid_query()`, `colbert_query()`, `clip_hybrid_query()`
- `WeaviateAdapter`: Implements `VectorDBAdapter` interface for Weaviate
  - Uses `WeaviateQuery` internally for search operations

**Usage:**
```python
from adapters import WeaviateAdapter, TritonModelProvider

# Initialize
weaviate_client = WeaviateAdapter.init_client(host="127.0.0.1", port="8080")
triton_client = TritonClient.InferenceServerClient(url="triton:8001")

# Create adapters
vector_db = WeaviateAdapter(weaviate_client=weaviate_client, triton_client=triton_client)
model_provider = TritonModelProvider(triton_client=triton_client)

# Use in evaluator
evaluator = BenchmarkEvaluator(
    vector_db=vector_db,
    model_provider=model_provider,
    dataset_loader=dataset_loader,
    collection_name="my_collection",
    query_method="clip_hybrid_query"  # Passed to Query.query() method
)
```

These adapters can be reused across all benchmark instances and are independent of `app/`.

### Framework Utilities (`framework/model_utils.py`)

Abstract interface for model utilities:

- **`ModelUtils`**: Abstract class with two methods:
  - **`calculate_embedding(text, image, model_name)`**: Calculate embeddings for text/image
  - **`generate_caption(image, model_name)`**: Generate captions for images
- **`fuse_embeddings()`**: Utility function for fusing embeddings

**Concrete implementations** are in `adapters/`:
- `TritonModelUtils` (in `adapters/triton.py`): Implements `ModelUtils` using Triton Inference Server
- You can create other implementations (e.g., OpenAI, HuggingFace) by implementing the `ModelUtils` interface

### Evaluator (`framework/evaluator.py`)

- `BenchmarkEvaluator`: Main evaluation class that works with any combination of adapters and dataset loaders
- Computes metrics: NDCG, precision, recall, accuracy
- Supports parallel query processing

## Creating a New Benchmark Instance

To create a new benchmark (e.g., for a different dataset):

1. **Create a dataset loader** implementing `DatasetLoader`:

```python
from framework.interfaces import DatasetLoader
import pandas as pd

class MyDatasetLoader(DatasetLoader):
    def load(self, split="test", **kwargs) -> pd.DataFrame:
        # Load your dataset
        return dataset_df
    
    def get_query_column(self) -> str:
        return "query"  # Column name with query text
    
    def get_query_id_column(self) -> str:
        return "query_id"  # Column name with query IDs
    
    def get_relevance_column(self) -> str:
        return "relevant"  # Column name with relevance labels (1/0)
    
    def get_metadata_columns(self) -> list:
        return ["category", "type"]  # Optional metadata columns
```

2. **Use or create adapters**:

You can reuse shared adapters from `adapters/`:
```python
from adapters import WeaviateAdapter, TritonModelProvider, WeaviateQuery, TritonModelUtils
```

**Using existing adapters:**
```python
import tritonclient.grpc as TritonClient

# Initialize clients
weaviate_client = WeaviateAdapter.init_client(host="127.0.0.1", port="8080")
triton_client = TritonClient.InferenceServerClient(url="triton:8001")

# Create adapters
vector_db = WeaviateAdapter(
    weaviate_client=weaviate_client,
    triton_client=triton_client
)
model_provider = TritonModelProvider(triton_client=triton_client)
```

**Creating new adapters:**

For a new vector database, implement `VectorDBAdapter` and `Query`:
```python
from framework.interfaces import VectorDBAdapter, Query, QueryResult
import pandas as pd

class MyQuery(Query):
    def query(self, near_text, collection_name, limit=25, query_method="default", **kwargs):
        # Your query implementation
        # query_method can be "vector", "keyword", "hybrid", etc.
        return pd.DataFrame(results)

class MyVectorDBAdapter(VectorDBAdapter):
    def __init__(self, client):
        self.client = client
        self.query_instance = MyQuery(client)
    
    @classmethod
    def init_client(cls, **kwargs):
        # Initialize your vector DB client
        return client
    
    def search(self, query, collection_name, limit=25, query_method="default", **kwargs):
        df = self.query_instance.query(query, collection_name, limit, query_method, **kwargs)
        return QueryResult(df.to_dict('records'))
    
    # Implement other required methods...
```

For a new model provider, implement `ModelProvider` and optionally `ModelUtils`:
```python
from framework.interfaces import ModelProvider
from framework.model_utils import ModelUtils

class MyModelUtils(ModelUtils):
    def calculate_embedding(self, text, image=None, model_name="default"):
        # Your embedding implementation
        return embedding
    
    def generate_caption(self, image, model_name="default"):
        # Your caption generation
        return caption

class MyModelProvider(ModelProvider):
    def __init__(self):
        self.model_utils = MyModelUtils()
    
    def get_embedding(self, text, image=None, model_name="default"):
        return self.model_utils.calculate_embedding(text, image, model_name)
    
    def generate_caption(self, image, model_name="default"):
        return self.model_utils.generate_caption(image, model_name)
```

3. **Create main.py**:

```python
import sys
import os
import logging
import tritonclient.grpc as TritonClient

# Add framework and adapters to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../framework'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../adapters'))

from framework.evaluator import BenchmarkEvaluator
from adapters import WeaviateAdapter, TritonModelProvider
from dataset_loader import MyDatasetLoader

# Environment variables
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
TRITON_HOST = os.getenv("TRITON_HOST", "triton")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_collection")
QUERY_METHOD = os.getenv("QUERY_METHOD", "clip_hybrid_query")  # Query method for Weaviate

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize clients
    weaviate_client = WeaviateAdapter.init_client(host=WEAVIATE_HOST)
    triton_client = TritonClient.InferenceServerClient(url=f"{TRITON_HOST}:8001")
    
    # Create adapters
    vector_db = WeaviateAdapter(
        weaviate_client=weaviate_client,
        triton_client=triton_client
    )
    model_provider = TritonModelProvider(triton_client=triton_client)
    
    # Create dataset loader
    dataset_loader = MyDatasetLoader()
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(
        vector_db=vector_db,
        model_provider=model_provider,
        dataset_loader=dataset_loader,
        collection_name=COLLECTION_NAME,
        query_method=QUERY_METHOD  # Passed to Query.query() method
    )
    
    # Run evaluation
    try:
        results, stats = evaluator.evaluate_queries(split="test")
        results.to_csv("results.csv", index=False)
        stats.to_csv("metrics.csv", index=False)
    finally:
        vector_db.close()

if __name__ == "__main__":
    main()
```

## Makefile System

The framework provides a reusable Makefile system for building, deploying, and managing benchmarks.

### Structure

- **Base Makefile** (`benchmarks/Makefile`): Contains generic commands for all benchmarks
- **Benchmark Makefiles** (e.g., `benchmarks/INQUIRE/Makefile`): Set benchmark-specific variables and include the base

### Common Commands

All benchmarks support these commands:

- `make build` - Build Docker images
- `make deploy` - Deploy to Kubernetes
- `make load` - Start data loader
- `make calculate` - Run benchmark evaluation
- `make get` - Copy results from pod
- `make status` - Show deployment status
- `make logs-evaluator` / `make logs-data-loader` - View logs
- `make clean` - Remove deployments and PVCs

### Creating a Benchmark Makefile

Each benchmark needs a Makefile that:
1. Sets required variables (`BENCHMARK_NAME`, `KUSTOMIZE_DIR`, etc.)
2. Includes the base Makefile: `include ../Makefile`

See `benchmarks/MAKEFILE.md` for detailed documentation and examples.

## Dockerfile System

The framework provides a Dockerfile template for consistent container builds across benchmarks.

### Structure

- **Template Directory** (`benchmarks/template/`): Complete template for creating new benchmarks
- **Template Files**: `Dockerfile.template`, `Dockerfile.benchmark`, `Dockerfile.data_loader`
- **Benchmark Dockerfiles**: Each benchmark creates `Dockerfile.benchmark` and `Dockerfile.data_loader` based on the template

### Creating Benchmark Dockerfiles

1. Copy the template directory: `cp -r benchmarks/template benchmarks/MYBENCHMARK`
2. Customize the `CMD` line for your entrypoint script
3. Adjust `PYTHONPATH` if your structure differs

See `benchmarks/DOCKER.md` for detailed documentation and examples.

## Template Directory

The `benchmarks/template/` directory provides a complete starting point for new benchmarks:

- **README.md**: Comprehensive guide for creating new benchmarks
- **Makefile**: Template Makefile with all required variables
- **Dockerfile.benchmark** & **Dockerfile.data_loader**: Ready-to-use Dockerfiles
- **Python Templates**: Template files for `main.py`, `load_data.py`, `dataset_loader.py`
- **requirements.txt**: Base dependencies

### Quick Start

```bash
cd benchmarking
cp -r benchmarks/template benchmarks/MYBENCHMARK
cd MYBENCHMARK
# Follow instructions in README.md
```

See `benchmarks/template/README.md` for complete setup instructions.

## Using the Framework

### Basic Usage Pattern

All benchmarks follow this pattern:

1. **Import adapters**:
   ```python
   from adapters import WeaviateAdapter, TritonModelProvider
   ```

2. **Initialize clients**:
   ```python
   weaviate_client = WeaviateAdapter.init_client(host="127.0.0.1", port="8080")
   triton_client = TritonClient.InferenceServerClient(url="triton:8001")
   ```

3. **Create adapters**:
   ```python
   vector_db = WeaviateAdapter(weaviate_client=weaviate_client, triton_client=triton_client)
   model_provider = TritonModelProvider(triton_client=triton_client)
   ```

4. **Create dataset loader** (benchmark-specific):
   ```python
   from dataset_loader import MyDatasetLoader
   dataset_loader = MyDatasetLoader()
   ```

5. **Create evaluator and run**:
   ```python
   evaluator = BenchmarkEvaluator(
       vector_db=vector_db,
       model_provider=model_provider,
       dataset_loader=dataset_loader,
       collection_name="my_collection",
       query_method="clip_hybrid_query"  # Query type for WeaviateQuery
   )
   results, stats = evaluator.evaluate_queries(split="test")
   ```

### Query Method Parameter

The `query_method` parameter in `BenchmarkEvaluator` is passed to the `Query.query()` method:

- **For Weaviate**: `query_method` can be `"clip_hybrid_query"`, `"hybrid_query"`, or `"colbert_query"`
- **For other vector DBs**: Implement your own query types in your `Query` implementation
- The `Query.query()` method routes to the appropriate implementation based on `query_method`

### Model Names

The `ModelProvider` and `ModelUtils` interfaces accept `model_name` parameters:

- **Embedding models**: `"clip"`, `"colbert"`, `"align"` (for TritonModelProvider)
- **Caption models**: `"gemma3"`, `"qwen2_5"` (for TritonModelProvider)
- Other implementations can define their own model names

## Existing Benchmarks

### INQUIRE

- **Location**: `benchmarks/INQUIRE/`
- **Dataset**: INQUIRE benchmark for natural world image retrieval (using Weaviate)
- **Usage**: See `benchmarks/INQUIRE/Readme.md`
- **Example**: Complete working implementation using the framework

## Adding New Components

### Adding a New Vector Database

1. **Create a Query class** implementing the `Query` interface:
   ```python
   # adapters/myvectordb.py
   from framework.interfaces import Query
   import pandas as pd
   
   class MyVectorDBQuery(Query):
       def query(self, near_text, collection_name, limit=25, query_method="vector", **kwargs):
           # Implement your query logic
           # query_method can be "vector", "keyword", "hybrid", etc.
           return pd.DataFrame(results)
   ```

2. **Create an adapter** implementing `VectorDBAdapter`:
   ```python
   # adapters/myvectordb.py
   from framework.interfaces import VectorDBAdapter, QueryResult
   
   class MyVectorDBAdapter(VectorDBAdapter):
       @classmethod
       def init_client(cls, **kwargs):
           # Initialize your vector DB client
           return client
       
       def __init__(self, client=None, **kwargs):
           if client is None:
               client = self.init_client(**kwargs)
           self.client = client
           self.query_instance = MyVectorDBQuery(client)
       
       def search(self, query, collection_name, limit=25, query_method="vector", **kwargs):
           df = self.query_instance.query(query, collection_name, limit, query_method, **kwargs)
           return QueryResult(df.to_dict('records'))
       
       # Implement other required methods...
   ```

3. **Export in `adapters/__init__.py`**:
   ```python
   from .myvectordb import MyVectorDBAdapter, MyVectorDBQuery
   __all__ = [..., 'MyVectorDBAdapter', 'MyVectorDBQuery']
   ```

4. **Use in any benchmark's `main.py`**:
   ```python
   from adapters import MyVectorDBAdapter
   ```

### Adding a New Model Provider

1. **Create ModelUtils implementation** (optional but recommended):
   ```python
   # adapters/mymodel.py
   from framework.model_utils import ModelUtils
   
   class MyModelUtils(ModelUtils):
       def calculate_embedding(self, text, image=None, model_name="default"):
           # Your embedding implementation
           return embedding
       
       def generate_caption(self, image, model_name="default"):
           # Your caption generation
           return caption
   ```

2. **Create ModelProvider**:
   ```python
   # adapters/mymodel.py
   from framework.interfaces import ModelProvider
   
   class MyModelProvider(ModelProvider):
       def __init__(self, **kwargs):
           self.model_utils = MyModelUtils(**kwargs)
       
       def get_embedding(self, text, image=None, model_name="default"):
           return self.model_utils.calculate_embedding(text, image, model_name)
       
       def generate_caption(self, image, model_name="default"):
           return self.model_utils.generate_caption(image, model_name)
   ```

3. **Export in `adapters/__init__.py`** and use in benchmarks

### Adding a New Dataset

1. **Create loader** implementing `DatasetLoader`:
   ```python
   # benchmarks/MYBENCHMARK/dataset_loader.py
   from framework.interfaces import DatasetLoader
   import pandas as pd
   
   class MyDatasetLoader(DatasetLoader):
       def load(self, split="test", **kwargs) -> pd.DataFrame:
           # Load your dataset
           return dataset_df
       
       def get_query_column(self) -> str:
           return "query"
       
       # Implement other required methods...
   ```

2. **Place in your benchmark's directory** (not in `adapters/`)

3. **Use in your benchmark's `main.py`**:
   ```python
   from dataset_loader import MyDatasetLoader
   ```

## How It Works

### Abstract Interface Pattern

The framework uses abstract interfaces to ensure consistency and extensibility:

1. **Framework defines interfaces** (`framework/interfaces.py`, `framework/model_utils.py`):
   - `VectorDBAdapter`, `ModelProvider`, `Query`, `ModelUtils`, `DatasetLoader`, etc.
   - These define the contract that all implementations must follow

2. **Adapters implement interfaces** (`adapters/`):
   - `TritonModelUtils` implements `ModelUtils`
   - `TritonModelProvider` implements `ModelProvider` and uses `TritonModelUtils`
   - `WeaviateQuery` implements `Query`
   - `WeaviateAdapter` implements `VectorDBAdapter` and uses `WeaviateQuery`

3. **Benchmarks use adapters**:
   - Import from `adapters/` (e.g., `from adapters import WeaviateAdapter, TritonModelProvider`)
   - Use the abstract interfaces, not concrete implementations
   - Easy to swap implementations without changing benchmark code

### Example: Using the Query Interface

```python
from adapters import WeaviateAdapter, WeaviateQuery

# WeaviateQuery implements the Query interface
query_instance = WeaviateQuery(weaviate_client, triton_client)

# Use the generic query() method
results = query_instance.query(
    near_text="search query",
    collection_name="my_collection",
    limit=25,
    query_method="clip_hybrid_query"  # Weaviate-specific query type
)

# Or use Weaviate-specific methods directly
results = query_instance.clip_hybrid_query("search query", "my_collection", limit=25)
```

### Example: Using the ModelUtils Interface

```python
from adapters import TritonModelProvider, TritonModelUtils

# TritonModelUtils implements the ModelUtils interface
model_utils = TritonModelUtils(triton_client)

# Use the abstract methods
embedding = model_utils.calculate_embedding("text", image=None, model_name="clip")
caption = model_utils.generate_caption(image, model_name="gemma3")

# Or use via ModelProvider
model_provider = TritonModelProvider(triton_client)
embedding = model_provider.get_embedding("text", image=None, model_name="clip")
caption = model_provider.generate_caption(image, model_name="gemma3")
```

## Benefits

1. **Reusability**: Framework code shared across all benchmarks
2. **Consistency**: Same evaluation metrics and methodology
3. **Maintainability**: Single source of truth for evaluation logic
4. **Extensibility**: Easy to add new components without modifying framework
5. **Testability**: Clear interfaces make testing easier
6. **Independence**: No dependency on `app/` - all functions in framework/adapters
7. **Type Safety**: Abstract interfaces ensure all implementations provide required functionality
8. **Flexibility**: Each implementation can define its own query types and model names

