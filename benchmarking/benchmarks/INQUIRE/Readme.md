# INQUIRE Benchmark

Here we use [INQUIRE](https://github.com/inquire-benchmark/INQUIRE) with Weaviate as the vector database for benchmarking. Different models were used to generate captions and keywords for the images. Also different models were used to generate the embeddings for the images.

## Usage

This benchmark is supposed to be used in conjuction with [Sage Image Search](../../../kubernetes/base/). The Makefile references components that are deployed in [Sage Image Search](../../../kubernetes/base/) and deploys additional containers that are used to run the INQUIRE Benchmark.

## Running the Example

### Prerequisites
To run this example, you'll need:
- **Kubernetes cluster** access with `kubectl` configured
- **kustomize** (or kubectl with kustomize support)
- **Docker** for building images
- **Weaviate and Triton** deployed (from `kubernetes/nrp-dev` or `kubernetes/nrp-prod` depending on the environment you want to use)

### Step-by-Step Setup

1. **Deploy Sage Image Search Infrastructure**:
   - Navigate to the main [kubernetes](../../../kubernetes) directory and deploy base services:
     ```bash
     kubectl apply -k nrp-dev or nrp-prod
     ```

2. **Build and Push Images**:
   - Build the benchmark images:
     ```bash
     cd benchmarking/benchmarks/INQUIRE
     make build
     ```
   - Push to registry (update registry in Makefile if needed):
     ```bash
     docker push <registry>/benchmark-inquire-evaluator:latest
     docker push <registry>/benchmark-inquire-data-loader:latest
     ```
>NOTE: You can also use the github actions to build and push the images to the registry. See `.github/workflows/benchmarking.yml` for more details.

3. **Deploy INQUIRE Benchmark**:
   - Deploy to Kubernetes:
     ```bash
     make deploy
     ```

4. **Load in the dataset**:
   - Start the data loader:
     ```bash
     make load
     ```
   - Monitor progress:
     ```bash
     make logs-data-loader
     ```
   >NOTE: This loads in [INQUIRE-Benchmark-small](https://huggingface.co/datasets/sagecontinuum/INQUIRE-Benchmark-small) into Weaviate for the INQUIRE benchmark.

5. **Calculate the Query Metrics**:
   - After dataset is fully loaded into Weaviate, run:
     ```bash
     make calculate
     ```
   - Monitor progress:
     ```bash
     make logs-evaluator
     ```
   >NOTE: Data loader logs will indicate when the dataset is fully loaded into Weaviate.

6. **Retrieve the Results**:
   - After the metrics are calculated, run:
     ```bash
     make get
     ```
   >NOTE: This will copy the csv files into your current working directory from the pod
   >Alternatively, results are stored in the `inquire-results-pvc` PVC

### Results

Once the benchmark is ran, two csv files will be generated:
- `image_search_results.csv`
    - This file includes the metadata of all images returned by Weaviate when different queries were being ran.
- `query_eval_metrics.csv`
    - This file includes the calculated metrics based on images returned by different queries.

There is multiple results placed in version folders. Each folder has a evaluate.ipynb notebook that goes into more details what that version tested and the metrics.

## References
- [Weaviate Blog: NDCG](https://weaviate.io/blog/retrieval-evaluation-metrics#normalized-discounted-cumulative-gain-ndcg)
- [RAG Evaluation](https://weaviate.io/blog/rag-evaluation)
- [Scikit-Learn NDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [A Guide on NDCG](https://www.aporia.com/learn/a-practical-guide-to-normalized-discounted-cumulative-gain-ndcg/)
- [Weaviate: Batch import](https://weaviate.io/developers/weaviate/manage-data/import)
- [Weaviate: Imports in Detail](https://weaviate.io/developers/weaviate/tutorials/import#data-import---best-practices)
- [INQUIRE](https://inquire-benchmark.github.io/)
- [Hugginface: Fine-tuning Florence2](https://huggingface.co/blog/finetune-florence2)
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

