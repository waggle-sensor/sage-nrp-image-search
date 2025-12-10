# INQUIRE_WEAV Benchmark

Here we use [INQUIRE](https://github.com/inquire-benchmark/INQUIRE) with Weaviate as the vector database for benchmarking. Different models were used to generate captions and keywords for the images. Also different models were used to generate the embeddings for the images.

## New Framework (Recommended)

**The benchmarking code has been refactored to use an abstract framework that reuses code from `app/` and `weavloader/` without duplication.**

- **New entry point**: `main.py` (uses abstract framework)
- **Framework**: `framework/` (abstract interfaces and evaluation logic)
- **Adapters**: `adapters/` (concrete implementations reusing app code)
- **Documentation**: See `README_FRAMEWORK.md` for details

The new framework:
- ✅ Independent implementations in `adapters/` (no dependency on `app/`)
- ✅ Abstract interfaces in `framework/` for extensibility
- ✅ Supports multiple vector databases and models
- ✅ Easy to extend with new implementations
- ✅ Uses `Query` interface for query classes
- ✅ Uses `ModelUtils` interface for model utilities

> **Note**: The old benchmarking code in `app/` is deprecated. Use `main.py` instead.

## Usage

This benchmark is supposed to be used in conjuction with [Hybrid Search](../HybridSearch_example/). The Makefile references components that are deployed in [Hybrid Search](../HybridSearch_example/). The Makefile in here deploys additional containers that are used to run the INQUIRE_WEAV Benchmark.

## Running the Example

### Prerequisites
To run this example, you'll need:
- **Kubernetes cluster** access with `kubectl` configured
- **kustomize** (or kubectl with kustomize support)
- **Docker** for building images
- **Weaviate and Triton** deployed (from main `kubernetes/` directory)

### Step-by-Step Setup

1. **Deploy Hybrid Search Infrastructure**:
   - Navigate to the main `kubernetes/` directory and deploy base services:
     ```bash
     kubectl apply -k base
     ```
   - Or use the nrp-dev/nrp-prod overlays as needed

2. **Build and Push Images**:
   - Build the benchmark images:
     ```bash
     cd benchmarking/benchmarks/INQUIRE_WEAV
     make build
     ```
   - Push to registry (update registry in Makefile if needed):
     ```bash
     docker push <registry>/benchmark-inquire-weav-evaluator:latest
     docker push <registry>/benchmark-inquire-weav-data-loader:latest
     ```

3. **Deploy INQUIRE_WEAV Benchmark**:
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
   >NOTE: This loads in [INQUIRE-Benchmark-small](https://huggingface.co/datasets/sagecontinuum/INQUIRE-Benchmark-small) into Weaviate for the INQUIRE_WEAV benchmark.

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
   >Alternatively, results are stored in the `inquire-weav-results-pvc` PVC

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
