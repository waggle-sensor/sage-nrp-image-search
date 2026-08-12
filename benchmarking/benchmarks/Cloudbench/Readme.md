# CloudBench Benchmark

This benchmark uses [CloudBench](https://huggingface.co/datasets/sagecontinuum/CloudBench) with Milvus (default) or Weaviate as the vector database for evaluating text-to-image retrieval in cloud and atmospheric science. CloudBench is a benchmark dataset for cloud image retrieval: natural language queries paired with images and binary relevance labels.

## Dataset

- **Source**: [sagecontinuum/CloudBench](https://huggingface.co/datasets/sagecontinuum/CloudBench) on Hugging Face
- **Contents**: Query–image pairs with relevance labels (0 = not relevant, 1 = relevant), plus metadata (cloud_coverage, viewpoint, lighting, confounder_type, occlusion_present, multiple_cloud_types, horizon_visible, ground_visible, sun_visible, precipitation_visible, overcast, multiple_layers, storm_visible)
- **Split**: The dataset provides a single `train` split (~4.6k rows)

## Usage

This benchmark is intended to be used with [Sage Image Search](../../../kubernetes/base/). The Makefile references components deployed there and runs the CloudBench benchmark job.

## Running the Benchmark

### Prerequisites

- **Kubernetes cluster** access with `kubectl` configured
- **kustomize** (or kubectl with kustomize support)
- **Docker** for building images
- **Triton** deployed and NRP Milvus credentials. Weaviate is optional (`VECTOR_DB=weaviate`).

### Steps

1. **Deploy Sage Image Search infrastructure** (from the main `kubernetes` directory):
   ```bash
   kubectl apply -k nrp-dev   # or nrp-prod
   ```

2. **Build and push the benchmark image**:
   ```bash
   cd benchmarking/benchmarks/Cloudbench
   make build
   docker push <registry>/benchmark-cloudbench-job:latest
   ```

3. **Run the CloudBench benchmark** (loads data and evaluates):
   ```bash
   make run   # defaults to dev environment
   make logs  # monitor progress
   ```
   This loads `sagecontinuum/CloudBench` into the configured vector database, runs the evaluation, and saves results.

4. **Run locally (development)**:
   ```bash
   make run-local
   ```
   Uses port-forwarding to Triton (and Weaviate if `VECTOR_DB=weaviate`). Pass `MILVUS_TOKEN` for NRP Milvus.

### Results

After a run, three files are produced:

- **`image_search_results.csv`**: Metadata of images returned for each query
- **`query_eval_metrics.csv`**: Evaluation metrics (NDCG, precision, recall, etc.) per query
- **`config_values.csv`**: Configuration used for the run (`config.to_csv()`)

Results are written to `/app/results` in Kubernetes (with a volume mount) or to the current directory when using `make run-local`. Optional S3 upload uses paths like `{S3_PREFIX}/{timestamp}/{filename}`.

#### Leaderboard

Cloudbench leaderboard is supported for benchmark results in `benchmarks/Cloudbench/results/leaderboard.ipynb`. Check out the notebook for more details.

## Environment Variables

- **CLOUDBENCH_DATASET**: HuggingFace dataset name (default: `sagecontinuum/CloudBench`)
- **COLLECTION_NAME**: collection name (default: `CloudBench`)
- **VECTOR_DB**: `milvus` (default) or `weaviate`
- **SAMPLE_SIZE**: Number of samples (0 = use full dataset)
- **SEED**, **HF_TOKEN**, **WORKERS**, **IMAGE_BATCH_SIZE**, **QUERY_BATCH_SIZE**: Data and processing
- **QUERY_METHOD**, **TARGET_VECTOR**, **RESPONSE_LIMIT**: Query and retrieval
- See `config.py` for the full list (Weaviate, Triton, S3, etc.).

## Citation

If you use CloudBench, cite the dataset:

```bibtex
@misc{cloudbench_2026,
    author       = { Sage Continuum and Francisco Lozano },
    affiliation  = { Northwestern University },
    title        = { CloudBench },
    year         = 2026,
    url          = { https://huggingface.co/datasets/sagecontinuum/CloudBench },
    doi          = { 10.57967/hf/7784 },
    publisher    = { Hugging Face }
}
```

## References

- [CloudBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/CloudBench)
- [Weaviate: NDCG and retrieval evaluation](https://weaviate.io/blog/retrieval-evaluation-metrics#normalized-discounted-cumulative-gain-ndcg)
- [imsearch_eval](https://github.com/waggle-sensor/imsearch_eval) framework
