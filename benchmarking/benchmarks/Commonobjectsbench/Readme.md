# CommonObjectsBench Benchmark

This benchmark uses [CommonObjectsBench](https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench) with Milvus (default) or Weaviate as the vector database for evaluating text-to-image retrieval on general objects and common scenes. CommonObjectsBench is a benchmark dataset for general object image retrieval: natural language queries paired with images and binary relevance labels.

## Dataset

- **Source (public)**: [sagecontinuum/CommonObjectsBench](https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench) on Hugging Face
- **Source (private)**: [sagecontinuum/CommonObjectsBench-private](https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench-private) — same schema, includes images that must be kept private; requires `HF_TOKEN`
- **Contents**: Query–image pairs with relevance labels (0 = not relevant, 1 = relevant), plus metadata (viewpoint, lighting, environment_type, urban_scene, rural_scene, outdoor_scene, person_present, animal_present, food_present, vehicle_present, etc.)
- **Split**: The dataset provides a single `train` split (~12k rows)

## Usage

This benchmark is intended to be used with [Sage Image Search](../../../kubernetes/base/). The Makefile references components deployed there and runs the CommonObjectsBench benchmark job.

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
   cd benchmarking/benchmarks/Commonobjectsbench
   make build
   docker push <registry>/benchmark-commonobjectsbench-job:latest
   ```
   See `.github/workflows/benchmarking.yml` for CI build/push.

3. **Run the CommonObjectsBench benchmark** (loads data and evaluates):
   ```bash
   make run   # defaults to dev environment
   make logs  # monitor progress
   ```
   This loads the selected CommonObjectsBench dataset (public or private) into the configured vector database, runs the evaluation, and saves results.

4. **Run locally (development)**:
   ```bash
   make run-local
   ```
   Uses port-forwarding to Triton (and Weaviate if `VECTOR_DB=weaviate`). Pass `MILVUS_TOKEN` for NRP Milvus. To use the private dataset locally, set `COMMONOBJECTSBENCH_USE_PRIVATE=true` and `HF_TOKEN`.

### Results

After a run, three files are produced:

- **`image_search_results.csv`**: Metadata of images returned for each query
- **`query_eval_metrics.csv`**: Evaluation metrics (NDCG, precision, recall, etc.) per query
- **`config_values.csv`**: Configuration used for the run (`config.to_csv()`)

Results are written to `/app/results` in Kubernetes (with a volume mount) or to the current directory when using `make run-local`. Optional S3 upload uses paths like `{S3_PREFIX}/{timestamp}/{filename}`.

#### Leaderboard

Commonobjectsbench leaderboard is supported for benchmark results in `benchmarks/Commonobjectsbench/results/leaderboard.ipynb`. Check out the notebook for more details.

## Environment Variables

- **COMMONOBJECTSBENCH_USE_PRIVATE**: Set to `"true"` to use the private dataset `sagecontinuum/CommonObjectsBench-private`; otherwise uses the public dataset `sagecontinuum/CommonObjectsBench` (default: `false`)
- **COLLECTION_NAME**: collection name (default: `CommonObjectsBench`)
- **VECTOR_DB**: `milvus` (default) or `weaviate`
- **SAMPLE_SIZE**: Number of samples (0 = use full dataset)
- **SEED**, **HF_TOKEN**, **WORKERS**, **IMAGE_BATCH_SIZE**, **QUERY_BATCH_SIZE**: Data and processing
- **QUERY_METHOD**, **TARGET_VECTOR**, **RESPONSE_LIMIT**: Query and retrieval
- See `config.py` for the full list (Weaviate, Triton, S3, etc.).

## Citation

If you use CommonObjectsBench, cite the dataset:

```bibtex
@misc{commonobjectsbench_2026,
  author       = { Sage Continuum and Francisco Lozano },
  affiliation  = { Northwestern University },
  title        = { CommonObjectsBench },
  year         = { 2026 },
  url          = { https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench },
  doi          = { 10.57967/hf/7728 },
  publisher    = { Hugging Face }
}
```

## References

- [CommonObjectsBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench)
- [Weaviate: NDCG and retrieval evaluation](https://weaviate.io/blog/retrieval-evaluation-metrics#normalized-discounted-cumulative-gain-ndcg)
- [imsearch_eval](https://github.com/waggle-sensor/imsearch_eval) framework
