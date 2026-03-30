# Sagebench Benchmark

This benchmark evaluates **image retrieval** on the Hugging Face dataset
`sagecontinuum/SageBench`, using the `imsearch_eval` framework for end-to-end evaluation. This benchmakr focuses on retrieving images using Sage metadata-aware queries.

## Dataset

- Source: [`sagecontinuum/SageBench`](https://huggingface.co/datasets/sagecontinuum/SageBench)
- Split: `train`
- Notes: queries are metadata-aware (e.g., `vsn`, `zone`, `job`, `camera`, `address`), and the dataset provides binary `relevance_label`.

## Running the Benchmark

### Kubernetes

1. Deploy the Sage Image Search infrastructure (from the main `kubernetes` directory):

   ```bash
   kubectl apply -k benchmarking/kubernetes/nrp-dev
   ```

2. Build and run the Sagebench benchmark job:

   ```bash
   cd benchmarking/benchmarks/Sagebench
   make build
   make run
   ```

3. Monitor logs:

   ```bash
   make logs
   ```

### Local (development)

```bash
cd benchmarking/benchmarks/Sagebench
make run-local SAMPLE_SIZE=50
```

## Outputs

The job writes these files to `/app/results`:

- `image_search_results.csv`
- `query_eval_metrics.csv`
- `config_values.csv`

If `UPLOAD_TO_S3=true`, results are uploaded to `S3_PREFIX/{timestamp}/`.

## Key Environment Variables

- `SAGEBENCH_DATASET` (default `sagecontinuum/SageBench`)
- `SAMPLE_SIZE` (default `0`)
- `SEED` (default `42`)
- `HF_TOKEN` (only needed if the dataset is private)
- `COLLECTION_NAME` (default `Sagebench`)
- `S3_PREFIX` (default `dev-metrics/sagebench` for dev)
- `RESPONSE_LIMIT` (default `50`)
>NOTE: look at config.py for more environment variables.
