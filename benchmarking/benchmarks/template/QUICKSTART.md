# Quick Start Guide

Create a new benchmark in 5 minutes!

## Step 1: Copy Template

```bash
cd benchmarking/benchmarks
cp -r template MYBENCHMARK
cd MYBENCHMARK
```

## Step 2: Update Makefile

Edit `Makefile` and replace `mybenchmark` with your benchmark name:

```makefile
BENCHMARK_NAME := mybenchmark  # Change this!
DOCKERFILE_JOB := Dockerfile.job
RESULTS_FILES := image_search_results.csv query_eval_metrics.csv
ENV ?= dev
ifeq ($(ENV),prod)
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-prod
else
  KUSTOMIZE_DIR := ../../kubernetes/MYBENCHMARK/nrp-dev
endif
```

## Step 3: Rename Template Files

```bash
mv benchmark_dataset.template.py benchmark_dataset.py
mv run_benchmark.template.py run_benchmark.py
# config.py is already named correctly, just customize it
```

## Step 4: Update config.py

Edit `config.py` and:
- Replace `MYBENCHMARK` with your benchmark name
- Update default values for your dataset, collection name, etc.
- Add any benchmark-specific hyperparameters

## Step 5: Implement BenchmarkDataset

Edit `benchmark_dataset.py` and implement:
- `load()` - Load your dataset
- `get_query_column()` - Return query column name
- `get_query_id_column()` - Return query ID column name
- `get_relevance_column()` - Return relevance column name

## Step 6: Update run_benchmark.py

Edit `run_benchmark.py` and:
- Import your `BenchmarkDataset` class (replace `MyBenchmarkDataset`)
- Import your `Config` class (replace `MyConfig`)
- Update the config instance creation
- Implement the `load_data(vector_db, model_provider)` function:
  - Load your dataset
  - Create collection schema
  - Process and insert data into vector database
- Implement the `run_evaluation(vector_db, model_provider)` function:
  - Create evaluator
  - Run evaluation queries
  - Return results

See `../INQUIRE/run_benchmark.py` for a complete example.

## Step 7: Create Kubernetes Config

```bash
cd ../../kubernetes
cp -r ../benchmarks/template/kubernetes MYBENCHMARK
cd MYBENCHMARK
# Replace MYBENCHMARK with your benchmark name
find . -type f -name "*.yaml" -exec sed -i '' 's/MYBENCHMARK/mybenchmark/g' {} +
# Update image name in nrp-dev/kustomization.yaml and nrp-prod/kustomization.yaml
# Update environment variables in nrp-dev/env.yaml and nrp-prod/env.yaml
```

## Step 8: Update config.py (if needed)

If your config needs Weaviate connection parameters, ensure they're in your config:
- `WEAVIATE_HOST`
- `WEAVIATE_PORT`
- `WEAVIATE_GRPC_PORT`

These are typically set via environment variables in Kubernetes.

## Step 9: Deploy

```bash
cd ../../benchmarks/MYBENCHMARK
make build    # Build image or use GitHub Actions to build and push to registry
make run      # Deploy and run benchmark job (loads data and evaluates)
make logs     # Monitor logs
```

## Files to Customize

| File | What to Change |
|------|----------------|
| `Makefile` | Benchmark name, kustomize dir, result files |
| `config.py` | Update with your benchmark name, dataset, and configuration values |
| `benchmark_dataset.py` | Implement benchmark dataset logic |
| `run_benchmark.py` | Import your classes, implement `load_data()` and `run_evaluation()` functions |
| `Dockerfile.job` | Usually no changes needed |
| `requirements.txt` | Add your dependencies |
| `kubernetes/nrp-dev/env.yaml` | Update environment variables for dev |
| `kubernetes/nrp-prod/env.yaml` | Update environment variables for prod |

## Need Help?

- See `README.md` for detailed instructions
- Check `../INQUIRE/` for a complete example (same directory level)
- Review `../README.md` for framework overview
