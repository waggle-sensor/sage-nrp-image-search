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
KUSTOMIZE_DIR := ../kubernetes/MYBENCHMARK  # Change this!
```

## Step 3: Rename Template Files

```bash
mv benchmark_dataset.py.template benchmark_dataset.py
mv main.py.template main.py
mv load_data.py.template load_data.py
```

## Step 4: Implement BenchmarkDataset

Edit `benchmark_dataset.py` and implement:
- `load()` - Load your dataset
- `get_query_column()` - Return query column name
- `get_query_id_column()` - Return query ID column name
- `get_relevance_column()` - Return relevance column name

## Step 5: Update main.py

Edit `main.py` and:
- Import your `BenchmarkDataset` class
- Update `COLLECTION_NAME` default
- Update `RESULTS_FILE` and `METRICS_FILE` names

## Step 6: Create Kubernetes Config

```bash
cd ../../kubernetes
cp -r ../benchmarks/template/kubernetes MYBENCHMARK
cd MYBENCHMARK
# Replace MYBENCHMARK with your benchmark name
find . -type f -name "*.yaml" -exec sed -i '' 's/MYBENCHMARK/mybenchmark/g' {} +
# Update image names in kustomization.yaml
# Update environment variables in env.yaml and data-loader-env.yaml
```

## Step 7: Deploy

```bash
make build    # Build images or use the github actions to build and push the images to the registry. See `.github/workflows/benchmarking.yml` for more details.
make deploy   # Deploy to Kubernetes
make load     # Load data
make calculate # Run evaluation
make get      # Get results
```

## Files to Customize

| File | What to Change |
|------|----------------|
| `Makefile` | Benchmark name, kustomize dir, result files |
| `benchmark_dataset.py` | Implement benchmark dataset logic |
| `main.py` | Update collection name, result file names |
| `load_data.py` | Implement data loading logic (if needed) |
| `Dockerfile.benchmark` | Usually no changes needed |
| `Dockerfile.data_loader` | Usually no changes needed |
| `requirements.txt` | Add your dependencies |

## Need Help?

- See `README.md` for detailed instructions
- Check `../INQUIRE/` for a complete example (same directory level)
- Review `../README.md` for framework overview

