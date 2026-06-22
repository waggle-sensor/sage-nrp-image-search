# Benchmarking

Sage Image Search includes a benchmarking suite to measure retrieval quality across domain-specific datasets. Benchmarks use the external [`imsearch_eval`](https://github.com/waggle-sensor/imsearch_eval) framework.

## Available benchmarks

| Benchmark | Dataset | Domain | Size |
|-----------|---------|--------|------|
| **Sagebench** | `sagecontinuum/SageBench` | Sage edge-camera metadata queries | Metadata-aware |
| **Firebench** | `sagecontinuum/FireBench` | Wildfire / fire science | ~4k query–image pairs |
| **Cloudbench** | `sagecontinuum/CloudBench` | Cloud / atmospheric science | ~4.6k pairs |
| **Commonobjectsbench** | `sagecontinuum/CommonObjectsBench` | General objects and scenes | ~12k pairs |
| **INQUIRE** | `sagecontinuum/INQUIRE-Benchmark-small` | Natural-world retrieval | Adapted from INQUIRE |

Each benchmark lives in `benchmarking/benchmarks/{Benchmark}/` with its own `config.py`, `run_benchmark.py`, `Dockerfile.job`, and `Makefile`.

## How benchmarks work

Each benchmark job:

1. Connects to a running Weaviate + Triton deployment
2. **Indexes** dataset images (caption generation, CLIP embeddings, Weaviate insert)
3. **Evaluates** all queries using `BenchmarkEvaluator.evaluate_queries()`
4. **Saves** three CSV files to `/app/results` (K8s) or the local directory
5. Optionally uploads results to S3/MinIO (`UPLOAD_TO_S3=true`)

## Running a benchmark

**Prerequisites:** Weaviate and Triton must be deployed (see [Getting Started](getting-started.md)).

From a benchmark directory (e.g. `benchmarking/benchmarks/INQUIRE/`):

```bash
make build        # Build Docker image
make run          # Deploy K8s Job on NRP (default: dev overlay)
make logs         # Tail job logs
make status       # Job/pod status
make down         # Remove deployment
```

**Local development:**

```bash
make run-local    # Port-forwards Weaviate/Triton, runs benchmark locally
```

Set `ENV=prod` to use the prod Kubernetes overlay instead of dev.

## Ablation studies

Toggle pipeline components via environment variables to measure their individual contribution:

- `ENABLE_CAPTION_GENERATION` — enable/disable VLM captioning
- `EMBED_IMAGE` / `EMBED_CAPTION` — control which embeddings are computed
- `ENABLE_BM25` — enable/disable keyword search component

Use a distinct `COLLECTION_NAME` per experiment. See [`benchmarking/helpers/ablation.py`](../benchmarking/helpers/ablation.py) and [`benchmarking/README.md`](../benchmarking/README.md).

## Results

Each run produces three CSV files:

| File | Contents |
|------|----------|
| `image_search_results.csv` | Retrieved images per query |
| `query_eval_metrics.csv` | Per-query metrics (precision, recall, NDCG, MRR, hit, diversity) |
| `config_values.csv` | Full run configuration snapshot |

Results are stored in `benchmarking/benchmarks/{Benchmark}/results/` organized by version (`baseline`, `v10`, `v11`, `ablation_*`, etc.).

## Primary metrics

At K=25 (fixed):

- **MRR** (Mean Reciprocal Rank) — average reciprocal rank of the first relevant result
- **Success@25** — fraction of queries where a relevant image appears in the top 25 (`mean(hit)`)
- **Diversity@25** (optional) — diversity of retrieved results

Leaderboard composite scores use 50/50 MRR + Success@25 (primary mode), or equal thirds with Diversity.

Full metric definitions: [`benchmarking/benchmarks/METRICS.md`](../benchmarking/benchmarks/METRICS.md)

## Leaderboards

Cross-version leaderboards compare results across system versions:

- **Per-benchmark:** `benchmarking/benchmarks/*/results/leaderboard.ipynb`
- **Overall (cross-benchmark):** `benchmarking/benchmarks/overall/leaderboard.ipynb`
- **Helpers:** `benchmarking/helpers/plot.py`

The GitHub Action `.github/workflows/leaderboards.yml` automatically re-executes leaderboard notebooks when results change.

## CI/CD

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `benchmarking.yml` | Changes to `benchmarking/benchmarks/**` or `helpers/**` | Builds and pushes benchmark Docker images to NRP GitLab registry |
| `leaderboards.yml` | Changes to `benchmarking/benchmarks/**/results/**` | Re-runs leaderboard notebooks |

CI builds images but does **not** run benchmarks on the cluster. Running benchmarks is manual via `make run`.

## Creating a new benchmark

```bash
cd benchmarking
cp -r benchmarks/template benchmarks/MYBENCHMARK
cd benchmarks/MYBENCHMARK
# Follow benchmarks/template/README.md
```

See also `benchmarks/template/QUICKSTART.md` for a 5-minute setup guide.

## Further reading

- [benchmarking/README.md](../benchmarking/README.md) — full benchmarking guide
- [benchmarking/benchmarks/MAKEFILE.md](../benchmarking/benchmarks/MAKEFILE.md) — Makefile commands
- [benchmarking/benchmarks/DOCKER.md](../benchmarking/benchmarks/DOCKER.md) — Docker image details
- [benchmarking/kubernetes/README.md](../benchmarking/kubernetes/README.md) — K8s deployment for benchmarks
