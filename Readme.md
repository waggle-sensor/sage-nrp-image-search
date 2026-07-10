# Sage Image Search on NRP

Hybrid image search for the [SAGE Continuum](https://sagecontinuum.org) sensor network. Images are captioned by a vision-language model, embedded with CLIP, and stored in Weaviate. Search combines vector similarity and keyword matching (BM25), then reranks results for relevance.

**[Full documentation →](docs/README.md)**

## Features

- **Caption generation** — VLM-generated captions for semantic and keyword search
- **Vector search** — CLIP embeddings of images and captions
- **Keyword search** — BM25 on captions and SAGE metadata
- **Hybrid search** — Fused vector + keyword retrieval
- **Reranker** — Cross-encoder re-scoring of top results
- **Continuous ingestion** — Automatic indexing from the SAGE data stream

## Quick start

**Local (Docker Compose):**

```bash
cp .env.example .env   # fill in SAGE_USER, SAGE_PASS, HF_TOKEN (see https://huggingface.co/docs/hub/en/security-tokens)
docker compose up -d --build
```

UI: [http://localhost:7860](http://localhost:7860)

**NRP (Kubernetes):**

```bash
kubectl apply -k kubernetes/nrp-dev
```

Dev UI: [https://dev-sage-hybrid-search.nrp-nautilus.io](https://dev-sage-hybrid-search.nrp-nautilus.io)

See [docs/getting-started.md](docs/getting-started.md) for full setup, secrets, and troubleshooting.

## Documentation

| Topic | Guide |
|-------|-------|
| Learning lab (Summer of AI) | [docs/learning.md](docs/learning.md) |
| Overview and concepts | [docs/overview.md](docs/overview.md) |
| System architecture | [docs/architecture.md](docs/architecture.md) |
| Using the search UI | [docs/using-the-search-ui.md](docs/using-the-search-ui.md) |
| Authentication | [docs/authentication.md](docs/authentication.md) |
| Configuration | [docs/configuration.md](docs/configuration.md) |
| Benchmarking | [docs/benchmarking.md](docs/benchmarking.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Glossary | [docs/glossary.md](docs/glossary.md) |
| Contributing & roadmap | [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) |

Component-specific docs: [kubernetes/](kubernetes/README.md) · [weavloader/](weavloader/README.md) · [benchmarking/](benchmarking/README.md)

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for how to open a pull request, contribution guidelines, the project roadmap, and research directions.

## CI/CD

GitHub Actions build and push Docker images for all microservices to the NRP GitLab registry on pushes to `main` and on pull requests.
