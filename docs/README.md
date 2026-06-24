# Sage Image Search Documentation

Sage Image Search is a hybrid image retrieval system for the [SAGE Continuum](https://sagecontinuum.org) sensor network. It continuously indexes camera images from SAGE, generates AI captions and embeddings, stores them in Weaviate, and lets you search with natural-language text queries.

## I want to...

| Goal | Start here |
|------|------------|
| Learn interactively (Summer of AI lab) | [Learning Lab](learning.md) |
| Search for images | [Using the Search UI](using-the-search-ui.md) |
| Run or deploy the stack | [Getting Started](getting-started.md) |
| Set up credentials | [Authentication](authentication.md) |
| Understand how it works | [Overview](overview.md) → [Architecture](architecture.md) |
| Configure models or search tuning | [Configuration](configuration.md) |
| Fix a problem | [Troubleshooting](troubleshooting.md) |
| Evaluate or benchmark the system | [Benchmarking](benchmarking.md) |
| Look up a term | [Glossary](glossary.md) |
| Contribute or see planned work | [Contributing & Roadmap](CONTRIBUTING.md) |

## Component documentation

For deeper operational detail on individual services, see:

| Component | Documentation |
|-----------|---------------|
| Kubernetes deployment | [kubernetes/README.md](../kubernetes/README.md) |
| Image ingestion (weavloader) | [weavloader/README.md](../weavloader/README.md) |
| Benchmarking suite | [benchmarking/README.md](../benchmarking/README.md) |

## Quick start

**Local (Docker Compose):**

```bash
cp .env.example .env   # fill in SAGE_USER, SAGE_PASS, HF_TOKEN
docker compose up -d --build
```

Open the search UI at [http://localhost:7860](http://localhost:7860).

**NRP (Kubernetes):**

```bash
kubectl apply -k kubernetes/nrp-dev
```

Dev UI: [https://dev-sage-hybrid-search.nrp-nautilus.io](https://dev-sage-hybrid-search.nrp-nautilus.io)

See [Getting Started](getting-started.md) for full setup instructions.
