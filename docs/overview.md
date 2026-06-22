# Overview

Sage Image Search indexes images from SAGE edge cameras and retrieves them using natural-language queries. Instead of searching raw pixels, the system generates a scientific caption for each image, embeds both the image and caption with CLIP, and stores everything in a vector database for fast hybrid retrieval.

## Who is this for?

- **Researchers and operators** who want to find SAGE camera images by describing what they are looking for
- **Platform operators** who deploy and maintain the stack on NRP or locally
- **Developers** who tune models, run benchmarks, or extend the ingestion pipeline

## How search works (high level)

1. **Caption generation** — A vision-language model (VLM) writes a detailed caption for each indexed image.
2. **Embedding** — CLIP produces vector embeddings from the image and caption.
3. **Storage** — Captions, metadata, embeddings, and the image blob are stored in Weaviate.
4. **Hybrid search** — When you query, the system combines:
   - **Vector search** — semantic similarity between your query and CLIP embeddings
   - **Keyword search** — BM25 text matching on captions and SAGE metadata fields
5. **Reranking** — A cross-encoder reranker re-scores the top results against your query for better ordering.

## What gets indexed

Each image record in Weaviate (`HybridSearchExample` collection) includes:

| Data | Description |
|------|-------------|
| Caption | AI-generated description of the image |
| Image blob | The original image bytes |
| CLIP embedding | Named vector `clip` for semantic search |
| SAGE metadata | `vsn`, `camera`, `zone`, `job`, `host`, `plugin`, `project`, `task`, `address`, `timestamp`, `link` |
| Location | GPS coordinates when available |

## Key features

- **Hybrid search** — Combines semantic and keyword retrieval for better accuracy than either alone
- **Continuous ingestion** — Weavloader polls the SAGE data stream and indexes new images automatically
- **Reranking** — Cross-encoder model refines result order after initial retrieval
- **Map view** — Search results with GPS data appear on an interactive map in the UI
- **Benchmarking** — Five domain-specific benchmarks measure retrieval quality (see [Benchmarking](benchmarking.md))

## Current limitations

- **Interim UI** — The Gradio app in `app/` is a temporary demo. A production UI integrated with beekeeper/beehive-data-api is planned.
- **Text search only** — Image-to-image search is not implemented yet (the Image Query tab is commented out).
- **No REST API** — There is no standalone search API today; the Gradio app queries Weaviate directly.
- **Static access control** — Results are filtered by a static node deny list (`UNALLOWED_NODES`), not per-user Sage ACL. Images you lack Sage access to are skipped during indexing.
- **Docker Compose vs Kubernetes** — Local Docker Compose uses a different vectorizer setup (`multi2vec-bind`) than the NRP Kubernetes deployment (user-provided CLIP vectors). See [Architecture](architecture.md) for details.

## Next steps

- [Architecture](architecture.md) — system components and data flows
- [Using the Search UI](using-the-search-ui.md) — how to search and read results
- [Getting Started](getting-started.md) — deploy locally or on NRP
