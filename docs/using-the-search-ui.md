# Using the Search UI

Sage Image Search provides a web interface for finding SAGE camera images with natural-language text queries.

## Accessing the UI

| Environment | URL | Notes |
|-------------|-----|-------|
| **Sage portal (production)** | [https://portal.sagecontinuum.org/labs/image-search](https://portal.sagecontinuum.org/labs/image-search) | React web app — primary user-facing UI |
| Local (Docker Compose) | [http://localhost:7860](http://localhost:7860) | Gradio API server (direct access during development) |
| NRP dev | [https://dev-sage-hybrid-search.nrp-nautilus.io](https://dev-sage-hybrid-search.nrp-nautilus.io) | Gradio API server |
| NRP prod | [https://sage-hybrid-search.nrp-nautilus.io](https://sage-hybrid-search.nrp-nautilus.io) | Gradio API server |

**How it fits together:** In production, the [Sage portal](https://portal.sagecontinuum.org/labs/image-search) is a React application. It calls the Gradio service in `app/` for hybrid search, image retrieval, and metadata. The Gradio URLs above are the API backend; developers and local Docker Compose users can open Gradio directly.

## Running a search

1. Open the search page (portal or Gradio).
2. Enter a description in the text box, for example:
   - `Show me images in Hawaii`
   - `Snowy Mountains`
   - `Cars in W049`
   - `intersection in the right camera`
3. Click **Submit**.
4. Review the result panels:
   - **Returned Images** — thumbnail gallery of matching images
   - **Metadata** — table with scores and SAGE fields for each result
   - **Image Locations** — map showing GPS coordinates when available (Gradio UI; portal may vary)

Example queries are available in the **Example Queries** dataset below the text box. Click one to populate the search field.

## Understanding results

Each result includes these fields:

| Field | Meaning |
|-------|---------|
| `caption` | AI-generated description of the image |
| `score` | Hybrid search fusion score (higher is more relevant) |
| `explainScore` | Breakdown of how vector and keyword scores were combined |
| `rerank_score` | Relevance score from Triton CLIP query–image similarity |
| `vsn` | SAGE node identifier (Virtual Sensor Number) |
| `camera` | Camera name on the node (e.g. `top`, `left`, `right`) |
| `zone` | Deployment zone |
| `project` | SAGE project |
| `timestamp` | When the image was captured |
| `host`, `job`, `plugin`, `task` | SAGE task metadata |
| `address` | Human-readable location string |
| `filename` | Original image filename |

Results are limited to **25** by default (`response_limit` in `app/HyperParameters.py`).

## How ranking works

Results are ranked in two stages:

1. **Hybrid retrieval** — Milvus combines vector similarity (CLIP embedding of your query vs. stored embeddings) with BM25 keyword matching on `search_text` (captions + metadata). The blend weight `alpha=0.4` means 40% vector, 60% keyword via `WeightedRanker`.
2. **Reranking** — Triton CLIP (`DFN5B-CLIP-ViT-H-14-378`) re-scores the top results by similarity between your query text and each retrieved image (same idea as HF CLIP `logits_per_image`).

Keyword search covers these fields: `caption`, `camera`, `host`, `job`, `vsn`, `plugin`, `zone`, `project`, `address`.

## Tips for effective queries

- **Be specific** — `smoke visible in top camera on W049` works better than `smoke`.
- **Use metadata terms** — Node IDs (`W049`), camera names (`top`, `right`), zones, and project names are searchable via keyword matching.
- **Describe scenes** — Semantic search handles natural descriptions like `rainy street at night` or `mountain landscape with snow`.
- **Combine both** — `clouds in the top camera on W040` leverages keyword matching on metadata and semantic matching on the scene.

## What you will and won't see

**You will see:**
- Images that have been indexed by weavloader into Milvus
- Results from nodes not on the `UNALLOWED_NODES` deny list
- Images your Sage credentials can download (in K8s deployments where SAGE creds are configured)

**You won't see:**
- Images from nodes on the deny list (filtered at query time)
- Images you don't have Sage access to (skipped during indexing, or fail to download for display)
- Image-to-image search (the Image Query tab is not implemented yet)

## Limitations

- The Gradio service in `app/` is the search API backend; the primary user interface is the React app at [portal.sagecontinuum.org/labs/image-search](https://portal.sagecontinuum.org/labs/image-search).
- There is no per-user access control yet — filtering uses a static node deny list, not your Sage token permissions.
- Local Docker Compose may not display images in the gallery unless SAGE credentials are configured for the Gradio container.

## Next steps

- [Learning lab](learning.md) — build your own Gradio search UI in the interactive notebook
- [Authentication](authentication.md) — set up Sage credentials
- [Troubleshooting](troubleshooting.md) — fix empty results or missing images
- [Glossary](glossary.md) — look up terms like VSN, hybrid search, reranker
