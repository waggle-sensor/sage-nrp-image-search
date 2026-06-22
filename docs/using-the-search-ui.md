# Using the Search UI

The search UI is a Gradio web app that lets you find SAGE camera images using natural-language text queries.

## Accessing the UI

| Environment | URL |
|-------------|-----|
| Local (Docker Compose) | [http://localhost:7860](http://localhost:7860) |
| NRP dev | [https://dev-sage-hybrid-search.nrp-nautilus.io](https://dev-sage-hybrid-search.nrp-nautilus.io) |
| NRP prod | [https://sage-hybrid-search.nrp-nautilus.io](https://sage-hybrid-search.nrp-nautilus.io) |

## Running a search

1. Open the **Text Query** tab.
2. Enter a description in the text box, for example:
   - `Show me images in Hawaii`
   - `Snowy Mountains`
   - `Cars in W049`
   - `intersection in the right camera`
3. Click **Submit**.
4. Review the three result panels:
   - **Returned Images** — thumbnail gallery of matching images
   - **Metadata** — table with scores and SAGE fields for each result
   - **Image Locations** — map showing GPS coordinates when available

Example queries are available in the **Example Queries** dataset below the text box. Click one to populate the search field.

## Understanding results

Each result includes these fields:

| Field | Meaning |
|-------|---------|
| `caption` | AI-generated description of the image |
| `score` | Hybrid search fusion score (higher is more relevant) |
| `explainScore` | Breakdown of how vector and keyword scores were combined |
| `rerank_score` | Relevance score from the cross-encoder reranker |
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

1. **Hybrid retrieval** — Weaviate combines vector similarity (CLIP embedding of your query vs. stored embeddings) with BM25 keyword matching on captions and metadata. The blend weight `alpha=0.4` means 40% vector, 60% keyword.
2. **Reranking** — A cross-encoder model (`ms-marco-MiniLM-L-6-v2`) re-scores the top results by comparing your query against each image's caption.

Keyword search covers these fields: `caption`, `camera`, `host`, `job`, `vsn`, `plugin`, `zone`, `project`, `address`.

## Tips for effective queries

- **Be specific** — `smoke visible in top camera on W049` works better than `smoke`.
- **Use metadata terms** — Node IDs (`W049`), camera names (`top`, `right`), zones, and project names are searchable via keyword matching.
- **Describe scenes** — Semantic search handles natural descriptions like `rainy street at night` or `mountain landscape with snow`.
- **Combine both** — `clouds in the top camera on W040` leverages keyword matching on metadata and semantic matching on the scene.

## What you will and won't see

**You will see:**
- Images that have been indexed by weavloader into Weaviate
- Results from nodes not on the `UNALLOWED_NODES` deny list
- Images your Sage credentials can download (in K8s deployments where SAGE creds are configured)

**You won't see:**
- Images from nodes on the deny list (filtered at query time)
- Images you don't have Sage access to (skipped during indexing, or fail to download for display)
- Image-to-image search (the Image Query tab is not implemented yet)

## Limitations

- The Gradio UI is interim. A production UI integrated with beekeeper is planned.
- There is no per-user access control yet — filtering uses a static node deny list, not your Sage token permissions.
- Local Docker Compose may not display images in the gallery unless SAGE credentials are configured for the Gradio container.

## Next steps

- [Authentication](authentication.md) — set up Sage credentials
- [Troubleshooting](troubleshooting.md) — fix empty results or missing images
- [Glossary](glossary.md) — look up terms like VSN, hybrid search, reranker
