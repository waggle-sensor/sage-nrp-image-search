# Learning Sage Image Search

Interactive learning materials for [Sage Grande: Summer of AI](https://sagecontinuum.org/docs/events/2026-Sage-Summer-Hackathon) (July 20–28, 2026) and anyone who wants to understand how hybrid image search works on SAGE edge-camera data.

## Learning objectives

After completing the lab, you will be able to:

- Explain the Sage Image Search pipeline: captioning, embedding, indexing, hybrid search, reranking, and search UI
- Build fused CLIP embeddings (image + caption) and index them with Milvus Lite BM25 hybrid search
- Run a simplified version of the stack locally using Milvus Lite, open_clip, Gemma 4 E2B, and a Gradio search UI
- Map each lab component to the production system (Weaviate, Triton, NRP `run_nrp_model()`, weavloader, Gradio API, React portal)
- Evaluate retrieval quality with MRR and Success@K on SageBench queries

Try the live production UI at **[portal.sagecontinuum.org/labs/image-search](https://portal.sagecontinuum.org/labs/image-search)** — a React web app backed by the Gradio API server in `app/`.

The hackathon [Image Search at the Edge](https://sagecontinuum.org/docs/events/2026-Sage-Summer-Hackathon#potential-hackathon-projects) project extends this work to pluginctl deployment, edge-friendly models, and NanoDB.

## Companion videos

Optional walkthroughs from the SAGE team (also linked in the notebook):

| Video | When to watch | Link |
|-------|---------------|------|
| **Sage Image Search** — system overview and production walkthrough | After **Architecture overview**, or before Steps 1–8 | [Watch on YouTube](https://youtu.be/hzfKL0smzFM) |
| **Image Search Benchmarking** — metrics, Sagebench, and evaluation | Before or after **Step 9: Mini evaluation** | [Watch on YouTube](https://youtu.be/NUEs7AeGk4I) |

You can complete the lab without watching either video.

## Prerequisites

Before starting the lab:

- **Python** and basic machine learning familiarity
- **1 GPU** (recommended; CPU fallback uses SageBench `summary` captions instead of Gemma)
- A **[Hugging Face](https://huggingface.co/)** account
- An **`HF_TOKEN`** — create a `read` token at [User access tokens](https://huggingface.co/docs/hub/en/security-tokens) and accept the [Gemma 4 license](https://huggingface.co/google/gemma-4-E2B) on the model page

No SAGE credentials are required (the lab uses public SageBench data).

## Lab notebook walkthrough

The main lab is [sage_image_search_lab.ipynb](notebooks/sage_image_search_lab.ipynb). Run cells **in order** from top to bottom.

| Section | What you do |
|---------|-------------|
| **Setup** | Install requirements; sign in at [Hugging Face](https://huggingface.co/); set `HF_TOKEN` ([access tokens](https://huggingface.co/docs/hub/en/security-tokens)); configure `PERSIST_DIR` in `User_Persistent_Storage` |
| **Load SageBench** | 50-image subset (`SEED=42`, `SAMPLE_SIZE=50`) from [`sagecontinuum/SageBench`](https://huggingface.co/datasets/sagecontinuum/SageBench) |
| **Architecture overview** | Compare production vs lab components (with diagram) |
| **Step 1: Embeddings** | open_clip helpers + `get_clip_embeddings()` fused with production `clip_alpha` |
| **Step 2: Captions** | Gemma 4 E2B-it captioning (or `summary` fallback on CPU) → fused vectors |
| **Step 3: Index** | Milvus Lite collection `sage_lab` (`sage_lab.db`) with dense vectors + BM25 on `search_text` |
| **Step 4: Vector search** | Semantic ANN on fused CLIP vectors |
| **Step 5: Keyword search** | BM25 on captions + metadata |
| **Step 6: Hybrid search** | Milvus `hybrid_search` + `WeightedRanker(0.4, 0.6)` |
| **Step 7: Reranking** | CrossEncoder `ms-marco-MiniLM-L-6-v2` |
| **Step 8: User Interface** | Gradio text-query UI wired to `hybrid_search()` + `rerank()` |
| **Step 9: Mini evaluation** | MRR and Success@25 on SageBench labeled queries |
| **Conclusion** | End-to-end workflow recap and key ideas |
| **Stretch** | Choose your next learning path (Explorer → Pioneer) |

**Default hyperparameters** (set near the top of the notebook): `SEED=42`, `SAMPLE_SIZE=50`, `QUERY_ALPHA=0.4`, `TOP_K=25`, `CLIP_ALPHA=0.7` (from production `model_config.py`).

Steps 4 and 5 teach vector and keyword search separately; **Steps 6 onward** always use `hybrid_search()` — matching production before reranking.

## NDP workspace setup

Follow the same workflow as the official [NDP demo workspace](https://github.com/pramonettivega/demo-workspace). Instructors create an NDP workspace per [Set up a Workspace](https://nationaldataplatform.org/documentation/quick-start/set-up-workspace/).

### Instructor setup (one-time)

1. Create an NDP workspace titled e.g. **Sage Image Search Lab**
2. Under **Workspace Codebase**, add: `https://github.com/waggle-sensor/sage-nrp-image-search`
3. Optionally add [`sagecontinuum/SageBench`](https://huggingface.co/datasets/sagecontinuum/SageBench) under Additional Resources
4. Tell students to sign up at [Hugging Face](https://huggingface.co/), create **`HF_TOKEN`** ([User access tokens](https://huggingface.co/docs/hub/en/security-tokens); `read` role is enough), accept the [Gemma 4 license](https://huggingface.co/google/gemma-4-E2B), and set the token in the notebook or environment
5. Point students to this page

### Student setup

1. **Launch JupyterHub** — from the NDP workspace page or NDP Widget
2. **Reserve resources:**

   | Setting | Default (camp) | Recommended | CPU fallback |
   |---------|----------------|-------------|--------------|
   | **Region** | — | based on your location| based on your location |
   | **Zone** | — | based on your location| based on your location |
   | **GPUs** | **1** | **1** | 0 |
   | **Cores** | 4 | 4 | 2 |
   | **RAM** | 32 GB | 32 GB | 16 GB |
   | **GPU Type** | Any (≥16 GB VRAM recommended) | Tesla T4 | — |
   | **Image** | Minimal NDP Starter JupyterLab | Minimal NDP Starter JupyterLab | Minimal NDP Starter JupyterLab |
   | **Timeout (sec)** | 1200  | 1200 | 1200 |
   | **Architecture** | amd64 | amd64 | amd64 |

   **Why:** Gemma 4 E2B captioning needs ~5–8 GB model memory and a GPU. Use **sequential model loading** (caption → embed → rerank). Reserve **≥5 GB** in `User_Persistent_Storage` for HF cache, `sage_lab.db`, and `gradio_images/` thumbnails.

   **Expected runtime:** ~10–15 min install + ~30–40 min notebook with GPU (includes Gemma captioning and Gradio UI).

3. **Launch server** — Start Server, wait for JupyterLab
4. **Set storage** — open `User_Persistent_Storage`. Files outside persistent storage are **lost** when the server stops.
5. **Select workspace** — choose Sage Image Search Lab in the NDP Widget
6. **Clone repository** — confirm Current Folder is persistent storage; click **Clone Repository**
7. **Install requirements** — File Browser → `sage-nrp-image-search/docs/notebooks/` → NDP Widget → **Install requirements.txt**. Re-run every new session.
8. **Open notebook** — `docs/notebooks/sage_image_search_lab.ipynb`
9. **Stop server** when done — File → Hub Control Panel → Stop Server

**Return visits:** repeat steps 1–4 and 7 only.

**Notes:**

- **[Hugging Face](https://huggingface.co/) account** and **`HF_TOKEN`** required for Gemma 4 E2B-it (gated model). Create a `read` token at [User access tokens](https://huggingface.co/docs/hub/en/security-tokens).
- No SAGE credentials required (public SageBench data)
- Fallback install: `%pip install -r requirements.txt` in the notebook

### Troubleshooting

| Issue | What to try |
|-------|-------------|
| `AllocTimestamp` / `Method not implemented` during Milvus insert | Harmless Milvus Lite quirk if you still see `Indexed N fused vectors + BM25 text → …/sage_lab.db`. Re-install from `requirements.txt` to reduce log noise. |
| Gradio gallery shows broken images with path text | Restart kernel, re-run **Load SageBench** through **Step 8**. Step 8 caches JPEG thumbnails under `gradio_images/` in persistent storage. Open the printed Gradio URL in a new tab if `inline=True` does not render images. |
| `HF_TOKEN` / gated model download errors | Sign up at [Hugging Face](https://huggingface.co/), accept the [Gemma 4 license](https://huggingface.co/google/gemma-4-E2B), then create a `read` token at [User access tokens](https://huggingface.co/docs/hub/en/security-tokens). |
| No GPU / Gemma OOM | CPU mode uses SageBench `summary` captions instead of Gemma; relaunch with 1 GPU for the full lab. |
| Files missing after restart | Confirm you are working inside `User_Persistent_Storage`. |

## Full vs lab mapping

| Production | Lab equivalent | Why simplified |
|------------|----------------|----------------|
| Weaviate `HybridSearchExample` | Milvus Lite `sage_lab` (`sage_lab.db`) | Embedded DB, no Kubernetes |
| NRP `gemma` → [gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) via `run_nrp_model()` | [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B) via `transformers` + optional `bitsandbytes` | Same Gemma 4 family; edge-sized model for student GPU |
| Triton `clip` (`DFN5B-CLIP-ViT-H-14-378`) + fused image+caption vector (`clip_alpha=0.7`) | `open_clip` ViT-B-32 + same fusion | Smaller CLIP for classroom GPUs; same `fuse_embeddings` math |
| Weaviate hybrid (`alpha=0.4`) | Milvus `hybrid_search` + `WeightedRanker(0.4, 0.6)` | Same dense+BM25 fusion via Milvus built-in ranker |
| `reranker-transformers` | `ms-marco-MiniLM-L-6-v2` CrossEncoder | Same reranker family |
| weavloader Celery ingestion | Notebook indexing loop | No distributed queues |
| Gradio API (`app/main.py`) + [React portal](https://portal.sagecontinuum.org/labs/image-search) | Gradio UI in notebook (`lab_text_query`) | Lab combines UI and API; production splits React frontend from Gradio backend |
| Full benchmarking suite | Mini MRR / Success@K on SageBench | Links to [Sagebench](../benchmarking/benchmarks/Sagebench/) |

### Captioning: production vs lab

| | Production | Lab |
|---|------------|-----|
| **Model** | [gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) | [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B) |
| **API** | `run_nrp_model()` in [weavloader/inference/model.py](../weavloader/inference/model.py) | `transformers` `AutoModelForMultimodalLM` |
| **Prompt** | [model_config.py](../weavloader/inference/model_config.py) `caption_model_prompt` | Same prompt text |

### Key ideas (from the lab conclusion)

1. **Captions bridge pixels and words** — VLM captions make images searchable by text; metadata (`vsn`, `camera`, `zone`) strengthens keyword matching.
2. **Hybrid beats either alone** — Vector search understands scene semantics; BM25 nails exact node IDs and metadata terms. Production blends them with `alpha=0.4`.
3. **Reranking sharpens the top of the list** — Hybrid retrieval casts a wide net; the cross-encoder re-scores query–caption pairs so the best matches rise to the top.
4. **Same metrics, different scale** — Lab MRR / Success@25 use the same definitions as [METRICS.md](../benchmarking/benchmarks/METRICS.md); run `make run-local` in Sagebench for production-grade numbers.

## Learning paths by difficulty

| Level | Audience | Activities | Difficulty | Next steps |
|-------|----------|------------|------------|------------|
| **1 — Explorer** | New to the system | Run the notebook; try queries in the Gradio UI; compare with [production portal](https://portal.sagecontinuum.org/labs/image-search) | Easy | [overview.md](overview.md), [glossary.md](glossary.md) |
| **2 — Tinkerer** | Comfortable with Python | Compare Gemma vs `summary` captions; tune `QUERY_ALPHA`; plot MRR@K | Medium | [configuration.md](configuration.md), [model_config.py](../weavloader/inference/model_config.py) |
| **3 — Builder** | Ready for real stack | Deploy dev UI on NRP; run Sagebench `make run-local`; try NRP API | Hard | [getting-started.md](getting-started.md), [benchmarking.md](benchmarking.md) |
| **4 — Contributor** | Wants to ship code | Pick a [roadmap item](CONTRIBUTING.md#roadmap); open a PR | Hard | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **5 — Pioneer** | Building something new from scratch | [Image Search at the Edge](https://sagecontinuum.org/docs/events/2026-Sage-Summer-Hackathon#potential-hackathon-projects) — pluginctl, edge models, NanoDB | Very Hard | Hackathon project page |

**Learners:** levels 1–2 (docs + notebook). **Contributors:** levels 3–4 (repo + PR workflow). **Pioneers:** level 5 (greenfield edge deployment).

## Notebooks

| Notebook | Runtime | Description |
|----------|---------|-------------|
| [sage_image_search_lab.ipynb](notebooks/sage_image_search_lab.ipynb) | ~30–40 min (GPU) | Full pipeline lab: SageBench subset → caption → fused CLIP → Milvus hybrid → rerank → Gradio UI → mini eval |

Additional notebooks (edge deployment, full Weaviate path) may be added later.

## Reference documentation

- [Using the Search UI](using-the-search-ui.md) — production portal and Gradio API
- [Architecture](architecture.md) — production components and data flows
- [Benchmarking](benchmarking.md) — full evaluation suite
- [Glossary](glossary.md) — terms used in the lab and production docs
- [Contributing & Roadmap](CONTRIBUTING.md) — how to contribute
