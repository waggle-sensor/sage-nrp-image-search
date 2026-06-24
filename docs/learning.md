# Learning Sage Image Search

Interactive learning materials for [Sage Grande: Summer of AI](https://sagecontinuum.org/docs/events/2026-Sage-Summer-Hackathon) (July 20–28, 2026) and anyone who wants to understand how hybrid image search works on SAGE edge-camera data.

## Learning objectives

After completing the lab, you will be able to:

- Explain the Sage Image Search pipeline: captioning, embedding, indexing, hybrid search, reranking, and search UI
- Run a simplified version of the stack locally using Milvus Lite, CLIP, Gemma 4 E2B, and a Gradio search UI
- Map each lab component to the production system (Weaviate, Triton, NRP `run_nrp_model()`, weavloader, Gradio API, React portal)
- Evaluate retrieval quality with MRR and Success@K on SageBench queries

Try the live production UI at **[portal.sagecontinuum.org/labs/image-search](https://portal.sagecontinuum.org/labs/image-search)** — a React web app backed by the Gradio API server in `app/`.

The hackathon [Image Search at the Edge](https://sagecontinuum.org/docs/events/2026-Sage-Summer-Hackathon#potential-hackathon-projects) project extends this work to pluginctl deployment, edge-friendly models, and NanoDB.

## NDP workspace setup

Follow the same workflow as the official [NDP demo workspace](https://github.com/pramonettivega/demo-workspace). Instructors create an NDP workspace per [Set up a Workspace](https://nationaldataplatform.org/documentation/quick-start/set-up-workspace/).

### Instructor setup (one-time)

1. Create an NDP workspace titled e.g. **Sage Image Search Lab**
2. Under **Workspace Codebase**, add: `https://github.com/waggle-sensor/sage-nrp-image-search`
3. Optionally add [`sagecontinuum/SageBench`](https://huggingface.co/datasets/sagecontinuum/SageBench) under Additional Resources
4. Tell students to set **`HF_TOKEN`** (Hugging Face account + accept the [Gemma 4 license](https://huggingface.co/google/gemma-4-E2B))
5. Point students to this page

### Student setup

1. **Launch JupyterHub** — from the NDP workspace page or NDP Widget
2. **Reserve resources:**

   | Setting | Default (camp) | Recommended | CPU fallback |
   |---------|----------------|-------------|--------------|
   | **GPUs** | **1** | **1** | 0 |
   | **Cores** | 4 | 4 | 2 |
   | **RAM** | 32 GB | 32 GB | 16 GB |
   | **GPU Type** | Any (≥16 GB VRAM recommended) | Any (≥16 GB VRAM) | — |
   | **Image** | PyTorch / ML starter image | PyTorch / ML starter image | Minimal NDP Starter JupyterLab |
   | **Architecture** | amd64 | amd64 | amd64 |

   **Why:** Gemma 4 E2B captioning needs ~5–8 GB model memory and a GPU. Use **sequential model loading** (caption → embed → rerank). Reserve **≥5 GB** in `User_Persistent_Storage` for HF cache + Milvus DB.

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

- **`HF_TOKEN` required** for Gemma 4 E2B-it (gated model)
- No SAGE credentials required (public SageBench data)
- Fallback install: `%pip install -r requirements.txt` in the notebook

## Full vs lab mapping

| Production | Lab equivalent | Why simplified |
|------------|----------------|----------------|
| Weaviate `HybridSearchExample` | Milvus Lite `sage_lab_images` | Embedded DB, no Kubernetes |
| NRP `gemma` → [gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) via `run_nrp_model()` | [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B) via `transformers` | Same Gemma 4 family; edge-sized model for student GPU |
| Triton `clip` (`DFN5B-CLIP-ViT-H-14-378`) + fused image+caption vector (`clip_alpha=0.7`) | `open_clip` ViT-B-32 + same fusion | Smaller CLIP for classroom GPUs; same `fuse_embeddings` math |
| Weaviate hybrid (`alpha=0.4`) | Milvus `hybrid_search` + `WeightedRanker` | Same dense+BM25 fusion via Milvus built-in ranker |
| `reranker-transformers` | `ms-marco-MiniLM-L-6-v2` CrossEncoder | Same reranker family |
| weavloader Celery ingestion | Notebook indexing loop | No distributed queues |
| Gradio API (`app/main.py`) + [React portal](https://portal.sagecontinuum.org/labs/image-search) | Gradio UI in notebook | Lab combines UI and API; production splits React frontend from Gradio backend |
| Full benchmarking suite | Mini MRR / Success@K on SageBench | Links to [Sagebench](../benchmarking/benchmarks/Sagebench/) |

### Captioning: production vs lab

| | Production | Lab |
|---|------------|-----|
| **Model** | [gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) | [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B) |
| **API** | `run_nrp_model()` in [weavloader/inference/model.py](../weavloader/inference/model.py) | `transformers` `AutoModelForMultimodalLM` |
| **Prompt** | [model_config.py](../weavloader/inference/model_config.py) `caption_model_prompt` | Same prompt text |

## Learning paths by difficulty

| Level | Audience | Activities | Next steps |
|-------|----------|------------|------------|
| **1 — Explorer** | New to the system | Run the notebook; try queries in the Gradio UI; compare with [production portal](https://portal.sagecontinuum.org/labs/image-search) | [overview.md](overview.md), [glossary.md](glossary.md) |
| **2 — Tinkerer** | Comfortable with Python | Compare Gemma vs `summary` captions; tune `alpha`; plot MRR@K | [configuration.md](configuration.md), [model_config.py](../weavloader/inference/model_config.py) |
| **3 — Builder** | Ready for real stack | Deploy dev UI on NRP; run Sagebench `make run-local`; try NRP API | [getting-started.md](getting-started.md), [benchmarking.md](benchmarking.md) |
| **4 — Contributor** | Wants to ship code | Pick a [roadmap item](CONTRIBUTING.md#roadmap); open a PR | [CONTRIBUTING.md](CONTRIBUTING.md) |

**Learners:** levels 1–2 (docs + notebook). **Contributors:** levels 3–4 (repo + PR workflow).

## Notebooks

| Notebook | Runtime | Description |
|----------|---------|-------------|
| [sage_image_search_lab.ipynb](notebooks/sage_image_search_lab.ipynb) | ~30–40 min (GPU) | Full pipeline lab on SageBench subset, ending with a Gradio search UI |

Additional notebooks (edge deployment, full Weaviate path) may be added later.

## Reference documentation

- [Using the Search UI](using-the-search-ui.md) — production portal and Gradio API
- [Architecture](architecture.md) — production components and data flows
- [Benchmarking](benchmarking.md) — full evaluation suite
- [Contributing & Roadmap](CONTRIBUTING.md) — how to contribute
