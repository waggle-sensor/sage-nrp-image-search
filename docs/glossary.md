# Glossary

Terms used throughout Sage Image Search documentation.

| Term | Definition |
|------|------------|
| **Alpha (query)** | Hybrid search blend parameter (`query_alpha`). A value of 0 means pure keyword (BM25) search; 1 means pure vector search. The default is 0.4 (40% vector, 60% keyword) via `WeightedRanker(0.4, 0.6)`. |
| **ANN** | Approximate nearest neighbor. Fast similarity search over embedding vectors; trades perfect accuracy for speed at large scale. |
| **BM25** | Best Matching 25 — a keyword ranking algorithm used for text-based search on `search_text` (caption + metadata). In Milvus, produced via `FunctionType.BM25` into a sparse vector field. |
| **Bi-encoder** | Model architecture that embeds query and document separately (e.g. CLIP). Used for first-stage retrieval and for CLIP-based reranking (query text vs caption). |
| **Caption** | An AI-generated text description of an image, produced by a vision-language model (VLM) such as Gemma 4 (production) or Florence-2 (learning lab). |
| **Celery** | Distributed task queue used by weavloader for image processing, monitoring, and cleanup. |
| **clip_alpha** | Index-time fusion weight when combining CLIP image and caption embeddings (default 0.7). Higher values weight the image vector more; lower values weight the caption more. See `fuse_embeddings()` in `weavloader/inference/model.py`. |
| **CLIP** | Contrastive Language-Image Pre-training. The production embedding model is `DFN5B-CLIP-ViT-H-14-378` (served via Triton). The learning lab uses a smaller `open_clip` ViT-B-32 variant with the same fusion math. |
| **Cross-encoder** | Neural model that scores a query–document pair jointly in one forward pass. The learning lab still uses `ms-marco-MiniLM-L-6-v2` for teaching; production reranks with Triton CLIP instead. |
| **DLQ** | Dead Letter Queue. Redis-backed archive of Celery tasks that failed after all retries. |
| **DFN5B-CLIP-ViT-H-14-378** | Apple's CLIP variant used in production for image and text embeddings. Served by Triton as the `clip` model. |
| **Embedding** | A fixed-length numerical vector representing an image or text. Similar embeddings indicate semantic similarity. |
| **Embedding fusion** | Combining separate CLIP image and caption vectors into one stored vector via `fuse_embeddings()` and `clip_alpha` before indexing. |
| **Flower** | Web UI for monitoring Celery workers and task queues. Runs inside the weavloader pod on port 5555. |
| **Fusion (query)** | The process of combining vector and keyword search scores at query time. Production and lab both use Milvus `WeightedRanker`. |
| **Florence-2** | Microsoft's vision foundation model family ([Florence-2-base](https://huggingface.co/microsoft/Florence-2-base), 0.23B params). The learning lab uses it for captioning via task prompts like `<MORE_DETAILED_CAPTION>`. |
| **Gemma** | Google's open vision-language model family. Production captioning on NRP uses `gemma-4-31B-it-qat-w4a16-ct` via the NRP AI Gateway. |
| **Gradio** | Python library for web UIs. In production, the Gradio app in `app/` is the **search API backend** (hybrid search, image fetch, metadata). The primary user-facing UI is the React portal; Gradio can also be opened directly for local dev. |
| **Hybrid search** | Search that combines vector (semantic) and keyword (BM25) retrieval, fused into a single ranked result set. |
| **HybridSearchExample** | Default Milvus collection name for production/local. Dev uses `HybridSearchExampleDev`. |
| **imagesampler** | SAGE Waggle plugin that produces camera image samples. Weavloader polls the data stream for new `imagesampler` tasks. |
| **imsearch_eval** | Python benchmarking framework ([waggle-sensor/imsearch_eval](https://github.com/waggle-sensor/imsearch_eval)) used to index datasets, run queries, and compute MRR and Success@K across Sagebench, Firebench, and other benchmarks. |
| **Ingestion** | The pipeline stage that downloads SAGE images, generates captions, computes embeddings, and writes records to the vector database. Handled by weavloader. |
| **Inner product (IP)** | Vector similarity metric (dot product). Used by Milvus dense indexes in the learning lab; higher scores mean closer vectors. Production dense metric is COSINE (equivalent for L2-normalized embeddings). |
| **Milvus** | Open-source vector database with native hybrid search (dense + sparse BM25). Compose and Kubernetes use NRP-managed Milvus at `milvus.nrp-nautilus.io:50051`. |
| **Milvus Lite** | Embedded Milvus mode that runs locally without a separate server. Used in `docs/notebooks/sage_image_search_lab.ipynb`. |
| **MMR** | Maximal Marginal Relevance. Diversity-aware reranking that reduces near-duplicate results. Discussed as a future improvement; not in production today. |
| **MRR** | Mean Reciprocal Rank. Benchmark metric measuring how quickly the first relevant result appears in the ranked list. |
| **NDP** | National Data Platform. Cloud JupyterHub environment used for the Summer AI learning lab and hackathon workspaces. |
| **NRP** | National Research Platform. The compute infrastructure where Sage Image Search is deployed. |
| **NRP AI Gateway** | External API for running LLM inference. Used for caption generation when `LLM_RUN_MODE=NRP` (default on Kubernetes). |
| **Portal (Image Search)** | Production React web UI at [portal.sagecontinuum.org/labs/image-search](https://portal.sagecontinuum.org/labs/image-search). Calls the Gradio API backend for search and image retrieval. |
| **Redis** | In-memory data store used as the Celery broker, result backend, ingestion cursor, and DLQ inside the weavloader pod. |
| **Reranker** | Re-scores top hybrid-search hits. Production uses Triton CLIP (`DFN5B-CLIP-ViT-H-14-378`) query-text vs image `logits_per_image` (cosine × `exp(logit_scale)`). |
| **Retrieval** | The first search stage: finding candidate images via hybrid (vector + keyword) search before reranking. |
| **SAGE** | Software-Defined Sensor Network platform ([sagecontinuum.org](https://sagecontinuum.org)). Source of camera images. |
| **SageBench** | Hugging Face dataset and benchmark (`sagecontinuum/SageBench`) with metadata-aware natural-language queries on SAGE edge-camera images. Used in benchmarking and the learning lab mini-evaluation. |
| **SAGE data stream** | The event stream of sensor data (including images) from SAGE nodes, polled by weavloader. |
| **search_text** | Concatenated caption + metadata string inserted into Milvus; tokenized for BM25 sparse search. |
| **Semantic search** | Synonym for vector search — finding results by meaning rather than exact keyword overlap. |
| **Sparse vector** | Keyword representation produced by BM25 indexing. In Milvus, stored alongside dense CLIP vectors for native hybrid search. |
| **Success@25** | Benchmark metric: fraction of queries where a relevant image appears in the top 25 results. |
| **Triton** | NVIDIA Triton Inference Server. Serves CLIP embeddings on NRP; may also serve Gemma caption models when `LLM_RUN_MODE=TRITON`. |
| **UNALLOWED_NODES** | Comma-separated list of VSNs excluded from indexing and filtered from query results. |
| **Vector search** | Semantic similarity search using embedding vectors (CLIP). Finds images conceptually similar to the query. |
| **VLM** | Vision-Language Model. Generates captions by understanding image content (e.g. Gemma 4 on NRP, Florence-2 in the learning lab, or Gemma 3 via Triton in Docker Compose). |
| **VSN** | Virtual Sensor Number. Unique identifier for a SAGE edge node (e.g. `W049`, `W040`). |
| **WeightedRanker** | Milvus component that fuses dense vector and sparse BM25 scores with configurable weights (e.g. `WeightedRanker(0.4, 0.6)`). |
| **Weavloader** | The ingestion service that monitors SAGE streams, processes images, and writes to Milvus (directory/image name retained for CI/registry compatibility). |
| **Weavmanage** | One-shot Kubernetes Job that applies Milvus schema migrations (directory/image name retained for CI/registry compatibility). |
