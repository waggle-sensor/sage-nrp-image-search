# Glossary

Terms used throughout Sage Image Search documentation.

| Term | Definition |
|------|------------|
| **Alpha** | Hybrid search blend parameter (`query_alpha`). A value of 0 means pure keyword (BM25) search; 1 means pure vector search. The default is 0.4 (40% vector, 60% keyword). |
| **BM25** | Best Matching 25 — a keyword ranking algorithm used for text-based search on captions and metadata fields. |
| **Caption** | An AI-generated text description of an image, produced by a vision-language model (VLM) such as Gemma-3. |
| **CLIP** | Contrastive Language-Image Pre-training. The embedding model used for vector search (`DFN5B-CLIP-ViT-H-14-378`). |
| **Celery** | Distributed task queue used by weavloader for image processing, monitoring, and cleanup. |
| **ColBERT** | Contextualized late interaction over BERT. An experimental embedding model; not used in the current production query path. |
| **DLQ** | Dead Letter Queue. Redis-backed archive of Celery tasks that failed after all retries. |
| **Fusion** | The process of combining vector and keyword search scores in hybrid retrieval. Uses `RELATIVE_SCORE` fusion in Weaviate. |
| **Gradio** | Python library for building web UIs. Used for the interim search interface. |
| **Hybrid search** | Search that combines vector (semantic) and keyword (BM25) retrieval, fused into a single ranked result set. |
| **HybridSearchExample** | The Weaviate collection name where indexed images are stored. |
| **ImageBind** | Multimodal embedding model (`multi2vec-bind`). Used in Docker Compose but not in the NRP Kubernetes deployment. |
| **MRR** | Mean Reciprocal Rank. Benchmark metric measuring how quickly the first relevant result appears. |
| **Named vector** | A Weaviate schema feature allowing multiple vector spaces per object. Sage Image Search uses the `clip` named vector. |
| **NRP** | National Research Platform. The compute infrastructure where Sage Image Search is deployed. |
| **NRP AI Gateway** | External API for running LLM inference. Used for caption generation when `LLM_RUN_MODE=NRP`. |
| **Reranker** | A cross-encoder model (`ms-marco-MiniLM-L-6-v2`) that re-scores search results for better relevance ordering. |
| **SAGE** | Software-Defined Sensor Network platform ([sagecontinuum.org](https://sagecontinuum.org)). Source of camera images. |
| **SAGE data stream** | The event stream of sensor data (including images) from SAGE nodes, polled by weavloader. |
| **Success@25** | Benchmark metric: fraction of queries where a relevant image appears in the top 25 results. |
| **Triton** | NVIDIA Triton Inference Server. Serves CLIP and Gemma models for embedding and caption generation. |
| **UNALLOWED_NODES** | Comma-separated list of VSNs excluded from indexing and filtered from query results. |
| **Vector search** | Semantic similarity search using embedding vectors (CLIP). Finds images conceptually similar to the query. |
| **VLM** | Vision-Language Model. Generates captions by understanding image content (e.g. Gemma-3-4b-it). |
| **VSN** | Virtual Sensor Number. Unique identifier for a SAGE edge node (e.g. `W049`, `W040`). |
| **Weaviate** | Open-source vector database used for storage, hybrid search, and reranker integration. |
| **Weavloader** | The ingestion service that monitors SAGE streams, processes images, and writes to Weaviate. |
| **Weavmanage** | One-shot Kubernetes Job that applies Weaviate schema migrations. |
