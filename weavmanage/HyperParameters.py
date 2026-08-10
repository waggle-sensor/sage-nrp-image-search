'''Hyperparameters for Milvus collection / index creation.

These are applied once at schema create time by weavmanage migrations.
Changing them later requires recreating (or reindexing) the collection.

Docs:
  HNSW: https://milvus.io/docs/hnsw.md
  BM25 / full-text: https://milvus.io/docs/full-text-search.md
  Analyzers: https://milvus.io/docs/analyzer-overview.md
  TIMESTAMPTZ: https://milvus.io/docs/timestamptz-field.md
  GEOMETRY: https://milvus.io/docs/geometry-field.md
  RTREE: https://milvus.io/docs/rtree.md
'''

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# CLIP DFN5B-ViT-H-14 projection dim. Must match Triton CLIP output.
vector_dim = 1024

# Allow undeclared fields on insert. Keep False for a strict production schema.
enable_dynamic_field = False

# VARCHAR limits
search_text_max_length = 65535
caption_max_length = 65535
scalar_varchar_max_length = 2048

# TIMESTAMPTZ scalar index (requires Milvus 2.6.6+). Accelerates time filters.
timestamptz_index_type = "STL_SORT"

# GEOMETRY spatial index (requires Milvus 2.6.4+). Accelerates st_* filters.
geometry_index_type = "RTREE"

# ---------------------------------------------------------------------------
# Analyzer (tokenization for BM25 on search_text)
# ---------------------------------------------------------------------------
# Built-in analyzers: "standard", "english", "chinese", ...
# Or custom: {"tokenizer": "standard", "filter": ["lowercase", ...]}
# https://milvus.io/docs/analyzer-overview.md
analyzer_params = {
    "type": "standard",
    # Optional stop-word list for the standard analyzer, e.g. ["a", "an", "the"]
    # "stop_words": [],
}

# ---------------------------------------------------------------------------
# Dense vector index (HNSW on fused CLIP embeddings)
# ---------------------------------------------------------------------------
# Metric: COSINE is correct for L2-normalized CLIP embeddings (IP is equivalent).
dense_metric_type = "COSINE"
dense_index_type = "HNSW"

# Max edges per node. Higher → better recall, more memory, slower build/insert.
# Typical range: [5, 100]. Default in Milvus docs often ~30.
hnsw_M = 16

# Candidate pool during index build. Higher → better graph quality, slower build.
# Typical range: [50, 500].
hnsw_ef_construction = 256

# ---------------------------------------------------------------------------
# Sparse BM25 index (on FunctionType.BM25 output field `sparse`)
# ---------------------------------------------------------------------------
sparse_index_type = "SPARSE_INVERTED_INDEX"
sparse_metric_type = "BM25"

# Query/build algorithm for the inverted index:
#   "DAAT_MAXSCORE" (default) — good for larger k / many query terms
#   "DAAT_WAND"              — good for small k / short queries
#   "TAAT_NAIVE"             — adapts to changing avgdl; slower baseline
#   "BLOCK_MAX_MAXSCORE", "BLOCK_MAX_WAND" — block-max variants (newer Milvus)
bm25_inverted_index_algo = "DAAT_MAXSCORE"

# Term-frequency saturation. Higher → repeated terms count more. Range ~[1.2, 2.0].
bm25_k1 = 1.2

# Document-length normalization. 0 = none, 1 = full. Typical default 0.75.
bm25_b = 0.75
