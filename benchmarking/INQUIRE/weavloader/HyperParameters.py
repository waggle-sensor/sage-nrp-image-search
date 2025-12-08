'''This file contains the hyper parameters that can be changed to fine tune
the system. 
NOTE: Not all params have been added here. More in depth search must be 
done to find more hyper params that can be altered'''

from weaviate.classes.config import VectorDistances, Configure
from weaviate.collections.classes.config_vector_index import VectorFilterStrategy

# 1) Weaviate module multi2vec-bind (Imagebind) weights
textWeight = 0.3
imageWeight = 0.7 # Increase the weighting here so that the embedding is more influenced by the image
audioWeight = 0 # Currently not being used
videoWeight = 0 # Currently not being used

# 2) Hierarchical Navigable Small World (hnsw) for Approximate Nearest Neighbor (ANN) hyperparamaters
# used hsnw since it works well with bigger datasets
# more info: https://weaviate.io/developers/weaviate/config-refs/schema/vector-index#hnsw-indexes
# configuration tips: https://weaviate.io/developers/weaviate/config-refs/schema/vector-index#hnsw-configuration-tips
# helpful article: https://gagan-mehta.medium.com/efficient-resource-understanding-and-planning-in-weaviate-ec673f065e86
hnsw_dist_metric=VectorDistances.COSINE
hnsw_ef=-1 #Balance search speed and recall, Weaviate automatically adjusts the ef value and creates a dynamic ef list when ef is set to -1
hnsw_ef_construction=100 #Balance index search speed and build speed. Changed from 128 to 100
hnsw_maxConnections=50 #Maximum number of connections per element. Changed from 32 to 50
hsnw_dynamicEfMax=500 #Upper bound for dynamic ef
hsnw_dynamicEfMin=200 #Lower bound for dynamic ef. Changed from 100 to 200
hnsw_ef_factor=20 #This setting is only used when hnsw_ef is -1, Sets the potential length of the search list. Changed from 8 to 20
hsnw_filterStrategy=VectorFilterStrategy.ACORN #The filter strategy to use for filtering the search results.
hnsw_flatSearchCutoff=40000 #cutoff to automatically switch to a flat (brute-force) vector search when a filter becomes too restrictive
hnsw_vector_cache_max_objects=1e12 #Maximum number of objects in the memory cache
# Auto Product Quantization (PQ)
#  https://weaviate.io/developers/weaviate/configuration/compression/pq-compression
hnsw_quantizer=Configure.VectorIndex.Quantizer.pq(
    training_limit=500000 #threshold to begin training
)

# 3) Experimental hyperparameters
align_alpha = 0.7
clip_alpha = 0.7
qwen2_5_prompt="""
role:
You are a world-class Scientific Image Captioning Expert.

context:
You will be shown a scientific image captured by edge devices. Your goal is to analyze its content and significance in detail. 

task:
Generate exactly one scientifically detailed caption that accurately describes what is visible in the image and its scientific relevance. 
Make it as detailed as possible. Also extract text and numbers from the images.

constraints:
- Only return:
  1. A single caption.
  2. a list of 15 keywords relevant to the image.
- Do not include any additional text, explanations, or formatting.

format:
  caption: <your_scientific_caption_here>
  keywords: <keyword1>, <keyword2>, ...
"""
gemma3_prompt=qwen2_5_prompt

# 3) Weaviate module reranker-transformers (ms-marco-MiniLM-L-6-v2 Reranker Model)
# Model info: https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2
# NOTE: there is no HPs I can change in this module