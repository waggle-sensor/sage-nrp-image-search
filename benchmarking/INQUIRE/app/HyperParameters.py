'''This file contains the hyper parameters that can be changed to fine tune
the system. 
NOTE: Not all params have been added here. More in depth search must be 
done to find more hyper params that can be altered'''

# 1) Hybrid Search Query hyperparameters
response_limit=50 #Number of objects to return, switched from 0 to 50 to match how INQUIRE benchmarks
query_alpha=0.4 #An alpha of 1 is a pure vector search, An alpha of 0 is a pure keyword search.
max_vector_distance=0.4 #max accepted distance for the vector search component
near_text_certainty=0.7 #The minimum similarity score to return. If not specified, the default certainty specified by the server is used.
#NOTE: USE max_vector_distance OR near_text_certainty
concepts_to_avoid=["police", "gun"] # Concepts to avoid
avoid_concepts_force=0 #the strength to avoid the concepts
# autocut limits results based on discontinuities
# more info: https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut
autocut_jumps=1 #To explicitly disable autocut, set the number of jumps to 0 or a negative value
#NOTE: USE autocut_jumps OR response_limit
hybrid_weight=0.7 #The weight of the hybrid search component in the unified score for hybrid colbert blend.
colbert_weight=0.3 #The weight of the colbert search component in the unified score for hybrid colbert blend.
hybrid_colbert_blend_top_k=50 #The number of top results to return from the hybrid colbert blend search.

# 2) Experimental hyperparameters
align_alpha = 0.7
clip_alpha = 0.7