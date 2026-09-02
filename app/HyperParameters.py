'''This file contains the hyper parameters that can be changed to fine tune
the system. 
NOTE: Not all params have been added here. More in depth search must be 
done to find more hyper params that can be altered'''
#NOTE: The hyperparameters will be split up based on what microservice it corresponds to. Or I can
#   make all the microservices apart of the same deployment so the HPs continue to be easily managed
#   and don't get split up.

#TODO: Grab a big enough sample set to test a real deployment so you can fine tune the HPs
#  NOTE: instead of recreating the db just update the HPs when testing

# 1) Hybrid Search Query hyperparameters
response_limit=25 #Number of objects to return
query_alpha=0.65 # Dense vs BM25: 1 = pure dense (image+caption), 0 = pure keyword (BM25).
# 2) CLIP modality mix at query time (not index-time fusion)
clip_alpha = 0.7 # Within dense: 1 = image_vector only, 0 = caption_vector only.
# v16 H defaults → WeightedRanker 46% image / 20% caption / 35% BM25.
# WeightedRanker(
#     query_alpha * clip_alpha,           # image_vector
#     query_alpha * (1.0 - clip_alpha),   # caption_vector
#     1.0 - query_alpha,                  # sparse BM25
# )

