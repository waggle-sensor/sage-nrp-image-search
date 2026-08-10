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
query_alpha=0.4 #An alpha of 1 is a pure vector search, An alpha of 0 is a pure keyword search.
# WeightedRanker(query_alpha, 1 - query_alpha) → dense weight, sparse/BM25 weight

# 2) CLIP fusion at query/ingest time
clip_alpha = 0.7
