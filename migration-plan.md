1. Migrate the production and dev environments to milvus: /Users/franciscolozano/.cursor/plans/weaviate_to_milvus_f84dca0c.plan.md
2. Migrate the benchmarking suite to milvus
    - this will be done by adding the milvus vector db adapter to imsearch_eval and switching the benchmark suite to use the milvus vector db adapter.
3. Migrate the init weaviate dataset (hosted in HF Hub) to be a general dataset that can be used for all vector databases.
    - TODO: First I have to open the dataset folder in this workspace