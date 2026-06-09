# Sage Image Search on NRP

This project uses a **Hybrid Search** approach where image captions are generated using **gemma-3-4b-it**, and a search is conducted using both:
1. **Vector Search**: Combining the vector embeddings of both the image and its generated caption.
2. **Keyword Search**: Leveraging the captions of the images for text-based search.

The **Hybrid Search** integrates both search types into one to improve accuracy and retrieval results. After retrieving the objects, they are passed into a **reranker model** to evaluate the relevance of the results based on the context of the query, ensuring that each object is compared more effectively.

---

## Features:
- **gemma-3-4b-it for Caption Generation**: Captions are generated for images using the gemma-3-4b-it model.
- **Vector Search**: Utilizes embeddings of both the images and their captions to perform semantic search.
- **Keyword Search**: Searches are also performed using keywords extracted from image captions.
- **Hybrid Search**: A combination of vector and keyword searches to return the most relevant results.
- **Reranker**: A model that refines the order of search results, ensuring that the most relevant documents or items are ranked higher. It goes beyond the initial retrieval step, considering additional factors such as semantic similarity, context, and other relevant features.

---

### Authentication
For this service to work, you need to have the following credentials:
- **SAGE_USER**: Your SAGE username
- **SAGE_TOKEN**: Your SAGE token
- **HF_TOKEN**: Your Hugging Face token

Your Sage credentials needs access to images on Sage. Any images that you don't have access to will be skipped.

Your Hugging Face token needs access to the models that are used in this service.

---

## CI/CD Workflow: Build & Push Images
This repository includes a GitHub Action that builds and pushes Docker images for all Hybrid Image Search microservices to NRPs public image registry. The workflow runs automatically on pushes to the main branch and on pull requests, detecting changes and publishing updated service images to the configured container registry.

---

## Docker compose
envs:
```
cp .env.example .env
```
Make sure to fill in the secrets (top three env vars)

Run:
```
docker compose up -d --build
```

Clean up:
```
docker compose down
```

All together:
```
docker compose down && docker compose up -d --build
```

Clean up (volumes):
```
docker compose down --volumes
```

Notes:
- Triton migh not be able load either one of the models (CLIP and gemma3) or for some reason OSErrors loading the model weights so this is a workaround to download the models to your local directory and then move them to the container:
   ```
   source .env #assumes that HF_TOKEN is set
   cd triton
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   huggingface-cli download --local-dir clip --revision "$CLIP_MODEL_VERSION" "$CLIP_HF_REPO"

   huggingface-cli download --local-dir gemma-3-4b-it --revision "$GEMMA_MODEL_VERSION" google/gemma-3-4b-it

   docker cp clip/. sage-nrp-image-search-triton-1:/models/clip/
   docker cp gemma-3-4b-it sage-nrp-image-search-triton-1:/models/
   ```

---

## Kubernetes
Developed and test with these versions for k8s and kustomize:
```
Client Version: v1.29.1
Kustomize Version: v5.0.4
```

Create k8s secrets for Sage credentials by editing the `sage-user-secret.yaml` file.

Create k8s secrets for Hugging Face credentials by editing the `huggingface-secret.yaml` file.

Deploy all services:
```
kubectl apply -k nrp-dev or nrp-prod
```
Delete all services:
```
kubectl delete -k nrp-dev or nrp-prod
```
Debugging - output to yaml:
```
kubectl kustomize nrp-dev -o sage-image-search-dev.yaml or kubectl kustomize nrp-prod -o sage-image-search-prod.yaml
```

---

## Workflow Overview

1. **Caption Generation with gemma-3-4b-it**:
   - The **gemma-3-4b-it** model generates captions for images, allowing for both semantic and keyword-based search.
   
2. **Vector Search**:
   - The embeddings of the images and their captions are stored in **Weaviate**. When a query is made, the relevant vectors are retrieved using similarity search (e.g., cosine similarity).

3. **Keyword Search**:
   - The captions are indexed and can be searched with keywords. This enables traditional text-based search capabilities (e.g., bm25 algorithm).

4. **Hybrid Search**:
   - A **hybrid search** combines the results from both the **vector search** and the **keyword search**. This improves result relevance by considering both semantic similarity and exact text matches.

5. **Reranking**:
   - After retrieving the results, a reranker model evaluates them against the original query. This model takes into account **context** to ensure that the most relevant and accurate results are returned.

---

## References

- **Weaviate Documentation**:  
   - [Managing Data in Weaviate](https://weaviate.io/developers/weaviate/manage-data)
   - [Named Vectors](https://weaviate.io/developers/weaviate/config-refs/schema/multi-vector)
   - [Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid)
   - [Hybrid Search Blog](https://weaviate.io/blog/hybrid-search-fusion-algorithms)
   - [Model Providers: ImageBind](https://weaviate.io/developers/weaviate/model-providers/imagebind/embeddings-multimodal)
   - [ImageBind Repo](https://github.com/weaviate/multi2vec-bind-inference)
   - [Reranker](https://weaviate.io/developers/weaviate/model-providers/transformers/reranker)
   - [Reranker Model Blog](https://weaviate.io/blog/ranking-models-for-better-search)
   - [Reranker Model: ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
   - [Reranker Repo v1.1.1](https://github.com/weaviate/reranker-transformers/tree/1.1.1)
   - [Vector Indexes](https://weaviate.io/developers/weaviate/config-refs/schema/vector-index)
   - [ANN Benchmarks](https://weaviate.io/developers/weaviate/benchmarks/ann)
   - [Env Variables](https://weaviate.io/developers/weaviate/config-refs/env-vars)
   - [Multi-vector Embeddings](https://weaviate.io/developers/weaviate/tutorials/multi-vector-embeddings?utm_source=newsletter.weaviate.io&utm_medium=referral&utm_campaign=weaviate-mcp-server-april-events-and-more-agents#option-2-user-provided-embeddings)
   

- **Triton Documentation**:
   - [Triton Server](https://github.com/triton-inference-server/server)
   - [Triton Server Tutorials](https://github.com/triton-inference-server/tutorials)
   - [Triton Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix)
   - [Triton Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
   - 
---

## TODOs
- [ ] Benchmark existing deployment using new framework
   - using...
      - https://huggingface.co/datasets/sagecontinuum/INQUIRE-Benchmark-small
      - https://huggingface.co/datasets/sagecontinuum/FireBench
      - ...
- [ ] try prompt repitition to see if it can improve the caption generation performance
   - https://arxiv.org/pdf/2512.14982
      - Paper Insights:
         * Repeating the full prompt (`<QUERY><QUERY>`) improves accuracy in many non-reasoning settings.
         * Gains were consistent across multiple major models.
         * It does **not** increase output length or generation latency (only input length).
         * Benefits shrink when explicit reasoning (“think step by step”) is enabled.
         * repitition x3 showed that it did even better than x2
         * Repeating a **long, structured prompt** (like our scientific captioning) is more likely to experience gains vs a short simmple instruction.
      - Repetition may improve:
         * Format compliance
         * Keyword count accuracy
         * Constraint adherence
         * It will double input tokens, so cost matters at scale.
   - Remember to add the paper to references section, if you decide to implement this.
- [ ] look into using text encoders only to see if just using caption-query comparisons can be enough or improve retrieval with embeddings. Essentially the image will NOT be embedded in the same vector space as the captions anymore.
   - embeddinggemma model: https://huggingface.co/google/embeddinggemma-300m
   - E5-mistral-7b-instruct: https://huggingface.co/intfloat/e5-mistral-7b-instruct
      - this is hosted by NRP so it will be easy to use.
- [ ] add structured output to the caption generation model to better format the output
   - maybe this can be used, https://github.com/guidance-ai/guidance
- [ ] Bechmark Milvus@NRP
   - using...
      - https://huggingface.co/datasets/sagecontinuum/INQUIRE-Benchmark-small
      - https://huggingface.co/datasets/sagecontinuum/FireBench
      - ...
- [ ] switch to reranking with Clip DFN5B-CLIP-ViT-H-14-378
   - before making the switch permanent run the benchmarking suite to see if there are any regressions
   - firebench results show that it is better than the current reranker model (ms-marco-MiniLM-L6-v2)
- [ ] look into MMR (maximal marginal relevance) to see if it can improve the reranking performance or to implement it as a "toggle" to apply it only to certain queries.
   - https://milvus.io/ai-quick-reference/how-is-diversity-in-search-results-achieved
- [ ] Integrate ShieldGemma 2 to implement policies and mark images as yes/no if the image violates the policy
   - [ShieldGemma 2 Model Card](https://ai.google.dev/gemma/docs/shieldgemma/model_card_2)
- [ ] add a heartbeat metric for Sage Object Storage (nrdstor)
   - specifically here in the code: https://github.com/waggle-sensor/sage-nrp-image-search/blob/main/weavloader/processing.py#L159
- [ ] add a metric to count the images that have been indexed into the vectordb
   - this answers the question "What is the total amount of images that have been indexed?"
- [ ] Use other benchmarks to test image retrieval in other domains (ex; Urban) & System-Level Performance
   - see [imsearch_benchmarks](https://github.com/waggle-sensor/imsearch_benchmarks) for the existing benchmarks
   - Sage focused
      - get a sample of images and create queries based on the metadata. For example, "animals in W09E"
      - this can also just be images from sage so it can truly test the image retrieval capabilities of the system on real data.
   - Urban-Focused
      - **CityFlow-NL (Natural Language Vehicle Retrieval):** A benchmark introduced via the AI City Challenge for retrieving traffic camera images of vehicles based on descriptions. Built on the CityFlow surveillance dataset, it provides **5,000+ unique natural language descriptions** for **666 target vehicles** captured across **3,028 multi-camera tracks** in a city. Descriptions include vehicle attributes (color, type), motion (e.g. “turning right”), and surrounding context (other vehicles, road type). *Relevance:* Focused on **urban street scenes** – traffic surveillance footage from a city, featuring cars, trucks, intersections, etc. *Evaluation:* Uses ranking metrics similar to person search – the challenge reports **mAP** (mean average precision) over the top 100 retrieved results, as well as **Recall\@1,5,10** hit rates for each query. For instance, the baseline in one study achieved \~29.6% Recall\@1 and \~64.7% Recall\@10, illustrating the task difficulty. **Access:** Dataset introduced in the *AI City Challenge 2021 (Track 5)*. Available through the challenge organizers (download via the [AI City Challenge website](https://www.aicitychallenge.org/) – data request required) or the authors’ GitHub repository which provides code and data links for CityFlow-NL.
         - Paper: https://arxiv.org/abs/2101.04741
         - code: https://github.com/fredfung007/cityflow-nl
   - text extraction benchmarks
      - for example how good can the image search return images based on text found in the image
      - to do this gather lots of images with text in the image and use imsearch_benchmaker to create the benchmark.
   - Compositional & Expert-Level Retrieval Benchmarks
      - **Cola (Compositional Localized Attributes):** A **compositional text-to-image retrieval** benchmark (NeurIPS 2023) designed to test fine-grained understanding of object-attribute combinations. **Cola contains \~1,236 queries** composed of **168 objects and 197 attributes** (e.g. “red car next to blue car”, “person in yellow shirt riding a bike”) with target images drawn from about **30K images**. Each query has challenging confounders (distractor images that have the right objects but wrong attribute pairing). *Relevance:* Not specific to urban scenes, but many queries could involve everyday objects (cars, people, etc. in various configurations) – useful for evaluating **relational understanding in images**. *Evaluation:* Measures whether the system retrieves the correct image that satisfies the composed query. Metrics include **Recall\@1 (accuracy)** – human performance is \~83% on this benchmark. The goal is to push models to avoid retrieving images that have partial matches (only one attribute-object correct). **Access:** The authors provide a project page and data download (Boston University) – see the [Cola project page](https://cs-people.bu.edu/array/research/cola/) for dataset and instructions.
   - Geographical Focused
      - https://www.flickr.com/groups/geographical_landforms/pool/
         * **Description and purpose:** A collection of images of geographical landforms, including mountains, rivers, oceans, and other natural features.
   - Atmospheric Science Focused (Focusing on weather)
      - I dont have a dataset for this yet
   - Catastrophe Focused
      - https://arxiv.org/abs/2201.04236
        * **Description and purpose:** A dataset of images of catastrophes, including earthquakes, floods, fires, etc.
   - System-Level Performance Benchmarks
      - Latency
         - Time taken per query (cold start vs. warm cache)
         - Breakdown: captioning time, vector embedding, fusion, reranking, search
      - Throughput
         - Number of queries processed per second/minute
         - Use Locust, JMeter, or k6 for load testing
      - Scalability
         - Horizontal (multiple Weaviate shards, vector databases, reranker replicas)
         - Measure with increased concurrent queries, dataset size growth
      - Resource Usage
         - CPU, RAM, disk (capture the image size), and GPU usage per component (captioner, embedder, Weaviate, reranker)
         - Use tools like Prometheus + Grafana, htop, nvidia-smi
      - Cold Start Time
         - How long to become operational from scratch?
         - Important for containerized deployments
      - examples here: https://chatgpt.com/c/684b1286-1144-8003-8a20-85a1045375c3
   - Indexing and Update Benchmarks
      - Indexing Time
         - How long to ingest N images and generate embeddings/captions?
         - Parallelization efficiency
         - use Weaviate Benchmarks CLI
      - Incremental Update Latency
         - Time between new image upload and being searchable
      - examples here: https://chatgpt.com/c/684b1286-1144-8003-8a20-85a1045375c3
- [ ] turn on batching for triton and utilize it in weavloader
