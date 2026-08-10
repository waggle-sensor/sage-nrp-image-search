# Contributing

Thank you for contributing to Sage Image Search. This guide covers how to propose changes and where to find planned work.

## How to contribute

### Do not commit directly to `main`

All changes should go through a pull request. The `main` branch is protected and used for deployments and CI image builds.

1. **Fork or branch** — Create a feature branch from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b your-name/short-description
   ```

2. **Make your changes** — Keep diffs focused. Match existing code style and conventions in the area you are editing.

3. **Test locally or on NRP dev environment** when possible:
   - Docker Compose for local stack testing (see [getting-started.md](getting-started.md))
   - NRP dev environment for testing the changes on the NRP dev environment (see [getting-started.md](getting-started.md#nrp-deployment-kubernetes))
   - Run benchmarks (see [benchmarking.md](benchmarking.md))

4. **Open a pull request** against `main` with:
   - A clear summary of what changed and why
   - Notes on how you tested
   - Links to related issues, if any

5. **Address review feedback** — Maintainers may request changes before merge.

6. **Keep your branch up to date** with `main`, especially before testing Kubernetes PR overlays (see [kubernetes/README.md](../kubernetes/README.md#testing-a-pull-request)).

### Pull request guidelines

- **One logical change per PR** when practical — easier to review and revert.
- **Do not commit secrets** — Never add `.env`, `._*-secret.yaml`, tokens, or passwords. Use templates in `kubernetes/base/` and [authentication.md](authentication.md).
- **Document user-facing changes** — Update docs in `docs/` or component READMEs when behavior, configuration, or setup steps change.
- **Benchmark model and search changes** — If you change caption prompts, embedding models, rerankers, or hybrid search parameters, run the [benchmarking suite](../benchmarking/README.md) and note results in the PR.
- **Kubernetes changes** — If you modify manifests, verify with `kubectl kustomize` before applying.

### What happens after merge

GitHub Actions build and push Docker images for changed microservices to the NRP GitLab registry on pushes to `main` and on pull requests. See the root [Readme.md](../Readme.md#cicd).

### Reporting issues

Open a GitHub issue for bugs, unexpected behavior, or feature requests. Include:

- Environment (local Docker Compose vs NRP dev/prod)
- Steps to reproduce
- Relevant logs or error messages
- Expected vs actual behavior

### Areas that need extra care

| Area | Notes |
|------|-------|
| `weavloader/` | Ingestion affects live indexing; test with a small node set when possible |
| `weavmanage/migrations/` | Schema migrations are hard to roll back; coordinate with maintainers |
| `kubernetes/base/` | Shared by dev and prod overlays; test both paths when feasible |
| `app/HyperParameters.py` | Search tuning affects all queries; benchmark before merging |

---

## Roadmap

Planned improvements and research directions. Items are not guaranteed or ordered by priority. Pick something that matches your interests and open a PR — or open an issue first if the scope is large.

### Benchmarking and evaluation

- [ ] Benchmark Milvus@NRP
   - using the [benchmarking suite](../benchmarking/README.md)

### Developer tooling

- [ ] Stand up an MCP server that interacts with the image search Milvus database (`image_search_svc` / `HybridSearchExample` + Dev)
   - so agents (Cursor, Claude, etc.) can list collections, inspect schema, query, and hybrid-search without one-off scripts
   - https://milvus.io/docs/milvus_and_mcp.md
   - related: [zilliztech/mcp-server-milvus](https://github.com/zilliztech/mcp-server-milvus), [Milvus development tools](authentication.md#milvus-development-tools)

### Caption generation

- [ ] Improve the caption that is generated to be more accurate. Measure "accurate" by using the [benchmarking suite](../benchmarking/README.md)
   - [ ] try prompt repitition to see if it can improve the caption generation performance
      - https://arxiv.org/pdf/2512.14982
         - Paper Insights:
            * Repeating the full prompt (`<QUERY><QUERY>`) improves accuracy in many non-reasoning settings.
            * Gains were consistent across multiple major models.
            * It does **not** increase output length or generation latency (only input length).
            * Benefits shrink when explicit reasoning ("think step by step") is enabled.
            * repitition x3 showed that it did even better than x2
            * Repeating a **long, structured prompt** (like our scientific captioning) is more likely to experience gains vs a short simmple instruction.
         - Repetition may improve:
            * Format compliance
            * Keyword count accuracy
            * Constraint adherence
            * It will double input tokens, so cost matters at scale.
      - Remember to add the paper to references section, if you decide to implement this.
   - [ ] try using structured output with the caption generation model to better format the output
   - [ ] Try a Multi-Agent system with a Master agent and expert agents to improve the caption generation performance
      - for example, all images pass through a Master agent that passes the images to the correct expert agent based on the objects in the image.
         - soo if it's an image with smoke the Master agent will pass it to the "Fire Scientist" agent that will then generate the caption.
      - This will allow us to make the prompts be more specific to domains.
         - For example, the prompt for the Fire Scientist agent will be more specific to fire science.
   - maybe this can be used, https://github.com/guidance-ai/guidance
- [ ] try https://github.com/chopratejas/headroom to reduce tokens and be more efficient with the caption generation model
   - remember to benchmark the performance of the caption generation model with headroom using the [benchmarking suite](../benchmarking/README.md)
   - blog about Headroom here: https://www.theregister.com/ai-ml/2026/05/31/netflix-wiz-creates-app-to-slash-ai-bills-then-open-sources-it/5248702
- [ ] try converting the caption generation model to an agent that can be used to generate the caption.
   - the agent will have a harness that will be helpful for the caption generation model to use.
- [ ] try out https://github.com/microsoft/SkillOpt to improve the prompt and skills

### Reranking and retrieval

- [ ] Improve the reranker model to be more accurate. Measure "accurate" by using the [benchmarking suite](../benchmarking/README.md)
   - [ ] switch to reranking with Clip DFN5B-CLIP-ViT-H-14-378
      - before making the switch permanent run the benchmarking suite to see if there are any regressions
      - firebench results show that it is better than the current reranker model (ms-marco-MiniLM-L6-v2)
   - [ ] try an LLM to rerank the results instead of a reranker model.
      - The prompt to the LLM will ask it to rank the results based on the relevance to the query and return a score for each result. Weaviate can then use the scores to rank the results.
   - [ ] look into other reranker models to see if they can improve the reranking performance
- [ ] look into MMR (maximal marginal relevance) to see if it can improve the reranking performance or to implement it as a "toggle" to apply it only to certain queries.
   - https://milvus.io/ai-quick-reference/how-is-diversity-in-search-results-achieved
- [ ] try implementing AgenticRAG to see if it can improve retrieval performance. Measure "performance" by using the [benchmarking suite](../benchmarking/README.md)
   - https://huggingface.co/docs/smolagents/en/examples/rag
   - https://youtu.be/p0FERNkpyHE?si=WV2_0OEwNfFZKwgO

### Safety and observability

- [ ] Integrate ShieldGemma 2 to implement policies and mark images as yes/no if the image violates the policy
   - [ShieldGemma 2 Model Card](https://ai.google.dev/gemma/docs/shieldgemma/model_card_2)
- [ ] add a heartbeat metric for Sage Object Storage (nrdstor)
   - specifically here in the code: https://github.com/waggle-sensor/sage-nrp-image-search/blob/main/weavloader/processing.py#L159
- [ ] add a metric to count the images that have been indexed into the vectordb
   - this answers the question "What is the total amount of images that have been indexed?"

### Benchmarking

- [ ] Create new benchmarks to be added to the [benchmarking suite](../benchmarking/README.md) to test image retrieval in other domains (ex; Urban) & System-Level Performance
   - see [imsearch_benchmarks](https://github.com/waggle-sensor/imsearch_benchmarks) for the existing benchmarks
   - Urban-Focused
      - **CityFlow-NL (Natural Language Vehicle Retrieval):** A benchmark introduced via the AI City Challenge for retrieving traffic camera images of vehicles based on descriptions. Built on the CityFlow surveillance dataset, it provides **5,000+ unique natural language descriptions** for **666 target vehicles** captured across **3,028 multi-camera tracks** in a city. Descriptions include vehicle attributes (color, type), motion (e.g. "turning right"), and surrounding context (other vehicles, road type). *Relevance:* Focused on **urban street scenes** – traffic surveillance footage from a city, featuring cars, trucks, intersections, etc. *Evaluation:* Uses ranking metrics similar to person search – the challenge reports **mAP** (mean average precision) over the top 100 retrieved results, as well as **Recall\@1,5,10** hit rates for each query. For instance, the baseline in one study achieved \~29.6% Recall\@1 and \~64.7% Recall\@10, illustrating the task difficulty. **Access:** Dataset introduced in the *AI City Challenge 2021 (Track 5)*. Available through the challenge organizers (download via the [AI City Challenge website](https://www.aicitychallenge.org/) – data request required) or the authors' GitHub repository which provides code and data links for CityFlow-NL.
         - Paper: https://arxiv.org/abs/2101.04741
         - code: https://github.com/fredfung007/cityflow-nl
   - text extraction benchmarks
      - for example how good can the image search return images based on text found in the image
      - to do this gather lots of images with text in the image and use imsearch_benchmaker to create the benchmark.
   - Compositional & Expert-Level Retrieval Benchmarks
      - **Cola (Compositional Localized Attributes):** A **compositional text-to-image retrieval** benchmark (NeurIPS 2023) designed to test fine-grained understanding of object-attribute combinations. **Cola contains \~1,236 queries** composed of **168 objects and 197 attributes** (e.g. "red car next to blue car", "person in yellow shirt riding a bike") with target images drawn from about **30K images**. Each query has challenging confounders (distractor images that have the right objects but wrong attribute pairing). *Relevance:* Not specific to urban scenes, but many queries could involve everyday objects (cars, people, etc. in various configurations) – useful for evaluating **relational understanding in images**. *Evaluation:* Measures whether the system retrieves the correct image that satisfies the composed query. Metrics include **Recall\@1 (accuracy)** – human performance is \~83% on this benchmark. The goal is to push models to avoid retrieving images that have partial matches (only one attribute-object correct). **Access:** The authors provide a project page and data download (Boston University) – see the [Cola project page](https://cs-people.bu.edu/array/research/cola/) for dataset and instructions.
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

### Knowledge Graphs
What if we build a massive knowledge graph of images and use it to answer scientific questions? This is still a research question, but it is a very interesting one. Maybe it can even help with searching for images.
- https://milvus.io/blog/vector-graph-rag-without-graph-database.md
- https://github.com/Graphify-Labs/graphify

### Performance

- [ ] Utilize batching for Triton for Weavloader to use it
