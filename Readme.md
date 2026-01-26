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
- [ ] Bechmark Milvus@NRP
   - using...
      - https://huggingface.co/datasets/sagecontinuum/INQUIRE-Benchmark-small
      - https://huggingface.co/datasets/sagecontinuum/FireBench
      - ...
- [ ] add a heartbeat metric for Sage Object Storage (nrdstor)
   - specifically here in the code: https://github.com/waggle-sensor/sage-nrp-image-search/blob/main/weavloader/processing.py#L159
- [ ] Use other benchmarks to test image retrieval in other domains (ex; Urban) & System-Level Performance
   - General Image-Caption Retrieval Benchmarks
      - **MS COCO Captions:** A widely used benchmark drawn from the MS-COCO dataset (Common Objects in Context). It contains **123,287 images** covering everyday scenes (including many urban street scenes with people, vehicles, buildings, etc.), each paired with 5 human-written captions. The standard split is \~82k images for training, 5k for validation, 5k for testing. *Relevance:* Although not exclusively urban, COCO features many city context images (e.g. street traffic, city parks, indoor scenes). *Evaluation:* Typically uses **Recall\@K** (K=1,5,10) as the primary metric – e.g. the percentage of queries for which the correct image is in the top K results. Some works also report mean average precision (mAP) on the 5K test set. **Access:** [COCO Dataset Page](https://cocodataset.org/#download) (captions and images are publicly downloadable).
      - **Flickr30K:** Another popular benchmark with **31,000 photographs** from Flickr, each image paired with 5 crowd-sourced textual descriptions. It is split into 29k images for train, 1k for validation, 1k for test. *Relevance:* Images cover a broad range of everyday situations (some urban, some rural, people and objects in various settings). *Evaluation:* Uses the same **Recall\@K** metrics as COCO (often evaluating Recall\@1, 5, 10 for text→image retrieval). Models today achieve high performance (e.g. near 99% recall\@10 for top methods). **Access:** Available via [Kaggle dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) or the original authors’ webpage (University of Illinois).
      *(**Note:** Flickr8K is an older, smaller dataset with 8,000 images and captions, now less commonly used in benchmarks.)*
      - **NUS-WIDE:** A large-scale **web image dataset** (269,648 Flickr images) with associated **user tags and 81 high-level concepts** annotated. While not caption-based, it is a standard benchmark for text-to-image retrieval using tags or keywords. Many concepts are object or scene categories (e.g. *building, car, street, person*, etc.), making it relevant for urban imagery retrieval. *Evaluation:* Typically uses **mean Average Precision (mAP)** over all queries, since multiple images can be relevant for a given tag query. NUS-WIDE is often used for evaluating cross-modal retrieval and hashing methods. **Access:** [NUS-WIDE on Kaggle](https://www.kaggle.com/datasets/xinleili/nuswide) (contains the images and annotations).
   - Urban-Focused
      - **CityFlow-NL (Natural Language Vehicle Retrieval):** A benchmark introduced via the AI City Challenge for retrieving traffic camera images of vehicles based on descriptions. Built on the CityFlow surveillance dataset, it provides **5,000+ unique natural language descriptions** for **666 target vehicles** captured across **3,028 multi-camera tracks** in a city. Descriptions include vehicle attributes (color, type), motion (e.g. “turning right”), and surrounding context (other vehicles, road type). *Relevance:* Focused on **urban street scenes** – traffic surveillance footage from a city, featuring cars, trucks, intersections, etc. *Evaluation:* Uses ranking metrics similar to person search – the challenge reports **mAP** (mean average precision) over the top 100 retrieved results, as well as **Recall\@1,5,10** hit rates for each query. For instance, the baseline in one study achieved \~29.6% Recall\@1 and \~64.7% Recall\@10, illustrating the task difficulty. **Access:** Dataset introduced in the *AI City Challenge 2021 (Track 5)*. Available through the challenge organizers (download via the [AI City Challenge website](https://www.aicitychallenge.org/) – data request required) or the authors’ GitHub repository which provides code and data links for CityFlow-NL.
         - Paper: https://arxiv.org/abs/2101.04741
         - code: https://github.com/fredfung007/cityflow-nl
   - Compositional & Expert-Level Retrieval Benchmarks
      - **Cola (Compositional Localized Attributes):** A **compositional text-to-image retrieval** benchmark (NeurIPS 2023) designed to test fine-grained understanding of object-attribute combinations. **Cola contains \~1,236 queries** composed of **168 objects and 197 attributes** (e.g. “red car next to blue car”, “person in yellow shirt riding a bike”) with target images drawn from about **30K images**. Each query has challenging confounders (distractor images that have the right objects but wrong attribute pairing). *Relevance:* Not specific to urban scenes, but many queries could involve everyday objects (cars, people, etc. in various configurations) – useful for evaluating **relational understanding in images**. *Evaluation:* Measures whether the system retrieves the correct image that satisfies the composed query. Metrics include **Recall\@1 (accuracy)** – human performance is \~83% on this benchmark. The goal is to push models to avoid retrieving images that have partial matches (only one attribute-object correct). **Access:** The authors provide a project page and data download (Boston University) – see the [Cola project page](https://cs-people.bu.edu/array/research/cola/) for dataset and instructions.
   - Atmospheric Science Focused
      - Multimodal Ground‐based Cloud Dataset (MGCD)
         * **Description and purpose:** A dataset of 8,000 ground-based hemispheric sky images collected in Tianjin, China (2017–2018) for cloud classification research. It includes seven cloud categories (grouped per WMO classification) such as cumulus, altocumulus/cirrocumulus, cirrus/cirrostratus, clear sky, stratocumulus/stratus/altostratus, cumulonimbus/nimbostratus, and mixed cloud. The dataset was created to improve automated cloud-type recognition and is labeled by meteorologists, ensuring high-quality ground truth.
         * **Camera type:** All-sky camera with a fisheye lens (ground-based hemispheric imager). This captures the full sky dome, providing a wide-angle view of cloud cover.
         * **Size and format:** 8,000 images in JPEG format at 1024×1024 pixel resolution (split into 4,000 training and 4,000 testing). Each image is paired with concurrent meteorological sensor readings (temperature, humidity, pressure, wind speed) stored as a 4-element vector.
         * **Type of annotations:** Each image is annotated with a **cloud category label** (one of the seven sky types) provided by experts. In addition, **numeric weather data** from a co-located station is included as auxiliary information. These human-defined labels serve as textual metadata (e.g. “cumulus” or “clear sky”) for retrieval tasks.
         * **Relevance to retrieval:** Highly relevant for weather/cloud queries – for example, a text query “cumulonimbus cloud” can be validated against images labeled *Cb* (cumulonimbus) in this set. The expert labels and broad sky coverage make it suitable as ground truth for text-to-image retrieval of cloud conditions.
         * **Access link:** Available free for research (under a data use agreement). **Download:** The authors provide a Google Drive link after agreeing to the MGCD terms (see the MGCD GitHub page for instructions).
      - TJNU Ground-Based Cloud Dataset (GCD)
         * **Description and purpose:** A large-scale cloud image dataset of 19,000 ground-based sky images collected across nine provinces in China (2019–2020). It was built to improve cloud classification robustness under diverse climatic regions. Like MGCD, it covers seven cloud/weather sky types (cumulus; altocumulus/cirrocumulus; cirrus/cirrostratus; clear sky; stratocumulus/stratus/altostratus; cumulonimbus/nimbostratus; mixed) annotated per WMO cloud classification guidelines. This curated dataset expands coverage of cloud appearances for research in automated cloud recognition.
         * **Camera type:** Ground-based **camera sensors** pointed at the sky (wide-field view). The exact lens type isn’t explicitly fisheye, but the images cover broad sky regions. The multi-location setup ensured varied atmospheric conditions (from coastal to inland).
         * **Size and format:** 19,000 JPEG images at 512×512 pixel resolution. The data are split into 10,000 training and 9,000 testing images. All images have been resized for uniformity (original cameras likely higher resolution).
         * **Type of annotations:** Each image has a **human-assigned cloud category label** (one of the seven sky condition classes). Labels were assigned collaboratively by meteorologists and cloud researchers, ensuring reliable textual tags for each image. No free-text captions are provided, but the standardized labels (e.g. “altocumulus”) serve as descriptive metadata.
         * **Relevance to retrieval:** Useful for text queries about cloud formations or sky conditions. The labels (covering multiple cloud genera and clear sky) match common weather descriptors, aiding retrieval testing (e.g. querying “stratus clouds” should retrieve images labeled *stratus/altostratus*). The wide geographic and temporal coverage improves the robustness of retrieval evaluations for various atmospheric conditions.
         * **Access link:** Provided freely for research with a usage agreement. **Download:** Via a Google Drive link after accepting the GCD license (refer to the TJNU GCD GitHub page for the agreement and download link).
      - Cirrus Cumulus Stratus Nimbus (CCSN) Cloud Database
         * **Description and purpose:** The **CCSN database** is a ground-based cloud image dataset containing 2,543 images categorized into 11 classes according to the World Meteorological Organization’s cloud genera (the ten standard cloud types) plus aircraft contrails. This dataset was introduced by Zhang *et al.* (2018) to advance cloud classification, notably being the first to include **contrail** (artificial cloud) as a class. It serves as a reference benchmark (named in the *CloudNet* paper) for evaluating algorithms on fine-grained cloud type recognition under meteorological standards.
         * **Camera type:** Ground-based sky imagers (likely all-sky or wide-angle cameras). The images are of clouds as seen from the ground; however, they have been preprocessed to a uniform small size (suggesting they may be patches or resized whole images). The original capture device isn’t explicitly stated, but the data represent typical sky views including horizon and zenith perspectives.
         * **Size and format:** 2,543 color images in JPEG format, each **fixed at 256×256 pixels**. The relatively low resolution indicates images were scaled or cropped for model training consistency. Despite the size, the dataset covers all major cloud formations (Cirrus, Cumulus, Stratus, etc., totaling 11 categories).
         * **Type of annotations:** Each image is labeled with a **cloud type genus** (e.g. Cu, Cb, Ci, St, etc.), corresponding to human-identified cloud categories. These labels are textual abbreviations (expanded in metadata to full names like “cumulonimbus”) and serve as ground truth tags. The inclusion of “Ct” for contrail is noteworthy, capturing a human-observed atmospheric phenomenon. No detailed sentences are provided, just the single-category tags per image.
         * **Relevance to retrieval:** Directly relevant for queries on specific cloud types. A retrieval system can be tested by using cloud genus names or descriptions (“nimbostratus cloud”, “aircraft contrail in sky”) and checking if images from the matching CCSN category are returned. The dataset’s strict adherence to meteorological cloud types makes it ideal for validating fine-grained weather image retrieval and classification.
         * **Download/access link:** **Publicly available** via Harvard Dataverse. The dataset can be downloaded from its DOI link (no login required). The project’s GitHub page also provides the DOI and originally required a sign-up form (now deprecated).
      - SWIMCAT-Ext (Extended Sky Image Categories)
         * **Description and purpose:** **SWIMCAT-ext** is an expanded version of the SWIMCAT dataset, published in 2020, that provides a larger and more diverse set of sky images. It consists of 2,100 sky/cloud images divided into 6 classes. The classes extend the original SWIMCAT by splitting cloud types more finely: *clear sky*, *patterned clouds*, *thick dark clouds*, ***thin** white clouds*, *thick white clouds*, and *veil clouds*. This extension was created to improve training of cloud classification models by providing more samples per category and a new “thin white clouds” class to distinguish faint cloud layers.
         * **Camera type:** Unlike the original SWIMCAT, the images in SWIMCAT-ext were **collected from the Internet** (various sources) rather than a single fisheye camera. They are ground-based shots of the sky but may not all be full hemispheric views – likely a mix of wide-angle photographs capturing the sky. All images were vetted and labeled by a technical expert to ensure they match the defined categories.
         * **Size and format:** 2,100 images (likely JPEG). The description does not specify resolution, but since they are web-sourced, resolutions may vary (possibly standardized during preprocessing). The focus is on quality and category balance rather than uniform size.
         * **Type of annotations:** Each image is annotated with one of six **human-defined labels** corresponding to cloud/sky appearance. The labels are natural-language category names (“clear sky”, “thin white clouds”, etc.), serving as metadata. These are effectively short captions describing the weather state in the image.
         * **Relevance to retrieval:** The dataset provides a rich set of real-world sky images with clear textual labels, ideal for testing text-to-image retrieval. For example, one can query “thin white clouds” and expect the system to retrieve images from this class. Since the images come from varied sources, it also tests retrieval robustness across different viewpoints and camera types under the same semantic category.
         * **Access link:** **Download via Mendeley Data:** SWIMCAT-ext is published openly (CC BY 4.0). The dataset can be downloaded from its Mendeley Data DOI link without special permission.
      - Weather Phenomena Image Dataset (Kaggle “Weather Image Recognition”)
         * **Description and purpose:** A comprehensive image dataset focusing on various **atmospheric weather phenomena**, compiled for weather condition recognition tasks. It contains **6,862 images** spanning 11 classes of weather events/conditions. Notably, these classes include phenomena such as *fog/smog, rain, snow, lightning, hail, dew, frost, glaze (ice), rainbow, rime* (ice frost), and *sandstorm*. The dataset’s purpose is to enable and evaluate classification and retrieval of images based on weather descriptions, especially for severe or visually distinctive events. It was featured on Kaggle to encourage machine learning projects in weather image classification.
         * **Camera type:** All images are **ground-based photographs** taken by people (crowdsourced or scraped from the web), typically showing outdoor scenes under specific weather conditions. The camera types vary – from regular consumer cameras or phones capturing phenomena (e.g. a landscape during a fog, a thunderstorm sky for lightning). Many are wide-angle shots of the outdoors, but not fish-eye or specialized instruments. They often include portions of landscape along with the sky, giving context to the weather event (e.g. ground covered in frost, or a lightning bolt against the sky).
         * **Size and format:** 6,862 images in common image formats (JPEG/PNG). Image resolutions vary, but they are generally of decent quality for recognition tasks. The dataset is organized into class-specific subfolders (one for each weather type). This structure facilitates retrieval by category or training classifiers.
         * **Type of annotations:** Each image is labeled with a **weather condition tag** corresponding to one of the 11 classes. These tags are human-readable descriptors (e.g., “rain” or “sandstorm”). In some cases, multiple phenomena might co-occur (like rain with lightning), but in this dataset each image is categorized by its primary phenomenon. The annotations are structured (one label per image) but effectively serve as short text descriptions of the image’s content (the weather event present).
         * **Relevance to retrieval:** This dataset directly supports text-to-image retrieval scenarios for weather events. For example, a query “lightning storm” or “dense fog” would correspond to the *lightning* or *fog/smog* categories, and relevant images can be retrieved and evaluated. Because it covers a wide array of weather phenomena (including hazardous events like hail and sandstorms), it’s valuable for testing retrieval across both common and relatively rare atmospheric conditions. The human-chosen labels act as ground truth keywords for evaluating retrieval accuracy.
         * **Download/access link:** Available on **Kaggle** (dataset titled “Weather Image Recognition”). Users can download it directly from the Kaggle page. Additionally, a GitHub repository by an author of the project provides the class breakdown and can be used as a reference for accessing the data. (Kaggle login may be required to access the files.)
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
- [ ] Update readme on new implementations
   - there is components that got replaced with new models
- [ ] turn on batching for triton and utilize it in weavloader
- [ ] Integrate ShieldGemma 2 to implement policies and mark images as yes/no if the image violates the policy
   - [ShieldGemma 2 Model Card](https://ai.google.dev/gemma/docs/shieldgemma/model_card_2)
