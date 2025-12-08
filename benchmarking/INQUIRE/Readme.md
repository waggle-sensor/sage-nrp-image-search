# INQUIRE Benchmark

This project uses the same setup as [Hybrid Search](../HybridSearch_example/) so that we can benchmark [Hybrid Search](../HybridSearch_example/) using [INQUIRE](https://github.com/inquire-benchmark/INQUIRE).
>NOTE: The INQUIRE benchmark needs to be updated to work with sage-nrp-image-search. The current code is not compatible with this service.

## Usage

This benchmark is supposed to be used in conjuction with [Hybrid Search](../HybridSearch_example/). The Makefile references components that are deployed in [Hybrid Search](../HybridSearch_example/). The Makefile in here deploys additional containers that are used to run the INQUIRE Benchmark.

## Running the Example

### Prerequisites
To run this example, you'll need:
- **Docker** installed on your machine with GPU access

### Step-by-Step Setup

1. **Spin up your Hybrid Search Instance**:
   - Navigate to the [Hybrid Search](../HybridSearch_example/) directory and follow those instructions to spin up a Hybrid Search Instance.

2. **Load in the dataset**:
   - Navigate back into this directory containing the `Makefile` file and run:
     ```bash
     make build && make load && docker logs inquire_weavloader -f
     ```
     >NOTE: This loads in [INQUIRE-Benchmark-small](https://huggingface.co/datasets/sagecontinuum/INQUIRE-Benchmark-small) into Weaviate.

3. **Calculate the Query Metrics**:
   - After dataset is fully loaded into Weaviate, run:
     ```bash
     make build && make calculate && docker logs inquire_benchmark -f
     ```
     >NOTE: inquire_weavloader's logs will indicate when the dataset is fully loaded into Weaviate.

4. **Retrieve the Results**:
   - After the metrics are calculated, run:
     ```bash
     make get
     ```
     >NOTE: This will copy the csv files into your currect working directory

### Results

Once the benchmark is ran, two csv files will be generated:
- `image_search_results.csv`
    - This file includes the metadata of all images returned by Weaviate when different queries were being ran.
- `query_eval_metrics.csv`
    - This file includes the calculated metrics based on images returned by different queries.

There is multiple results placed in version folders. Each folder has a evaluate.ipynb notebook that goes into more details what that version tested and the metrics.

## References
- [Weaviate Blog: NDCG](https://weaviate.io/blog/retrieval-evaluation-metrics#normalized-discounted-cumulative-gain-ndcg)
- [RAG Evaluation](https://weaviate.io/blog/rag-evaluation)
- [Scikit-Learn NDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [A Guide on NDCG](https://www.aporia.com/learn/a-practical-guide-to-normalized-discounted-cumulative-gain-ndcg/)
- [Weaviate: Batch import](https://weaviate.io/developers/weaviate/manage-data/import)
- [Weaviate: Imports in Detail](https://weaviate.io/developers/weaviate/tutorials/import#data-import---best-practices)
- [INQUIRE](https://inquire-benchmark.github.io/)
- [Hugginface: Fine-tuning Florence2](https://huggingface.co/blog/finetune-florence2)
- [Medium: Fine-tuning Florence2](https://medium.com/@amit25173/fine-tuning-florence-2-aa9c99b2a83d)

## Citation
```
@article{vendrow2024inquire,
  title={INQUIRE: A Natural World Text-to-Image Retrieval Benchmark},
  author={Vendrow, Edward and Pantazis, Omiros and Shepard, Alexander and Brostow, Gabriel and Jones, Kate E and Mac Aodha, Oisin and Beery, Sara and Van Horn, Grant},
  journal={NeurIPS},
  year={2024},
}
```
