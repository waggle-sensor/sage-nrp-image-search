## Summary: Evaluation Metrics for Sage Image Search (Binary Labels, Top-25 Retrieval)

### 1️⃣ Keep K Fixed

* Always evaluate at **K = response_limit**, since the system returns the top `response_limit` results.
>NOTE: at this moment, the response limit is 25. This may change in the future. - 03/01/2026
* Do **not** change K per query based on how many relevant items exist.
* Fixed K ensures:
  * Fair comparison across queries
  * Fair comparison across model versions
  * Reproducibility

---

### 2️⃣ Problem Context

* Labels are **binary (relevant / non-relevant)**.
* The **average number of relevant items per query varies across benchmarks** (fire, urban, atmospheric, biology).
* Relevant items are often sparse compared to total corpus size.
* Early ranking quality matters more than full-corpus coverage.

---

### 3️⃣ Recommended Metric Strategy

Use a combination of ranking-sensitive and top-K metrics:

### Primary Decision Metrics

* **MRR (Mean Reciprocal Rank)**

  * Measures how early the first relevant result appears.
  * Very stable when relevant counts are small.
  * Strong indicator of ranking quality.
>NOTE: imsearch_eval returns reciprocal rank for each query so we need to take the mean of the reciprocal rank for all queries to get the MRR.

* **Success@25 (Hit Rate@25)**

  * Measures whether at least one relevant result appears in the top 25.
  * Easy to interpret for stakeholders.
  * Reflects practical user satisfaction.
>NOTE: imsearch_eval returns a binary value for each query so we need to take the mean of the binary value for all queries to get the Success@25.

---

### Supporting Metrics

* **Precision@25**

  * Measures how clean the first page of results is.
  * Directly aligned with user-visible output.

* **NDCG@25**

  * Evaluates ranking quality with position discounting.
  * Useful even with binary labels.

* **Recall@25**

  * Measures how much of the relevant set appears in the top 25.
  * Interpreted carefully when relevant counts vary across benchmarks.

* **Diversity@25**

  * Measures the variety of relevant results in the top 25.
  * Encourages exploration of different relevant items.
>NOTE: If you are interested in "spread" of results, Diversity is a primary metric.

---

### 4️⃣ Interpretation Guidance

* Precision reflects first-page quality.
* MRR reflects early ranking strength.
* Success@25 reflects practical usefulness.
* NDCG captures ranking structure.
* Diversity@25 encourages exploration of different relevant items.
* Metric comparisons should primarily be made:

  * Within the same benchmark
  * Across model versions

Cross-benchmark comparisons require awareness of differing relevant densities.

---

### 5️⃣ Standard Reporting Recommendation

For each benchmark, report:

* MRR
* Success@25
* Precision@25
* NDCG@25
* Recall@25
* Diversity@25

Use MRR and Precision@25 as primary model selection signals.
