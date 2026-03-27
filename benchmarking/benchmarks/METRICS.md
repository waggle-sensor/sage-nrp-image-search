# Evaluation Metrics for Sage Image Search

This document outlines the evaluation metrics for the Sage Image Search system.

## 1️⃣ Keep K Fixed

* Always evaluate at **K = response_limit**, since the system returns the top `response_limit` results.
>NOTE: at this moment, the response limit is 25. This may change in the future. - 03/01/2026
* Do **not** change K per query based on how many relevant items exist.
* Fixed K ensures:
  * Fair comparison across queries
  * Fair comparison across model versions
  * Reproducibility

---

## 2️⃣ Problem Context

* Labels are **binary (relevant / non-relevant)**.
* The **average number of relevant items per query varies across benchmarks** (fire, urban, atmospheric, biology).
* Relevant items are often sparse compared to total corpus size.
* Early ranking quality matters more than full-corpus coverage.

---

## 3️⃣ Recommended Metric Strategy

Use a combination of ranking-sensitive and top-K metrics:

## Primary Decision Metrics

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

## Supporting Metrics

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

## 4️⃣ Interpretation Guidance

* Precision reflects first-page quality.
* MRR reflects early ranking strength.
* Success@25 reflects practical usefulness.
* NDCG captures ranking structure.
* Diversity@25 encourages exploration of different relevant items.
* Metric comparisons should primarily be made:

  * Within the same benchmark
  * Across model versions

Cross-benchmark comparisons require awareness of differing relevant densities.

### Metric Differences to Detect

As a reference, here is a rough estimation of the number of evaluation samples needed to be 95% confident that one system is better. Values from OpenAI. A useful rule is that for every 3x decrease in score difference, the number of samples needed increases by 10x (This is because the square root of 10 is 3.162).

| Difference to detect | Sample size needed for 95% confidence |
| -------------------- | -------------------------------------- |
| 30%                 | ~10                                |
| 10%                 | ~100                                |
| 3%                 | ~1,000                                 |
| 1%                 | ~10,000                               |

Among evaluation benchmarks in Eleuther's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), the median number of examples is 1,000, and the average is 2,159. The organizers of the [Inverse Scaling prize](https://irmckenzie.co.uk/round1) suggested that 300 examples is the absolute mininum and they would prefer at least 1,000, especially if the examples are being synthesized - [Inverse Scaling: When Bigger Isn't Better](https://arxiv.org/abs/2306.09479).
>NOTE: These notes were retrieved from the book[AI Engineering by Chip Huyen](https://github.com/chiphuyen/aie-book) which mentions that the values from the table were retrieved from OpenAI. I was unable to find the original source for these values.

---

## 5️⃣ Standard Reporting Recommendation

For each benchmark, report:

* MRR
* Success@25
* Precision@25
* NDCG@25
* Recall@25
* Diversity@25

Use MRR and Success@25 as primary model selection signals.

---

## 6️⃣ Leaderboard Ranking Configuration

For cross-version leaderboards (`v10+`), use weighted composites.

### Primary Leaderboard (default)

Default equal metric weights:

* MRR: `0.50`
* Success@25: `0.50`

Where:

* `Success@25 = HitRate@25 = mean(hit)` across queries.

### Primary + Diversity Leaderboard

Default equal metric weights:

* MRR: `1/3`
* Success@25: `1/3`
* Diversity@25: `1/3`

### Benchmark Weights for Overall Leaderboard

The overall cross-benchmark leaderboard aggregates benchmark scores per system version using benchmark-level weights.

Default benchmark weighting is equal:

* `INQUIRE: 1`
* `Firebench: 1`
* `Commonobjectsbench: 1`
* `Cloudbench: 1`

You can override either metric weights or benchmark weights, but all weights are normalized to sum to `1.0` before scoring.
