# AI Benchmark Data Analysis: Open LLM Leaderboard

I treated the Hugging Face **Open LLM Leaderboard** (dataset `open-llm-leaderboard/contents`, 4,576 model entries, 6 benchmarks) as a benchmark report under review: validate the data, find trends, correlations and anomalies, and explain them to a non-technical reader.

**Stack:** Python, Pandas, NumPy, SciPy, Matplotlib, Seaborn
**Notebook:** `ai_benchmark_analysis.ipynb`

## Executive summary

The leaderboard's scoring is internally consistent: every reported average matched my own recalculation, and no score was missing or outside 0-100. The problems are in the metadata: 79 duplicate model names, 10 entries with a missing or zero size, and two entries with an implausibly tiny size (a few million parameters).

BBH and MMLU-PRO move together almost perfectly (Spearman rho 0.95), while IFEval and MUSR are the least related (rho 0.37). A single average score therefore hides very different skill profiles.

Bigger models score higher (rho 0.63), but the gain is not constant. The median rises from about 8 points under 3B parameters to about 37 at 13-35B, then stops improving (about 36 at 35-80B). The over-80B group has only 16 models, so it is too small to conclude anything.

Most statistical outliers (250 models) come from MATH Lvl 5 and GPQA, whose scores have a long upper tail, so they reflect specialisation rather than errors. A further 64 models have very uneven benchmark profiles, and at least 11 of the 15 most extreme are variants of the same base model (Phi-4: strong GPQA, weak IFEval). These are family effects, not 64 independent surprises. None of this proves contamination or error; model cards and training data must be checked before concluding anything.

## Method

1. **Data integrity checks:** dtype, missing values, duplicates, 0-100 range, recomputed vs reported average, invalid and implausible model sizes.
2. **EDA and statistics:** score distributions, Spearman correlation between benchmarks, size vs score (log scale), gain per size bucket, comparison by model type.
3. **Anomaly detection:** IQR outliers per benchmark, residuals from the size-vs-score trend, cross-benchmark imbalance (z-score spread), clustering of anomalies by base model, and a cross-check with the leaderboard's `Flagged` column.

## Key results

| Check | Result |
|---|---|
| Raw / clean records | 4,576 / 4,497 |
| Reported average vs recomputed | 0 differences |
| Duplicate model names | 79 |
| Missing or non-positive size | 10 |
| Most correlated benchmarks | BBH and MMLU-PRO (rho 0.95) |
| Least correlated benchmarks | IFEval and MUSR (rho 0.37) |
| Size vs average score | rho 0.63 |
| IQR outliers | 250 models (MATH Lvl 5: 181, GPQA: 80) |
| Imbalanced-profile models | 64 |

## Limitations

- Duplicates were detected by model name; I did not check whether they differ in precision, so some may be legitimate re-evaluations.
- Median score per size bucket is influenced by the mix of model types in each bucket.
- IQR assumes a roughly symmetric distribution; MATH Lvl 5 and GPQA are skewed, so outlier counts there are inflated.
- Possible reasons for anomalies (test-set contamination, genuine training improvements, merged models inheriting strengths, reporting errors) cannot be separated using this data alone.

## Charts

Saved in `images/`: correlation heatmap, size vs score, gain per size bucket, scores by model type, anomaly scatter.

## Reproduce

```bash
pip install datasets pandas numpy matplotlib seaborn scipy jupyter
jupyter notebook ai_benchmark_analysis.ipynb
```
If the dataset is unavailable, load a CSV copy (e.g. from Kaggle) in the fallback line of the loading cell.
