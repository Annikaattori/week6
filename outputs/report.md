# Week 6 Unsupervised Learning Report

## A1) Problem definition (FA)
Goal is to compress remote-work strain, recovery, and performance indicators into 2-4 latent factors for clearer interpretation and visualization.
 
## A2) Feature choice for FA
Used numeric features: work_hours, screen_time_hours, meetings_count, breaks_taken, sleep_hours, task_completion_rate, burnout_score.
Binary `after_hours_work` was excluded from FA to avoid mixing binary structure with continuous latent-factor assumptions.

## A3) Number of factors
Kaiser criterion (eigenvalue > 1) selected **k = 3**. Scree plot saved to `outputs/fa_scree_plot.png`.

## A4) FA fit + interpretation
Estimated total common variance proxy (average communality): **0.598**.
Note: this value is the mean of variable communalities (sum of squared loadings per variable). sklearn's `FactorAnalysis` does not directly report per-factor "variance explained" like `PCA` does; the average communality is therefore used as a proxy for the fraction of each variable's variance explained by the common factors. If an explicit per-factor explained-variance decomposition is required, consider using the `factor_analyzer` package which provides that output.
Inferred factor labels based on strongest loadings:
- 
- Factor1: task_completion_rate + burnout_score
- Factor2: work_hours + screen_time_hours
- Factor3: meetings_count + breaks_taken

## B1) Problem definition (clustering)
Goal is to discover natural employee segments from workload, recovery, and performance indicators to identify risk-oriented work patterns.

## B3-B5) Methods and evaluation
Methods: k-means and hierarchical (Ward). Metrics:

| model            |   silhouette |   calinski_harabasz |   davies_bouldin |
|:-----------------|-------------:|--------------------:|-----------------:|
| kmeans_k=2       |     0.220082 |             552.154 |          1.74585 |
| hierarchical_k=2 |     0.161521 |             351.042 |          2.12491 |

## B6) Cluster interpretation
Cluster profiles are available in:
- `outputs/cluster_profiles_original_scale.csv`
- `outputs/cluster_profiles_standardized.csv`
- `outputs/cluster_profile_heatmap.png`

## FA + Clusters combined
See `outputs/clusters_in_fa_space.png` for cluster separation in Factor1/Factor2 space.

## DBSCAN note
DBSCAN is included in the app as a supplementary (bonus) algorithm. It is useful for discovering non-globular clusters and noise points, but its output is sensitive to `eps` and `min_samples`. In the app you can either set these sliders manually or enable "Auto-tune DBSCAN (grid search)", which performs a small parameter sweep and selects parameters that maximize silhouette score (when a valid multi-cluster partition is found). For the main quantitative comparison in this analysis we focus on k-means vs hierarchical clustering; DBSCAN is provided as an exploratory alternative.

## Reflection
Clusters are meaningful when silhouette is moderate and profile differences are coherent; if silhouette is low, overlap is likely.
FA appears intuitive when factor loadings group into workload/recovery/performance dimensions.
Comparing k-means vs hierarchical checks method robustness.
Limitations include correlated features, binary variables in latent models, and potential engineered structure in burnout-related fields.

- Expected: higher work/screen load associates with higher burnout-oriented factor loadings.
- Surprising: cluster boundaries may remain moderate (silhouette not very high), indicating overlap between work profiles.
