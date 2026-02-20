# Week 6 - Unsupervised Learning (FA + Clustering)

This repository contains:

- `analysis_week6.py`: batch analysis script that performs preprocessing, factor analysis, clustering, evaluation, and saves plots/tables.
- `app.py`: Streamlit app with pages:
  - Data & EDA
  - Factor Analysis
  - Clustering
  - Documentation / Learning

## Dataset

Place your CSV file at repository root. The app and script now default to the newest `.csv` file in the repo root.

Default fallback filename is:

- `work_from_home_burnout_dataset.csv`

Expected columns include:

- ID/labels: `user_id`, `day_type`, `burnout_risk`
- Numeric features: `work_hours`, `screen_time_hours`, `meetings_count`, `breaks_taken`, `after_hours_work`, `sleep_hours`, `task_completion_rate`, `burnout_score`

## Install

```bash
pip install -r requirements.txt
```

## Run analysis script

```bash
python analysis_week6.py --output outputs
```

Outputs are saved under `outputs/`, including:

- FA scree, loadings heatmap, factor score scatter
- k-means elbow/silhouette plots
- hierarchical dendrogram
- cluster profile heatmap
- CSV tables (loadings, communalities, metrics, profiles, labels)
- `report.md` with interpretation and reflection

## Run Streamlit app

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

By default, the sidebar CSV path is prefilled with the newest `.csv` file found in the repo root.
```

App supports:

- preprocessing with imputation and optional `day_type` one-hot
- FA with selectable number of factors and Kaiser/Scree support
- clustering with k-means, hierarchical, or DBSCAN
- evaluation metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)
- cluster visualization in FA space

## VPS path

Deploy behind reverse proxy so app is available under `/week6` path.
