from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Week 6 - FA & Clustering", layout="wide")

FA_RECOMMENDED = [
    "work_hours",
    "screen_time_hours",
    "meetings_count",
    "breaks_taken",
    "sleep_hours",
    "task_completion_rate",
    "burnout_score",
]

CLUSTER_RECOMMENDED = [
    "work_hours",
    "screen_time_hours",
    "meetings_count",
    "breaks_taken",
    "after_hours_work",
    "sleep_hours",
    "task_completion_rate",
    "burnout_score",
]


def default_csv_path() -> str:
    repo_root = Path(__file__).resolve().parent
    csv_files = sorted(repo_root.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csv_files[0].name if csv_files else "work_from_home_burnout_dataset.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


@st.cache_data
def preprocess(df: pd.DataFrame, include_day_type: bool) -> pd.DataFrame:
    if df.empty:
        return df
    data = df.copy()
    data = data.drop(columns=["user_id"], errors="ignore")

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in data.columns if c not in num_cols]

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        data[num_cols] = num_imputer.fit_transform(data[num_cols])

    for c in cat_cols:
        mode_val = data[c].mode().iloc[0] if not data[c].mode().empty else "Unknown"
        data[c] = data[c].fillna(mode_val)

    if "day_type" in data.columns:
        if include_day_type:
            data = pd.get_dummies(data, columns=["day_type"], drop_first=True)
        else:
            data = data.drop(columns=["day_type"])
    return data


@st.cache_resource
def fit_factor_analysis(fa_df: pd.DataFrame, n_factors: int):
    scaler = StandardScaler()
    X = scaler.fit_transform(fa_df)

    pca = PCA()
    pca.fit(X)
    eigenvalues = pca.explained_variance_

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    scores = fa.fit_transform(X)
    loadings = pd.DataFrame(
        fa.components_.T,
        index=fa_df.columns,
        columns=[f"Factor{i+1}" for i in range(n_factors)],
    )
    communalities = (loadings**2).sum(axis=1).to_frame("communality")
    return eigenvalues, scores, loadings, communalities


@st.cache_resource
def compute_kmeans_sweeps(cluster_df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df)

    max_k = min(9, len(cluster_df) - 1)
    ks = list(range(2, max_k + 1))
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return X, ks, inertias, silhouettes


def metric_table(X: np.ndarray, labels_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, labels in labels_map.items():
        uniq = np.unique(labels)
        if len(uniq) < 2 or (len(uniq) == 1 and uniq[0] == -1):
            rows.append({"model": name, "silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan})
            continue
        rows.append(
            {
                "model": name,
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }
        )
    return pd.DataFrame(rows)


def ensure_data(path: str, include_day_type: bool):
    raw = load_data(path)
    if raw.empty:
        st.error(f"Dataset not found at `{path}`. Add work_from_home_burnout_dataset.csv to repo root or provide path.")
        st.stop()
    clean = preprocess(raw, include_day_type)
    fa_cols = [c for c in FA_RECOMMENDED if c in clean.columns]
    cl_cols = [c for c in CLUSTER_RECOMMENDED if c in clean.columns]
    if not fa_cols or not cl_cols:
        st.error("Required modeling columns were not found in dataset.")
        st.stop()
    return raw, clean, clean[fa_cols], clean[cl_cols]


st.title("Week 6: Unsupervised Learning (Factor Analysis + Clustering)")

with st.sidebar:
    page = st.radio("Navigation", ["Data & EDA", "Factor Analysis", "Clustering", "Documentation / Learning"])
    dataset_path = st.text_input("CSV path", value=default_csv_path())
    include_day_type = st.checkbox("Include day_type as one-hot in modeling", value=False)

raw, clean, fa_df, cluster_df = ensure_data(dataset_path, include_day_type)

if page == "Data & EDA":
    st.header("Data & EDA")
    st.subheader("Dataset preview")
    st.dataframe(raw.head(10), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Summary statistics")
        st.dataframe(raw.describe(include="all").transpose(), width="stretch")
    with c2:
        st.subheader("Missing values")
        st.dataframe(raw.isna().sum().rename("missing_count"), width="stretch")

    st.subheader("Correlation matrix")
    num = clean.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution example")
    col = st.selectbox("Histogram variable", num.columns.tolist(), index=0)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(num[col], bins=20, color="#4c72b0", alpha=0.85)
    ax2.set_title(f"Histogram: {col}")
    st.pyplot(fig2)

elif page == "Factor Analysis":
    st.header("Factor Analysis")
    st.write(
        "**Problem definition:** The goal is to summarize remote-work workload, recovery, and performance measures "
        "into latent factors that aid interpretation and visualization."
    )

    max_factors = min(6, fa_df.shape[1])
    n_factors = st.slider("Number of factors", min_value=1, max_value=max_factors, value=min(3, max_factors))

    eigenvalues, scores, loadings, communalities = fit_factor_analysis(fa_df, n_factors)
    avg_communality = communalities["communality"].mean() if not communalities.empty else np.nan
    st.metric("Average communality (proxy for variance explained)", f"{avg_communality:.3f}")
    st.write("Note: sklearn's `FactorAnalysis` does not provide per-factor variance-explained like PCA.\n"
             "Average communality (mean of variable communalities) is shown as a proxy for the fraction of each variable's variance explained by the common factors.\n"
             "For a more explicit variance decomposition you can use the `factor_analyzer` package which reports explained variance per factor.")
    kaiser_k = int(max(1, np.sum(eigenvalues > 1)))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Kaiser suggestion (eigenvalue > 1)", kaiser_k)
    with c2:
        st.write(f"Selected factors: **{n_factors}**")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker="o")
    ax.axhline(1.0, color="red", linestyle="--")
    ax.set_title("Scree plot")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    st.pyplot(fig)

    st.subheader("Factor loadings")
    st.dataframe(loadings, width="stretch")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(loadings, cmap="coolwarm", center=0, annot=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Communalities")
    st.dataframe(communalities, width="stretch")

    scores_df = pd.DataFrame(scores, columns=[f"Factor{i+1}" for i in range(scores.shape[1])])
    st.subheader("Factor scores")
    st.dataframe(scores_df.head(20), width="stretch")

    if scores.shape[1] >= 2:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
        ax3.set_xlabel("Factor1")
        ax3.set_ylabel("Factor2")
        ax3.set_title("Factor score scatter")
        st.pyplot(fig3)

    st.markdown(
        "**Interpretation tip:** Name your factors according to the largest loadings (e.g., Workload, Recovery, Performance/Burnout)."
    )

elif page == "Clustering":
    st.header("Clustering")
    st.write(
        "**Problem definition:** The goal is to group employees into natural segments based on workload, "
        "recovery, and performance."
    )

    X, ks, inertias, silhouettes = compute_kmeans_sweeps(cluster_df)

    c1, c2 = st.columns(2)
    with c1:
        fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4))
        ax_elbow.plot(ks, inertias, marker="o")
        ax_elbow.set_title("Elbow (k-means)")
        ax_elbow.set_xlabel("k")
        ax_elbow.set_ylabel("Inertia")
        st.pyplot(fig_elbow)
    with c2:
        fig_s, ax_s = plt.subplots(figsize=(7, 4))
        ax_s.plot(ks, silhouettes, marker="o", color="green")
        ax_s.set_title("Silhouette vs k")
        ax_s.set_xlabel("k")
        ax_s.set_ylabel("Silhouette")
        st.pyplot(fig_s)

    algo = st.selectbox("Algorithm", ["k-means", "hierarchical", "DBSCAN"])
    best_k = ks[int(np.argmax(silhouettes))]

    if algo == "k-means":
        k = st.slider("k", min_value=2, max_value=max(2, min(9, len(cluster_df) - 1)), value=int(best_k))
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(X)
    elif algo == "hierarchical":
        k = st.slider("n_clusters", min_value=2, max_value=max(2, min(9, len(cluster_df) - 1)), value=int(best_k))
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)

        st.subheader("Dendrogram")
        fig_d, ax_d = plt.subplots(figsize=(10, 4))
        dendrogram(linkage(X, method="ward"), no_labels=True, ax=ax_d)
        st.pyplot(fig_d)
    else:
        eps = st.slider("eps", min_value=0.1, max_value=3.0, value=1.2, step=0.1)
        min_samples = st.slider("min_samples", min_value=3, max_value=20, value=6)
        auto_tune = st.checkbox("Auto-tune DBSCAN (grid search)", value=False)
        if auto_tune:
            eps_candidates = np.round(np.linspace(0.3, 3.0, 14), 2)
            min_samples_candidates = [3, 4, 5, 6, 8]
            best = {"silhouette": -np.inf, "eps": None, "min_samples": None, "labels": None}
            for e in eps_candidates:
                for m in min_samples_candidates:
                    model_try = DBSCAN(eps=float(e), min_samples=int(m))
                    labels_try = model_try.fit_predict(X)
                    uniq = np.unique(labels_try)
                    # require at least 2 clusters (noise allowed)
                    if len(uniq) < 2 or (len(uniq) == 2 and -1 in uniq and np.sum(labels_try != -1) < 2):
                        continue
                    try:
                        s = silhouette_score(X, labels_try)
                    except Exception:
                        continue
                    if s > best["silhouette"]:
                        best.update({"silhouette": s, "eps": e, "min_samples": m, "labels": labels_try})
            if best["eps"] is not None:
                st.write(f"Best DBSCAN (by silhouette): eps={best['eps']}, min_samples={best['min_samples']}, silhouette={best['silhouette']:.3f}")
                labels = best["labels"]
            else:
                st.write("Auto-tune found no suitable DBSCAN clustering (too few clusters or all noise). Try expanding parameter ranges.")
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X)
        else:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)

    label_map = {
        f"kmeans_k={best_k}": KMeans(n_clusters=best_k, n_init=20, random_state=42).fit_predict(X),
        f"hierarchical_k={best_k}": AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X),
        "selected_model": labels,
    }
    metrics = metric_table(X, label_map)
    st.subheader("Evaluation metrics")
    st.dataframe(metrics, width="stretch")

    centered = pd.DataFrame(X, columns=cluster_df.columns)
    centered["cluster"] = labels
    profile = centered.groupby("cluster").mean(numeric_only=True)
    st.subheader("Cluster profile heatmap (standardized)")
    fig_h, ax_h = plt.subplots(figsize=(9, 4))
    sns.heatmap(profile, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax_h)
    st.pyplot(fig_h)

    st.subheader("Cluster scatter in FA space")
    evals, fa_scores, _, _ = fit_factor_analysis(fa_df, n_factors=min(2, fa_df.shape[1]))
    if fa_scores.shape[1] >= 2:
        fig_sc, ax_sc = plt.subplots(figsize=(7, 5))
        ax_sc.scatter(fa_scores[:, 0], fa_scores[:, 1], c=labels, cmap="tab10", alpha=0.8)
        ax_sc.set_xlabel("Factor1")
        ax_sc.set_ylabel("Factor2")
        ax_sc.set_title("Clusters projected in FA space")
        st.pyplot(fig_sc)

    st.markdown("Brief interpretation: if clusters separate clearly in the plot, the segments are likely meaningful.")

else:
    st.header("Documentation / Learning")
    # Top summary and visual metric cards
    st.subheader("Analysis summary")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("### Silhouette")
        st.write("Range: -1 … 1")
        st.write("Higher → better separation between clusters")
    with m2:
        st.markdown("### Calinski–Harabasz (CH)")
        st.write("Higher → more compact and well-separated clusters")
    with m3:
        st.markdown("### Davies–Bouldin (DB)")
        st.write("Lower → better (less intra-cluster scatter relative to inter-cluster distance)")

    st.markdown("---")
    st.subheader("Practical checks")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- Visual separability: FA scatter, dendrogram, heatmaps")
        st.markdown("- Domain sense: do profiles match expected patterns (workload → burnout)")
    with col2:
        st.markdown("- Robustness: are patterns similar across methods (k-means, hierarchical, DBSCAN)")
        st.markdown("- Actionability: can clusters inform interventions or further analysis?")

    st.markdown("---")
    st.subheader("Quick checklist — files to inspect")
    st.markdown(
        "- FA: `outputs/fa_scree_plot.png`, `outputs/fa_loadings.csv`, `outputs/fa_communalities.csv`\n"
        "- Clustering: `outputs/clustering_elbow.png`, `outputs/clustering_silhouette_k.png`, `outputs/clustering_metrics.csv`, `outputs/cluster_profiles_standardized.csv`, `outputs/cluster_profile_heatmap.png`\n"
        "- Report: `outputs/report.md`"
    )

    with st.expander("Detailed analysis and results (expand)"):
        st.markdown(
            """
The same dataset was used in my previous week’s supervised learning project, so I already had some familiarity with the variables and their relationships. This helped in interpreting the FA factors and cluster profiles.
 
1) Factor Analysis: what latent dimensions did I find?

Variables used in FA:
work_hours, screen_time_hours, meetings_count, breaks_taken, sleep_hours, task_completion_rate, burnout_score
(Note: breaks_taken and sleep_hours are weakly explained by the factor model in this dataset.)

Factor 1 – Workload & burnout vs. performance

Strong positive loadings: work_hours (~0.73), screen_time_hours (~0.73), burnout_score (~0.74)

Strong negative loading: task_completion_rate (~-0.74)

Interpretation:
This factor captures a core “overall strain/workload” dimension where longer working time and more screen time are associated with higher burnout and lower task completion.
Higher Factor 1 scores imply higher workload + higher burnout + weaker output.

Factor 2 – Disengagement/inefficiency (burnout not purely driven by high workload)

Positive loading: burnout_score (~+0.65)

Negative loadings: work_hours (~-0.65), screen_time_hours (~-0.65), task_completion_rate (~-0.64)

meetings_count is mildly negative (~-0.32)

Interpretation:
This factor separates cases where burnout and reduced performance can appear even when measured workload (hours/screen time) is not high.
Conceptually, it resembles a “disengagement/inefficiency” or “burnout without obvious workload” axis.

Note: factor signs are arbitrary (a factor can be multiplied by -1). Interpretation relies on which variables move together vs. in opposite directions.

Factor 3 – Meeting intensity / daily rhythm (weaker but meeting-related)

Most visible loading: meetings_count (~0.25) and it correlates most strongly with Factor 3 scores (~0.56)

breaks_taken and sleep_hours load only weakly

Interpretation:
Factor 3 primarily reflects meeting intensity (and only very weakly “daily rhythm” aspects). It is noticeably weaker than Factors 1 and 2.

Communalities: which variables are well explained by FA?

Very well explained (communality ~0.96):
work_hours, screen_time_hours, task_completion_rate, burnout_score

Poorly explained:
breaks_taken (~0.01), sleep_hours (~0.007)

meetings_count is moderate (~0.33)

Conclusion:
FA identifies a strong latent structure mainly on the axis (work/screen/burnout/completion). In contrast, sleep and breaks do not align strongly with the same latent structure in this dataset (or the measurement noise dominates), so they don’t form a strong factor here.

2) Clustering: what groups did I find and how good are they?

Selected number of clusters: k = 2 (best_k = 2)

Model comparison (2–3 metrics)

From my metrics:

- k-means (k=2): silhouette ≈ 0.22, CH ≈ 552, DB ≈ 1.75

- hierarchical (k=2): silhouette ≈ 0.16, CH ≈ 351, DB ≈ 2.12

Interpretation:
k-means is clearly better across these metrics. A silhouette around 0.22 is low-to-moderate, meaning the clusters exist but are not perfectly separated (common in behavioral/people data).

3) Cluster profiles: what do the clusters represent in practice?

k-means produced two segments (N=1800):

Cluster 0 (n=832, ~46%) — “Higher workload + higher burnout + lower performance”
- work_hours: 8.65
- screen_time_hours: 11.45
- meetings_count: 3.06
- task_completion_rate: 69.54
- burnout_score: 48.34

Cluster 1 (n=968, ~54%) — “Lower workload + lower burnout + better performance”
- work_hours: 4.68
- screen_time_hours: 7.40
- meetings_count: 0.98
- task_completion_rate: 74.69
- burnout_score: 40.29

Which variables separate clusters the most?

From standardized profiles, the biggest differences are:
- work_hours (≈ +0.93 vs -0.80)
- screen_time_hours (≈ +0.90 vs -0.78)
- meetings_count (≈ +0.66 vs -0.57)

Conclusion: clustering is driven primarily by the workload/screen/meetings bundle; burnout and completion differences follow that structure.

4) Joint interpretation: how do factors and clusters connect?

Overall pattern:

Cluster 0 corresponds to high Factor 1 (“workload & burnout vs completion”): heavy workload + higher burnout + lower completion.

Cluster 1 corresponds to low Factor 1: lighter workload + lower burnout + higher completion.

So: clusters are mainly separated along Factor 1, suggesting the dominant latent dimension in this dataset is the workload–burnout–performance axis.

5) Practical meaning (plain-language takeaway)

The dataset shows a strong and consistent structure: higher workload (hours + screen time + meetings) is associated with higher burnout and reduced task completion.

FA also suggests a secondary phenomenon: burnout/low performance is not always purely a direct function of measured workload (Factor 2), but clustering mostly captures the main “mass effect” (Factor 1).
            """
        )
