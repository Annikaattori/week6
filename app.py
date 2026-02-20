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
    dataset_path = st.text_input("CSV path", value="work_from_home_burnout_dataset.csv")
    include_day_type = st.checkbox("Include day_type as one-hot in modeling", value=False)

raw, clean, fa_df, cluster_df = ensure_data(dataset_path, include_day_type)

if page == "Data & EDA":
    st.header("Data & EDA")
    st.subheader("Dataset preview")
    st.dataframe(raw.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Summary statistics")
        st.dataframe(raw.describe(include="all").transpose(), use_container_width=True)
    with c2:
        st.subheader("Missing values")
        st.dataframe(raw.isna().sum().rename("missing_count"))

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
        "**Problem definition:** Tavoitteena on tiivistää etätyön kuormitus-, palautumis- ja suoriutumismittarit "
        "latentteihin faktoreihin, jotka helpottavat tulkintaa ja visualisointia."
    )

    max_factors = min(6, fa_df.shape[1])
    n_factors = st.slider("Number of factors", min_value=1, max_value=max_factors, value=min(3, max_factors))

    eigenvalues, scores, loadings, communalities = fit_factor_analysis(fa_df, n_factors)
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
    st.dataframe(loadings, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(loadings, cmap="coolwarm", center=0, annot=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Communalities")
    st.dataframe(communalities)

    scores_df = pd.DataFrame(scores, columns=[f"Factor{i+1}" for i in range(scores.shape[1])])
    st.subheader("Factor scores")
    st.dataframe(scores_df.head(20), use_container_width=True)

    if scores.shape[1] >= 2:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
        ax3.set_xlabel("Factor1")
        ax3.set_ylabel("Factor2")
        ax3.set_title("Factor score scatter")
        st.pyplot(fig3)

    st.markdown(
        "**Interpretation tip:** Nimeä faktorisi suurimpien latausten perusteella (esim. Workload, Recovery, Performance/Burnout)."
    )

elif page == "Clustering":
    st.header("Clustering")
    st.write(
        "**Problem definition:** Tavoitteena on ryhmitellä työntekijät luonnollisiin segmentteihin kuormituksen, "
        "palautumisen ja suoriutumisen perusteella."
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
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

    label_map = {
        f"kmeans_k={best_k}": KMeans(n_clusters=best_k, n_init=20, random_state=42).fit_predict(X),
        f"hierarchical_k={best_k}": AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X),
        "selected_model": labels,
    }
    metrics = metric_table(X, label_map)
    st.subheader("Evaluation metrics")
    st.dataframe(metrics, use_container_width=True)

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

    st.markdown("Lyhyt tulkinta: jos klusterit erottuvat selvästi kuvassa, segmentit ovat todennäköisesti mielekkäitä.")

else:
    st.header("Documentation / Learning")
    st.markdown(
        """
### Mikä on ohjaamaton oppiminen?
Ohjaamattomassa oppimisessa dataa analysoidaan ilman valmiita luokkatunnisteita. Tavoitteena on löytää rakenteita, ryhmiä ja latentteja tekijöitä.

### Mitä faktorit ja klusterit ovat?
- **Faktorit (FA)**: latentteja ulottuvuuksia, jotka tiivistävät useita korreloivia muuttujia.
- **Klusterit**: havaintojen ryhmiä, joissa saman ryhmän havainnot muistuttavat toisiaan enemmän kuin muita ryhmiä.

### Miksi standardointi on tärkeää?
Ilman standardointia suuren mittakaavan muuttujat dominoivat etäisyyksiä ja varianssia. Standardointi tekee muuttujista vertailukelpoisia.

### Tärkeimmät löydökset (täydennä analyysin jälkeen)
1. Kaiser + scree antavat perustellun faktorimäärän.
2. Klusterien laatu voidaan arvioida silhouette-, CH- ja DB-indekseillä.
3. FA-avaruus auttaa tulkitsemaan, erottuvatko klusterit kuormitus- tai palautumissuunnissa.
        """
    )
