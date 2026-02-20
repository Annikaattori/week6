from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


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
    return str(csv_files[0].name) if csv_files else "work_from_home_burnout_dataset.csv"


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. Provide --csv pointing to work_from_home_burnout_dataset.csv"
        )
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame, include_day_type: bool = False) -> pd.DataFrame:
    data = df.copy()

    drop_candidates = ["user_id"]
    for col in drop_candidates:
        if col in data.columns:
            data = data.drop(columns=[col])

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in data.columns if c not in num_cols]

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        data[num_cols] = num_imputer.fit_transform(data[num_cols])

    for c in cat_cols:
        data[c] = data[c].fillna(data[c].mode().iloc[0] if not data[c].mode().empty else "Unknown")

    if not include_day_type and "day_type" in data.columns:
        data = data.drop(columns=["day_type"])
    elif include_day_type and "day_type" in data.columns:
        data = pd.get_dummies(data, columns=["day_type"], drop_first=True)

    return data


def prepare_feature_sets(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    burnout_risk = data["burnout_risk"] if "burnout_risk" in data.columns else None

    model_df = data.drop(columns=["burnout_risk"], errors="ignore")

    fa_cols = [c for c in FA_RECOMMENDED if c in model_df.columns]
    cl_cols = [c for c in CLUSTER_RECOMMENDED if c in model_df.columns]

    return model_df[fa_cols].copy(), model_df[cl_cols].copy(), burnout_risk


def scale_df(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(df), scaler


def choose_factors_by_kaiser(X_scaled: np.ndarray) -> Tuple[np.ndarray, int]:
    pca = PCA()
    pca.fit(X_scaled)
    eigenvalues = pca.explained_variance_
    k = int(max(1, np.sum(eigenvalues > 1.0)))
    return eigenvalues, k


def run_factor_analysis(fa_df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame | np.ndarray | int | float]:
    X_scaled, _ = scale_df(fa_df)
    eigenvalues, k = choose_factors_by_kaiser(X_scaled)

    fa_model = FactorAnalysis(n_components=k, random_state=42)
    scores = fa_model.fit_transform(X_scaled)
    loadings = pd.DataFrame(
        fa_model.components_.T,
        index=fa_df.columns,
        columns=[f"Factor{i+1}" for i in range(k)],
    )

    communalities = pd.Series((loadings**2).sum(axis=1), name="communality")
    variance_explained = float((loadings**2).sum().sum() / loadings.shape[0])

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    x = np.arange(1, len(eigenvalues) + 1)
    plt.plot(x, eigenvalues, marker="o")
    plt.axhline(1.0, color="red", linestyle="--", label="Kaiser = 1.0")
    plt.title("Scree Plot (PCA eigenvalues)")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fa_scree_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
    plt.title("Factor Loadings Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "fa_loadings_heatmap.png", dpi=150)
    plt.close()

    if scores.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
        plt.xlabel("Factor1 score")
        plt.ylabel("Factor2 score")
        plt.title("Factor Scores: Factor1 vs Factor2")
        plt.tight_layout()
        plt.savefig(output_dir / "fa_scores_scatter.png", dpi=150)
        plt.close()

    return {
        "k": k,
        "eigenvalues": eigenvalues,
        "loadings": loadings,
        "communalities": communalities.to_frame(),
        "scores": scores,
        "variance_explained_est": variance_explained,
    }


def compute_cluster_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique = np.unique(labels)
    if len(unique) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": silhouette_score(X_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels),
    }


def run_clustering(cluster_df: pd.DataFrame, fa_scores: np.ndarray, output_dir: Path) -> Dict[str, pd.DataFrame]:
    X_scaled, scaler = scale_df(cluster_df)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_k = min(9, len(cluster_df) - 1)
    ks = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = ks[int(np.argmax(silhouettes))] if ks else 2
    kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    km_labels = kmeans.fit_predict(X_scaled)

    hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hc_labels = hierarchical.fit_predict(X_scaled)

    metric_rows = []
    for name, labels in [(f"kmeans_k={best_k}", km_labels), (f"hierarchical_k={best_k}", hc_labels)]:
        m = compute_cluster_metrics(X_scaled, labels)
        metric_rows.append({"model": name, **m})
    metrics_df = pd.DataFrame(metric_rows)

    plt.figure(figsize=(7, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Plot (k-means)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(output_dir / "clustering_elbow.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ks, silhouettes, marker="o", color="green")
    plt.title("Silhouette vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(output_dir / "clustering_silhouette_k.png", dpi=150)
    plt.close()

    Z = linkage(X_scaled, method="ward")
    plt.figure(figsize=(10, 4))
    dendrogram(Z, no_labels=True, color_threshold=None)
    plt.title("Hierarchical Clustering Dendrogram (Ward)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "clustering_dendrogram.png", dpi=150)
    plt.close()

    if fa_scores.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(fa_scores[:, 0], fa_scores[:, 1], c=km_labels, cmap="tab10", alpha=0.8)
        plt.xlabel("Factor1")
        plt.ylabel("Factor2")
        plt.title("k-means clusters in FA space")
        plt.tight_layout()
        plt.savefig(output_dir / "clusters_in_fa_space.png", dpi=150)
        plt.close()

    cluster_df_std = pd.DataFrame(X_scaled, columns=cluster_df.columns)
    cluster_df_orig = cluster_df.copy()
    cluster_df_orig["cluster"] = km_labels
    cluster_df_std["cluster"] = km_labels

    profile_orig = cluster_df_orig.groupby("cluster").mean(numeric_only=True)
    profile_std = cluster_df_std.groupby("cluster").mean(numeric_only=True)

    plt.figure(figsize=(9, 5))
    sns.heatmap(profile_std, cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("Cluster Profiles (standardized means)")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_profile_heatmap.png", dpi=150)
    plt.close()

    return {
        "metrics": metrics_df,
        "profile_orig": profile_orig,
        "profile_std": profile_std,
        "labels": pd.DataFrame({"kmeans_cluster": km_labels, "hierarchical_cluster": hc_labels}),
        "best_k": pd.DataFrame({"best_k": [best_k]}),
    }


def infer_factor_names(loadings: pd.DataFrame) -> List[str]:
    names = []
    for col in loadings.columns:
        top = loadings[col].abs().sort_values(ascending=False).head(2).index.tolist()
        names.append(f"{col}: {' + '.join(top)}")
    return names


def write_report(output_dir: Path, fa_results: Dict, cl_results: Dict) -> None:
    factor_labels = infer_factor_names(fa_results["loadings"])
    metrics_df = cl_results["metrics"]

    expected = "Expected: higher work/screen load associates with higher burnout-oriented factor loadings."
    surprising = "Surprising: cluster boundaries may remain moderate (silhouette not very high), indicating overlap between work profiles."

    report = f"""# Week 6 Unsupervised Learning Report

## A1) Problem definition (FA)
Goal is to compress remote-work strain, recovery, and performance indicators into 2-4 latent factors for clearer interpretation and visualization.

## A2) Feature choice for FA
Used numeric features: {', '.join(fa_results['loadings'].index)}.
Binary `after_hours_work` was excluded from FA to avoid mixing binary structure with continuous latent-factor assumptions.

## A3) Number of factors
Kaiser criterion (eigenvalue > 1) selected **k = {fa_results['k']}**. Scree plot saved to `outputs/fa_scree_plot.png`.

## A4) FA fit + interpretation
Estimated total common variance proxy: **{fa_results['variance_explained_est']:.3f}**.
Inferred factor labels based on strongest loadings:
- """
    report += "\n- ".join([""] + factor_labels)
    report += f"""

## B1) Problem definition (clustering)
Goal is to discover natural employee segments from workload, recovery, and performance indicators to identify risk-oriented work patterns.

## B3-B5) Methods and evaluation
Methods: k-means and hierarchical (Ward). Metrics:

{metrics_df.to_markdown(index=False)}

## B6) Cluster interpretation
Cluster profiles are available in:
- `outputs/cluster_profiles_original_scale.csv`
- `outputs/cluster_profiles_standardized.csv`
- `outputs/cluster_profile_heatmap.png`

## FA + Clusters combined
See `outputs/clusters_in_fa_space.png` for cluster separation in Factor1/Factor2 space.

## Reflection
Clusters are meaningful when silhouette is moderate and profile differences are coherent; if silhouette is low, overlap is likely.
FA appears intuitive when factor loadings group into workload/recovery/performance dimensions.
Comparing k-means vs hierarchical checks method robustness.
Limitations include correlated features, binary variables in latent models, and potential engineered structure in burnout-related fields.

- {expected}
- {surprising}
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 6 FA + clustering analysis")
    parser.add_argument("--csv", default=default_csv_path(), help="Path to CSV dataset")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--include-day-type", action="store_true", help="Include day_type one-hot in modeling")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)

    raw = load_data(csv_path)
    clean = preprocess(raw, include_day_type=args.include_day_type)
    fa_df, cluster_df, burnout_risk = prepare_feature_sets(clean)

    if fa_df.empty or cluster_df.empty:
        raise ValueError("Feature set is empty. Check input columns.")

    fa_results = run_factor_analysis(fa_df, output_dir)
    cl_results = run_clustering(cluster_df, fa_results["scores"], output_dir)

    fa_results["loadings"].to_csv(output_dir / "fa_loadings.csv")
    fa_results["communalities"].to_csv(output_dir / "fa_communalities.csv")
    pd.DataFrame(fa_results["scores"], columns=[f"Factor{i+1}" for i in range(fa_results["scores"].shape[1])]).to_csv(
        output_dir / "fa_scores.csv", index=False
    )

    cl_results["metrics"].to_csv(output_dir / "clustering_metrics.csv", index=False)
    cl_results["profile_orig"].to_csv(output_dir / "cluster_profiles_original_scale.csv")
    cl_results["profile_std"].to_csv(output_dir / "cluster_profiles_standardized.csv")
    cl_results["labels"].to_csv(output_dir / "cluster_labels.csv", index=False)
    cl_results["best_k"].to_csv(output_dir / "best_k.csv", index=False)

    write_report(output_dir, fa_results, cl_results)

    print(f"Analysis completed. Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
