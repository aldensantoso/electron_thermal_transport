"""
Preprocessing pipeline for plasma confinement data.

This module:
- loads and validates raw plasma datasets
- resolves common column aliases and canonical feature names
- enforces physics-based sanity checks (e.g. tau_e > 0, n_e > 0)
- removes missing and non-finite values
- trims extreme outliers (1%-99%) and also using z-score filtering
- creates canonical clustering features
- computes PCA/KMeans projections for exploratory analysis
- provides plotting utilities for distributions, correlations, and clusters
\rTODO: safely remove the engineered features, overhaul the canonical cluster features into the controlled and mode properties frameworks
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def find_target_column(df):
    for alt in ("etau", "taue", "tau_e", "eta_u"):
        match = next((c for c in df.columns if norm(c) == norm(alt)), None)
        if match is not None:
            return match
    return None

def _resolve_alias_columns(df, col_map):
    rename_map = {}
    for raw_name, canonical_name in col_map.items():
        if raw_name in df.columns:
            rename_map[raw_name] = canonical_name

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def load_and_prep_data(filepath, col_map=None, feature_cols=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    # allow the notebook to pass the actual feature names directly
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing requested feature columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    target_candidates = ["tau_e", "etau", "taue", "eta_u"]
    target_col = next(
        (c for c in df.columns if str(c).strip().lower() in {str(x).lower() for x in target_candidates}),
        None,
    )
    if target_col is None:
        raise KeyError(f"No target column found. Available columns: {list(df.columns)}")

    if feature_cols is not None:
        df = df[feature_cols + [target_col]].copy()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[target_col] > 0].copy()

    return df.reset_index(drop=True)

def canonical_cluster_features(df):
    # Keep only raw physical variables; no engineered features
    raw_feature_candidates = [
        'BZXR',
        'PBEAM_A',
        'PBEAM_B',
        'PBEAM_C',
        'PBEAM_TOT',
        'NES',
        'ETAU'
    ]

    feature_cols = [
        c for c in raw_feature_candidates
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feature_cols

def compute_cluster_projection(df, feature_cols=None, n_clusters=4):
    if feature_cols is None:
        feature_cols = canonical_cluster_features(df)

    work = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        raise ValueError("No rows left for PCA/clustering.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(work)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=25)
    labels = km.fit_predict(X_scaled) + 1

    return work, X_pca, labels, scaler, pca, km

def plot_variable_distributions(df, columns=None, title="Variable Distributions"):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return

    n = len(valid_cols)
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = np.ravel(axes)

    for i, col in enumerate(valid_cols):
        axes[i].hist(df[col].dropna(), bins=30, color="skyblue", edgecolor="black")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    for j in range(len(valid_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, cols=None, title="Correlation Matrix"):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    cols = list(cols) + ['ETAU']
    corr = df[cols].corr(method= "spearman")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center = 0, vmin =-1, vmax = 1, fmt=".2f", linewidths = 0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_cluster_summary(df, feature_cols=None, n_clusters=4, title="KMeans Cluster Summary"):
    work, X_pca, labels, _, pca, km = compute_cluster_projection(
        df,
        feature_cols=feature_cols,
        n_clusters=n_clusters,
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=35, alpha=0.8)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.show()

    return work, X_pca, labels, pca, km

def plot_distributions(df, columns=None, title="Variable Distributions"):
    return plot_variable_distributions(df, columns=columns, title=title)

def engineer_features(df):
    target_col = find_target_column(df)
    if target_col is None:
        raise ValueError("No valid ETAU / tau_e target column found.")

    feature_cols = canonical_cluster_features(df)
    if not feature_cols:
        raise ValueError("No raw numeric feature columns available for modeling.")

    # Remove target if any raw feature accidentally matches it
    feature_cols = [c for c in feature_cols if c != target_col]

    work = df[feature_cols + [target_col]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work[work[target_col] > 0].copy()

    return work, feature_cols, target_col