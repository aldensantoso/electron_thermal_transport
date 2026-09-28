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


def load_and_prep_data(filepath, col_map):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    df = _resolve_alias_columns(df, col_map)

    required = ["P_beam", "B_T", "I_P", "n_e", "f", "n", "dB", "tau_e"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing mapped columns after rename: {missing}")

    # coerce to numeric before filtering
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # physical cuts
    df = df[(df["tau_e"] > 0) & (df["n_e"] > 0)].copy()

    # drop missing values in the required fields
    df = df.dropna(subset=required).copy()

    # z-score filter on selected raw variables
    var_cols = [c for c in required if c in df.columns and df[c].nunique() > 1]
    if var_cols:
        z_scores = np.abs(stats.zscore(df[var_cols].astype(float)))
        df = df[(z_scores < 3).all(axis=1)].copy()

    return df.reset_index(drop=True)

    return df

def canonical_cluster_features(df):
    feature_cols = ["P_beam", "B_T", "I_P", "n_e", "f", "n", "dB"]
    return [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

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
    labels = km.fit_predict(X_scaled)

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

    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
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
    X_cols = [c for c in ["P_beam", "B_T", "I_P", "n_e", "f", "n", "dB"] if c in df.columns]
    numeric_cols = [
        c for c in df.columns
        if c not in X_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    Y_target_col = "tau_e" if "tau_e" in df.columns else None
    if Y_target_col is None:
        Y_target_col = next((c for c in numeric_cols if "tau" in c.lower()), numeric_cols[0])

    Y_wave_cols = [c for c in numeric_cols if c != Y_target_col]

    feat_df = df[X_cols + [Y_target_col] + Y_wave_cols].copy()
    return feat_df, X_cols, Y_wave_cols, Y_target_col