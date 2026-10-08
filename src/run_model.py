import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

CSV_PATH = "/Users/aldensantoso/Documents/plasma_research/ML Project/combined_plasma_data_clean_clean.csv"

def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def find_target_col(df):
    for alt in ("etau", "taue", "tau_e", "eta_u"):
        match = next((c for c in df.columns if _norm(c) == _norm(alt)), None)
        if match is not None:
            return match
    return None

def load_and_prepare_data(csv_path=CSV_PATH, feature_cols=None, target_col=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    if target_col is None:
        target_col = "ETAU"
        if target_col not in df.columns:
            target_col = "tau_e"
        if target_col not in df.columns:
            target_col = find_target_col(df)

    if target_col is None or target_col not in df.columns:
        raise ValueError(f"No ETAU / tau_e target column found. Available columns: {list(df.columns)}")

    if feature_cols is None:
        feature_cols = [
            "BZXR",
            "PBEAM_A",
            "PBEAM_B",
            "PBEAM_C",
            "PBEAM_TOT",
            "PTRANSP",
            "NES",
        ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    work = df[feature_cols + [target_col]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work[work[target_col].notna() & (work[target_col] > 0)].copy()

    # Remove extreme target outliers: 1st to 99th percentile
    low, high = work[target_col].quantile(0.01), work[target_col].quantile(0.99)
    work = work[(work[target_col] >= low) & (work[target_col] <= high)].copy()

    return work, target_col, feature_cols

def _cluster_summary_table(work, target_col, selected_features):
    summary_cols = selected_features + ([target_col] if target_col in work.columns else [])
    sort_by = target_col if target_col in work.columns else selected_features[0]
    cluster_summary = (
        work.groupby("Cluster_Label")[summary_cols]
        .mean()
        .sort_values(by=sort_by, ascending=False)
    )
    return cluster_summary

def run_analysis(feature_cols=None, regime_name="controlled", csv_path=CSV_PATH, n_clusters=4):
    if feature_cols is None:
        feature_cols = [
            "BZXR",
            "PBEAM_A",
            "PBEAM_B",
            "PBEAM_C",
            "PBEAM_TOT",
            "PTRANSP",
            "NES",
        ]

    work, target_col, selected_features = load_and_prepare_data(
        csv_path=csv_path,
        feature_cols=feature_cols,
        target_col="ETAU",
    )

    X = work[selected_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=25)
    work["Cluster_Label"] = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, work["Cluster_Label"])
    cluster_counts = work["Cluster_Label"].value_counts().sort_index()

    print("================ CLUSTERING SANITY CHECK ================")
    print(f"Regime: {regime_name}")
    print(f"Features used: {selected_features}")
    print(f"Silhouette Score: {sil_score:.3f}")
    if sil_score > 0.5:
        print(" -> Assessment: Strong, well-separated cluster structure.")
    elif sil_score > 0.25:
        print(" -> Assessment: Fair structure; expected overlap across continuous plasma regimes.")
    else:
        print(" -> Assessment: Low separation; consider adjusting n_clusters or feature selection.")

    print("\nCluster Size Balance:")
    for label, count in cluster_counts.items():
        pct = (count / len(work)) * 100
        print(f"  Cluster {label}: {count} shots ({pct:.1f}%)")
    print("=========================================================")

    cluster_summary = _cluster_summary_table(work, target_col, selected_features)
    print("\n--- Cluster Means ---")
    print(cluster_summary.to_string())

    # Plot 1: target by cluster
    if target_col in work.columns:
        plt.figure(figsize=(9, 4.5))
        sns.boxplot(data=work, x="Cluster_Label", y=target_col, palette="viridis")
        sns.stripplot(data=work, x="Cluster_Label", y=target_col, color="black", alpha=0.15, size=2)
        plt.title(f"{regime_name}: {target_col} Distribution by Cluster")
        plt.xlabel("Cluster Label")
        plt.ylabel(target_col)
        plt.tight_layout()
        plt.show()

    # Plot 2: RF prediction of target
    if target_col in work.columns:
        X_model = work[selected_features + ["Cluster_Label"]].copy()
        y_model = work[target_col].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X_model, y_model, test_size=0.2, random_state=42
        )

        rf = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        eta_r2 = r2_score(y_test, y_pred)
        eta_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n{target_col} prediction using KMeans clusters: R^2 = {eta_r2:.3f}")
        print(f"{target_col} prediction using KMeans clusters: RMSE = {eta_rmse:.3f}")

        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, y_pred, s=40, alpha=0.7, color="tab:blue")
        min_v = min(y_test.min(), y_pred.min())
        max_v = max(y_test.max(), y_pred.max())
        plt.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.5, label="1:1 line")
        plt.xlabel(f"Actual {target_col}")
        plt.ylabel(f"Predicted {target_col}")
        plt.title(f"{regime_name}: {target_col} predicted vs actual")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.text(
            0.02, 0.98,
            f"R² = {eta_r2:.3f}\nRMSE = {eta_rmse:.3f}",
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9),
        )
        plt.tight_layout()
        plt.show()

    # Plot 3: PCA cluster map
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    var1, var2 = pca.explained_variance_ratio_ * 100
    total_var = var1 + var2

    plt.figure(figsize=(8, 5))
    for cl in sorted(work["Cluster_Label"].unique()):
        mask = (work["Cluster_Label"] == cl).values
        plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1], s=30, alpha=0.7, label=f"Cluster {cl}")

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        s=180,
        marker="X",
        c="black",
        linewidths=1.5,
        label="Centroid",
    )

    plt.title(f"{regime_name}: 2D PCA View of Operating Space ({total_var:.1f}% variance preserved)")
    plt.xlabel(f"PC 1 ({var1:.1f}% Variance)")
    plt.ylabel(f"PC 2 ({var2:.1f}% Variance)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PCA loadings table
    pca_table = pd.DataFrame(
        pca.components_,
        columns=selected_features,
        index=["PC1", "PC2"],
    )
    print("\nPCA component loadings:")
    print(pca_table.to_string())

    report = {
        "regime_name": regime_name,
        "target_col": target_col,
        "selected_features": selected_features,
        "n_clusters": n_clusters,
        "silhouette_score": sil_score,
        "cluster_counts": cluster_counts.to_dict(),
        "cluster_summary": cluster_summary,
    }

    return {
        "report": report,
        "cluster_summary": cluster_summary,
        "target_col": target_col,
        "selected_features": selected_features,
        "cluster_labels": work["Cluster_Label"].values,
    }