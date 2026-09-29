import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.random.randn(100,6)
X[:, 1] = X[:, 0] ** 3 + np.random.normal(0, 0.1, 100)
X[:, 3] = -np.log(np.abs(X[:, 2]) + 1)

columns = ['Var_A', 'Var_B', 'Var_C', 'Var_D', 'Var_E', 'Var_F']
df = pd.DataFrame(X, columns = columns)

spearman_corr = df.corr(method = 'spearman')

corr_distance = 1 - spearman_corr

n_feature_clusters = 3
kmeans_features = KMeans(n_clusters=n_feature_clusters, random_state =42, n_init=10)
feature_cluster_labels = kmeans_features.fit_predict(corr_distance)

feature_results = pd.DataFrame({'Feature': df.columns, 'Cluster': feature_cluster_labels})
print("Feature Clusters (Based on Spearman Distance)")
print(feature_results.sort_values(by='Cluster'))

df_ranked = df.rank()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ranked)

n_sample_clusters = 3
kmeans_samples = KMeans(n_clusters=n_sample_clusters, random_state=42, n_init=10)
df['Sample_Cluster'] = kmeans_samples.fit_predict(X_scaled)

print("Frint 5 rows with sample cluster assignments")
print(df.head())

fig, axes = plt.subplots(1, 2, figsize=(14,5))
sns.heatmap(spearman_corr, annot=True, cmap='vlag', fmt = '.2f', ax=axes[0], vmin =-1, vmax = 1)
axes[0].set_title('Spearman Correlation Matrix')

# Plot Feature Clusters
sns.barplot(data=feature_results, x='Feature', y='Cluster', palette='viridis', ax=axes[1])
axes[1].set_title('K-Means Feature Cluster Assignment')

plt.tight_layout()
plt.show()