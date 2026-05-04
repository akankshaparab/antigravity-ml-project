import numpy as np
import matplotlib.pyplot as plt
from pinecone import Pinecone
from sklearn.decomposition import PCA
import seaborn as sns

# 1. Load your local Spider 768D data
print("Loading local Spider 768-dim embeddings...")
local_data = np.load('spider_768_embeddings.npz')
X_spider = local_data['X']

# 2. Connect to Pinecone
pc = Pinecone(api_key="pcsk_4Hj5Sb_4ChZ2R6ooBPbcMfGgXL4mUQNQV4QWD9kMjVF3ggD1gHFMkpVN6UTM356TZSJEao")
index = pc.Index(host="data-analyst-sandbox-04ia07m.svc.aped-4627-b74a.pinecone.io")

# 3. Fetch the 1,742 live records
# We use a dummy vector query to "pull" the top 1,742 records
print("Fetching 1,742 records from Pinecone index...")
dummy_vector = [0.0] * 768
query_response = index.query(
    vector=dummy_vector,
    top_k=1742,
    include_values=True
)

# Extract vectors into a numpy array
X_live = np.array([match['values'] for match in query_response['matches']])
print(f"Successfully retrieved {len(X_live)} vectors from live platform.")

# 4. Dimensionality Reduction (PCA) for Comparison
print("Performing PCA comparison (768D -> 2D)...")
pca = PCA(n_components=2)

# Combine both datasets to fit the PCA space equally
X_combined = np.vstack([X_spider, X_live])
X_pca = pca.fit_transform(X_combined)

# Split them back for plotting
pca_spider = X_pca[:len(X_spider)]
pca_live = X_pca[len(X_spider):]

# 5. Visualization
plt.figure(figsize=(12, 8))
plt.scatter(pca_spider[:, 0], pca_spider[:, 1], alpha=0.4, label='Spider Dataset', color='#6366f1')
plt.scatter(pca_live[:, 0], pca_live[:, 1], alpha=0.6, label='Live Platform Data', color='#f59e0b', marker='x')

plt.title("Distribution Comparison: Spider vs. Live Platform (768D PCA)", fontsize=15)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('live_vs_spider_comparison.png', dpi=300)
print("Analysis complete! Plot saved as 'live_vs_spider_comparison.png'")
plt.show()
