import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Load the new 768D embeddings
input_file = 'spider_768_embeddings.npz'
print(f"Loading {input_file}...")
data = np.load(input_file)
X = data['X']

# 2. Run PCA to calculate explained variance
print("Calculating PCA variance components...")
pca = PCA().fit(X)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# 3. Create the Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot: 768D Intrinsic Dimensionality')
plt.axhline(y=0.90, color='r', linestyle='-', label='90% Variance Threshold')
plt.axhline(y=0.95, color='g', linestyle='-', label='95% Variance Threshold')
plt.grid(True)
plt.legend()

# 4. Save and Show
save_path = 'scree_plot_768.png'
plt.savefig(save_path)
print(f"Scree plot saved to {save_path}")
plt.show()

# 5. Output some key stats
n_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\n--- Results ---")
print(f"To capture 90% info: {n_90} dimensions needed.")
print(f"To capture 95% info: {n_95} dimensions needed.")
print(f"Total Dimensions available: 768")
