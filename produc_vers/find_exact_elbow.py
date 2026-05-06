import numpy as np
from sklearn.decomposition import PCA

# 1. Load the 768D embeddings
data = np.load('produc_vers/spider_768_embeddings.npz')
X = data['X']

# 2. Calculate Variance
pca = PCA().fit(X)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = len(cumulative_variance)
x = np.arange(n_components)

# 3. Geometric Elbow Detection (Distance from the chord)
# Connect the first and last points of the curve with a line
start_point = np.array([x[0], cumulative_variance[0]])
end_point = np.array([x[-1], cumulative_variance[-1]])
line_vec = end_point - start_point
line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

# Find the point furthest from that line
distances = []
for i in range(len(x)):
    point = np.array([x[i], cumulative_variance[i]])
    vec_to_point = point - start_point
    scalar_proj = np.dot(vec_to_point, line_vec_norm)
    proj_vec = scalar_proj * line_vec_norm
    dist_vec = vec_to_point - proj_vec
    distances.append(np.sqrt(np.sum(dist_vec**2)))

elbow_index = np.argmax(distances)
elbow_val = cumulative_variance[elbow_index]

print(f"--- Precise Mathematical Conclusion ---")
print(f"Mathematical Elbow Point: Component {elbow_index + 1}")
print(f"Variance explained at Elbow: {elbow_val:.2%}")
print(f"\nConclusion for Paper: The 'true' information density of this space is captured by the first {elbow_index + 1} components.")
