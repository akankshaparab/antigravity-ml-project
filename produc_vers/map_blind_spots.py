import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Load the 768D embeddings
print("Loading embeddings...")
data = np.load('spider_768_embeddings.npz')
X = data['X']
y = data['y']

# 2. Reduce to 2D for visualization
print("Reducing dimensions for mapping...")
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# 3. Create the Visualization
plt.figure(figsize=(12, 8))
# Mapping dataset labels to our display configuration
config = {
    'Easy': {'color': '#4CAF50', 'label': 'Easy'},
    'Medium': {'color': '#FFC107', 'label': 'Medium'},
    'Hard': {'color': '#FF9800', 'label': 'Hard'},
    'Extra Hard': {'color': '#F44336', 'label': 'Extra Hard'}
}

for diff_label, settings in config.items():
    mask = (y == diff_label)
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                label=settings['label'], 
                color=settings['color'],
                alpha=0.6, s=15)

plt.title('768D Embedding Space: Class Distribution & Blind Spots', fontsize=15, weight='bold')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Highlight "Blind Spot" areas (Regions with low density)
plt.annotate('Potential Blind Spot', xy=(5, 0), xytext=(7, 2),
             arrowprops=dict(facecolor='black', shrink=0.05))

# 4. Save and Show
save_path = 'blind_spot_map.png'
plt.savefig(save_path)
print(f"Map saved to {save_path}")

