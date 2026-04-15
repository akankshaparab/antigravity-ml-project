import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# --- Configuration ---
sns.set_theme(style="darkgrid")
# Custom palette for complexity labels
COMPLEXITY_LABELS = ["Easy", "Medium", "Hard", "Extra Hard"]

def run_phase_3_analysis():
    input_file = 'spider_reduced_embeddings.npz'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Ensure Phase 1 and 2 are complete.")
        return

    print("--- Phase 3: Geometric Cluster Analysis (Objective 3) ---")
    print(f"Loading data from {input_file}...")
    
    # Load data
    data = np.load(input_file, allow_pickle=True)
    X = data['X']
    y_raw = data['y'] 

    # Map string labels to numeric values for coloring if they are strings
    label_map = {"Easy": 0, "Medium": 1, "Hard": 2, "Extra Hard": 3}
    if isinstance(y_raw[0], str):
        y = np.array([label_map.get(label, -1) for label in y_raw])
    else:
        y = y_raw

    print(f"Dataset shape: {X.shape} (Reduced embeddings)")

    # 1. Dimensionality Reduction (Step 1 of image)
    print("\n[Step 1] Reducing dimensions for visualization...")
    
    # PCA: Captures global structural variance
    print("Running PCA (Linear Projection)...")
    pca_2d = PCA(n_components=2).fit_transform(X)
    
    # t-SNE: Captures local clusters and neighborhoods
    print("Running t-SNE (Non-Linear Manifold)... This may take a moment.")
    tsne_2d = TSNE(
        n_components=2, 
        perplexity=30, 
        random_state=42, 
        init='pca'
    ).fit_transform(X)

    # 2. Apply Statistical Metrics (Step 3 of image)
    print("\n[Step 3] Calculating Cluster Cohesion Metrics...")
    # Silhouette Score: Higher means more coherent clusters
    s_score = silhouette_score(X, y)
    print(f">>> Global Silhouette Score: {s_score:.4f}")
    
    if s_score > 0.1:
        print("Conclusion: Queries of the same type form COHERENT geometric clusters.")
    else:
        print("Conclusion: Queries are largely GEOMETRICALLY DISPERSED relative to difficulty labels.")

    # 3. Visual Inspection (Step 2 of image)
    print("\n[Step 2] Generating Scatter Plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot PCA
    scatter1 = ax1.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y, cmap='viridis', alpha=0.5, s=25)
    ax1.set_title("PCA Projection (Linear)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot t-SNE
    scatter2 = ax2.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=y, cmap='viridis', alpha=0.5, s=25)
    ax2.set_title("t-SNE Projection (Non-Linear)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Add a unified legend
    legend_elements = scatter1.legend_elements()[0]
    fig.legend(legend_elements, COMPLEXITY_LABELS, loc='lower center', ncol=4, 
               title="SQL Complexity Labels", fontsize=11, title_fontsize=12)

    plt.suptitle(f"Phase 3: Geometric Analysis of SQL Complexity Embeddings\nModel: BAAI/bge-small-en-v1.5 | Silhouette Score: {s_score:.4f}", 
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    output_plot = 'phase3_geometric_clusters.png'
    plt.savefig(output_plot, dpi=300)
    print(f"\n[SUCCESS] Analysis plot saved to: {output_plot}")

if __name__ == "__main__":
    run_phase_3_analysis()
