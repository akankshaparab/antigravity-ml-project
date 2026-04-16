import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Visual Styling ---
sns.set_theme(style="white", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial', 'DejaVu Sans']

def generate_similarity_heatmap():
    """
    Creates a high-resolution similarity heatmap to visualize how SQL 
    queries cluster together based on their complexity labels.
    """
    input_file = 'spider_reduced_embeddings.npz'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the previous phases first.")
        return

    print("--- Phase 3: Similarity Heatmap Analysis ---")
    print(f"Loading data from {input_file}...")
    
    # 1. Load data
    data = np.load(input_file, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # 2. Filtering and Sorting
    # We take a subset (e.g., 50 samples per label) to keep the heatmap readable
    labels = ["Easy", "Medium", "Hard", "Extra Hard"]
    n_per_label = 40
    
    indices = []
    for label in labels:
        # Get indices for this label and take first n_per_label
        label_indices = np.where(y == label)[0]
        if len(label_indices) > 0:
            indices.extend(label_indices[:n_per_label])
    
    if not indices:
        print("Error: No labeled data found to plot.")
        return

    X_subset = X[indices]
    y_subset = y[indices]
    
    print(f"Generating similarity matrix for {len(X_subset)} samples...")
    
    # 3. Compute Cosine Similarity
    # This matrix shows how similar every query is to every other query
    sim_matrix = cosine_similarity(X_subset)
    
    # 4. Generate Heatmap
    plt.figure(figsize=(16, 12))
    
    # Use a premium color palette (magma or icefire look sophisticated)
    heatmap = sns.heatmap(
        sim_matrix, 
        cmap='magma', 
        xticklabels=False, 
        yticklabels=False,
        cbar_kws={'label': 'Cosine Similarity Score'}
    )
    
    # Add lines to separate the difficulty groups
    for i in range(1, len(labels)):
        pos = i * n_per_label
        plt.axhline(y=pos, color='white', linewidth=2, linestyle='--')
        plt.axvline(x=pos, color='white', linewidth=2, linestyle='--')
    
    # Add group labels at the center of each block
    for i, label in enumerate(labels):
        pos = (i * n_per_label) + (n_per_label // 2)
        plt.text(-5, pos, label, ha='right', va='center', fontweight='bold', fontsize=12)
        plt.text(pos, len(X_subset) + 5, label, ha='center', va='top', fontweight='bold', fontsize=12, rotation=45)

    # --- Polish Details ---
    plt.title('Similarity Heatmap: Clustering by SQL Complexity\nModel: BAAI/bge-small-en-v1.5', 
              fontsize=22, fontweight='bold', pad=30)
    
    plt.xlabel('SQL Query Samples (Grouped by Difficulty)', fontsize=14, labelpad=40)
    plt.ylabel('SQL Query Samples (Grouped by Difficulty)', fontsize=14, labelpad=40)
    
    # Annotate the diagonal (Self-Similarity)
    plt.annotate('Diagonal = Identity (Self-Similarity)', xy=(5, 5), xytext=(35, -15),
                 arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=5),
                 color='white', fontsize=11, fontweight='bold')

    plt.tight_layout()
    
    output_image = 'phase3_similarity_heatmap.png'
    plt.savefig(output_image, dpi=300)
    print(f"\n[SUCCESS] Heatmap saved to: {output_image}")
    plt.show()

if __name__ == "__main__":
    generate_similarity_heatmap()
