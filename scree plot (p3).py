import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# --- Visual Styling ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial', 'DejaVu Sans']

def run_scree_analysis():
    """
    Loads original dense embeddings and generates a high-fidelity Scree Plot 
    to visualize explained variance and justify dimension reduction.
    """
    input_file = 'spider_reduced_embeddings.npz'
    
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found. Ensure Phase 1 is complete.")
        return

    print(f"--- Phase 3: PCA Scree Plot Analysis ---")
    print(f"Loading embeddings from {input_file}...")
    
    # Load data
    data = np.load(input_file)
    X = data['X']
    
    print(f"Original dimension: {X.shape[1]}")
    
    # Fit PCA to capture the most significant variance
    # Using 100 components is usually enough to see the 'elbow'
    n_components = min(100, X.shape[1])
    print(f"Analyzing variance across top {n_components} components...")
    
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    # --- Create Premium Visualization ---
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 1. Bar Plot: Individual Variance
    ax1.bar(range(1, n_components + 1), exp_var, alpha=0.7, color='#3498db', 
            label='Individual Variance Explained', edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Principal Component Index', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Individual Variance Ratio', fontsize=13, fontweight='bold', color='#2980b9')
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 2. Line Plot: Cumulative Variance
    ax2 = ax1.twinx()
    ax2.plot(range(1, n_components + 1), cum_var, marker='o', markersize=4, 
             color='#e74c3c', linewidth=2.5, label='Cumulative Variance Explained')
    ax2.set_ylabel('Cumulative Variance Ratio', fontsize=13, fontweight='bold', color='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    
    # 3. Threshold Line (95% variance)
    threshold = 0.95
    ax2.axhline(y=threshold, color='#27ae60', linestyle='--', linewidth=2, label=f'{int(threshold*100)}% Threshold')
    
    # Annotate the intersection
    components_at_threshold = np.argmax(cum_var >= threshold) + 1
    ax2.annotate(f'{components_at_threshold} PCs for {int(threshold*100)}% Info', 
                 xy=(components_at_threshold, threshold), 
                 xytext=(components_at_threshold + 5, threshold - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12, fontweight='bold', color='#1e8449',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1e8449", alpha=0.9))

    # --- Polish Details ---
    plt.title('Final Dimensionality Analysis: PCA Scree Plot', fontsize=20, fontweight='bold', pad=25)
    
    # Merging legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', 
               frameon=True, shadow=True, fontsize=11)

    plt.tight_layout()
    
    # Save the plot
    output_filename = 'phase3_scree_plot.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\n[SUCCESS] Scree plot visualization saved to: {output_filename}")
    
    # Final Summary
    print(f"\nAnalysis Summary:")
    print(f"- Total components analyzed: {n_components}")
    print(f"- Components required for 95% variance: {components_at_threshold}")
    print(f"- Top component captures {exp_var[0]*100:.2f}% of total variance.")
    
    plt.show()

if __name__ == "__main__":
    run_scree_analysis()
