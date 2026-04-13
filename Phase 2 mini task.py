import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, KernelPCA
import time
import os

# --- Configuration & Styling ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def run_phase_2_mini_task():
    # 1. Load the data generated in Phase 1
    input_file = 'spider_final_embeddings.npz'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run Phase 1 first.")
        return

    print(f"--- Phase 2: Basis Selection Mini Task ---")
    print(f"Loading embeddings from {input_file}...")
    data = np.load(input_file)
    X = data['X']  # The 384-dimensional vectors
    y = data['y']  # The difficulty labels
    
    print(f"Input Matrix X shape: {X.shape}")
    print(f"Label Vector y shape: {y.shape}")

    # 2. Evaluate PCA Variants (Objective 1)
    # We define a benchmark function to measure time and performance
    def benchmark_pca(name, transformer, data_subset):
        print(f"\nRunning {name}...")
        start = time.time()
        transformed = transformer.fit_transform(data_subset)
        duration = time.time() - start
        print(f"{name} completed in {duration:.4f} seconds.")
        return transformed

    # Comparative Analysis
    # Standard PCA (Baseline)
    std_pca = PCA(n_components=50) # Comparing the first 50 components
    X_std = benchmark_pca("Standard PCA", std_pca, X)

    # Incremental PCA (Memory Efficient)
    inc_pca = IncrementalPCA(n_components=50)
    X_inc = benchmark_pca("Incremental PCA", inc_pca, X)

    # Kernel PCA (Non-linear projection using RBF)
    # Note: Using a subset for speed in mini-task
    k_pca = KernelPCA(n_components=50, kernel='rbf', n_jobs=-1)
    X_kernel = benchmark_pca("Kernel PCA", k_pca, X[:2000])

    # Sparse PCA (Interpretability/Sparsity focus)
    # Note: Highly intensive, using a smaller subset and fewer components
    s_pca = SparsePCA(n_components=10, random_state=42, n_jobs=-1)
    X_sparse = benchmark_pca("Sparse PCA", s_pca, X[:1000])

    # 3. Find the Intrinsic Dimensionality (Objective 2 & 3)
    print("\n--- Identifying the Basis & Intrinsic Dimensionality ---")
    # Fit PCA on the full 384 dimensions to see the absolute variance capture
    full_pca = PCA().fit(X)
    variance_ratios = full_pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratios)

    # Identify the number of components needed for 95% variance
    target_threshold = 0.95
    intrinsic_dim = np.argmax(cumulative_variance >= target_threshold) + 1
    
    print(f"Total Dimensions in Original Space: {X.shape[1]}")
    print(f"Dimensions needed for {target_threshold*100}% variance: {intrinsic_dim}")
    print(f"Variance explained by first 2 components: {cumulative_variance[1]*100:.2f}%")

    # 4. Visualizing the "True" Dimensionality
    plt.figure(figsize=(12, 6))
    
    # Plotting the Cumulative Explained Variance
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             color='#4f46e5', lw=3, label='Cumulative Variance')
    
    # Visual markers for the threshold and result
    plt.axhline(y=target_threshold, color='#ef4444', linestyle='--', alpha=0.8, 
                label=f'{target_threshold*100}% Threshold')
    plt.axvline(x=intrinsic_dim, color='#10b981', linestyle=':', alpha=0.8, 
                label=f'Intrinsic Dim ({intrinsic_dim})')
    
    plt.fill_between(range(1, intrinsic_dim + 1), 0, cumulative_variance[:intrinsic_dim], 
                     color='#4f46e5', alpha=0.1)

    plt.title('Subspace Projection: Cumulative Explained Variance (BGE Embeddings)', fontsize=14, pad=15)
    plt.xlabel('Principal Component Index', fontsize=12)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    plt.xlim(0, 150) # Zooming in on the first 150 components for better visibility
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save output for record-keeping
    plot_filename = 'phase2_variance_analysis.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    print(f"\n[SUCCESS] Analysis plot saved as '{plot_filename}'")
    
    print("\nPhase 2 Mini Task Conclusion:")
    print(f"The 'meaning' of your 384D query space is effectively captured in just {intrinsic_dim} dimensions.")
    print("This confirms that the embedding space is highly redundant and can be significantly compressed.")

if __name__ == "__main__":
    run_phase_2_mini_task()

plt.savefig('variance_analysis_plot.png')
print("Graph saved as variance_analysis_plot.png")