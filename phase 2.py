import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# --- Visualization Styling ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial', 'DejaVu Sans']

def run_phase_2():
    input_file = 'spider_final_embeddings.npz'
    output_file = 'spider_reduced_embeddings.npz'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run Phase 1 first.")
        return

    print("--- Phase 2: Dimensionality Reduction Comparison ---")
    
    # 1. Load Data
    print(f"Loading dense embeddings from {input_file}...")
    data = np.load(input_file)
    X = data['X']  # 384-dimensional embeddings
    y = data['y']  # Labels
    print(f"Original X shape: {X.shape}")

    # --- TECHNIQUE 1: STANDARD PCA ---
    print("\n[1] Running Standard PCA (95% variance)...")
    start = time.time()
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X)
    duration = time.time() - start
    n_comp = X_pca.shape[1]
    
    intrinsic_dim = X_pca.shape[1]
    print(f"Intrinsic Dimensionality: {intrinsic_dim}")
    
    X_recon_pca = pca.inverse_transform(X_pca)
    mse_pca = np.mean((X - X_recon_pca) ** 2)
    var_pca = np.sum(pca.explained_variance_ratio_) * 100
    
    print(f"Time: {duration:.4f}s | Components: {n_comp} | MSE: {mse_pca:.10f} | Variance: {var_pca:.2f}%")

    # --- VISUALIZATION: CUMULATIVE VARIANCE ---
    print("\nGenerating Variance Analysis Plot...")
    # We fit a full PCA to see the elbow curve
    full_pca = PCA().fit(X)
    cum_var = np.cumsum(full_pca.explained_variance_ratio_)
    
    plt.figure(figsize=(12, 6), dpi=100)
    
    # Plot the curve
    plt.plot(range(1, len(cum_var) + 1), cum_var, color='#4f46e5', lw=3, label='Cumulative Variance')
    
    # Markers for target threshold (95%) and intrinsic dimensionality
    plt.axhline(y=0.95, color='#ef4444', linestyle='--', alpha=0.8, label='95% Information Threshold')
    plt.axvline(x=intrinsic_dim, color='#10b981', linestyle=':', alpha=0.8, label=f'Intrinsic Dim ({intrinsic_dim})')
    
    # Highlight the captured region
    plt.fill_between(range(1, intrinsic_dim + 1), 0, cum_var[:intrinsic_dim], color='#4f46e5', alpha=0.1)

    # Styling and Labels
    plt.title('Subspace Projection: Cumulative Explained Variance', fontsize=16, pad=20, weight='bold')
    plt.xlabel('Principal Component Index', fontsize=12, labelpad=10)
    plt.ylabel('Cumulative Variance Ratio', fontsize=12, labelpad=10)
    plt.xlim(0, 200) # Focusing on the most significant components
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = 'variance_analysis_plot.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Variance analysis plot saved to: {plot_path}")

    # --- TECHNIQUE 2: INCREMENTAL PCA ---
    # Useful for datasets that don't fit in memory (processed in batches)
    print(f"\n[2] Running Incremental PCA (using {n_comp} components)...")
    start = time.time()
    ipca = IncrementalPCA(n_components=n_comp, batch_size=500)
    X_ipca = ipca.fit_transform(X)
    duration = time.time() - start
    
    X_recon_ipca = ipca.inverse_transform(X_ipca)
    mse_ipca = np.mean((X - X_recon_ipca) ** 2)
    var_ipca = np.sum(ipca.explained_variance_ratio_) * 100
    
    print(f"Time: {duration:.4f}s | MSE: {mse_ipca:.10f} | Variance: {var_ipca:.2f}%")

    # --- TECHNIQUE 3: KERNEL PCA ---
    # Good for non-linear dimensionality reduction (using RBF kernel)
    print(f"\n[3] Running Kernel PCA (RBF kernel, {n_comp} components)...")
    print("Note: This can be computationally expensive...")
    start = time.time()
    # fit_inverse_transform=True allows us to calculate reconstruction error
    kpca = KernelPCA(n_components=n_comp, kernel='rbf', fit_inverse_transform=True, n_jobs=-1, random_state=42)
    X_kpca = kpca.fit_transform(X)
    duration = time.time() - start
    
    X_recon_kpca = kpca.inverse_transform(X_kpca)
    mse_kpca = np.mean((X - X_recon_kpca) ** 2)
    print(f"Time: {duration:.4f}s | MSE: {mse_kpca:.10f}")

    # --- TECHNIQUE 4: SPARSE PCA ---
    # Finds components that are sparse combinations of original features
    print(f"\n[4] Running Sparse PCA ({n_comp} components)...")
    print("Note: This might take a while for high dimensions...")
    start = time.time()
    spca = SparsePCA(n_components=n_comp, alpha=1, random_state=42, n_jobs=-1)
    X_spca = spca.fit_transform(X)
    duration = time.time() - start
    
    # Reconstruction for SparsePCA: X_reconstructed = X_transformed * components_
    X_recon_spca = np.dot(X_spca, spca.components_) + X.mean(axis=0)
    mse_spca = np.mean((X - X_recon_spca) ** 2)
    print(f"Time: {duration:.4f}s | MSE: {mse_spca:.10f}")

    # 4. Save the best/standard result for downstream phases
    print(f"\nSaving transformed embeddings (Standard PCA) to {output_file}...")
    np.savez_compressed(output_file, X=X_pca, y=y)
    print("Phase 2 complete! Use 'spider_reduced_embeddings.npz' for clustering in Phase 3.")

if __name__ == "__main__":
    run_phase_2()
