import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA
import os
import time

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
