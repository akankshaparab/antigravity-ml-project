import numpy as np
from sklearn.decomposition import PCA
import os
import time

def run_phase_2():
    input_file = 'spider_final_embeddings.npz'
    output_file = 'spider_reduced_embeddings.npz'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run Phase 1 first.")
        return

    print("--- Phase 2: Dimensionality Reduction ---")
    
    # 1. Load Data
    print(f"Loading dense embeddings from {input_file}...")
    data = np.load(input_file)
    X = data['X']  # 384-dimensional embeddings
    y = data['y']  # Labels
    
    print(f"Original X shape: {X.shape}")

    # 2. Fit PCA dynamically to retain 95% of information variant 
    print("\nFitting PCA to retain 95% of the total variance...")
    start = time.time()
    
    # Passing a float (0.0 < n_components < 1.0) automatically selects 
    # the number of components necessary to capture that much variance ratio.
    pca = PCA(n_components=0.95, random_state=42)
    X_reduced = pca.fit_transform(X)
    
    duration = time.time() - start
    
    # 3. Present Results
    print(f"PCA completed in {duration:.4f} seconds.")
    print(f"New X shape: {X_reduced.shape}")
    print(f"Dimensions compressed from {X.shape[1]} down to {X_reduced.shape[1]}")
    
    # 4. Save Reduced Dataset
    print(f"\nSaving transformed embeddings to {output_file}...")
    np.savez_compressed(output_file, X=X_reduced, y=y)
    print("Process Complete. Ready for Phase 3!")

if __name__ == "__main__":
    run_phase_2()
