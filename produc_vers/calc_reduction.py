import pickle
import os

with open('router_artifacts.pkl', 'rb') as f:
    data = pickle.load(f)
    pca = data['pca']
    print(f"PCA Components: {pca.n_components_}")
    print(f"Original Dimensions: {pca.components_.shape[1]}")
    
    # Calculate reduction ratio
    ratio = pca.n_components_ / pca.components_.shape[1]
    print(f"Reduction Ratio: {ratio:.2%}")
    
    # Original file size
    orig_size = os.path.getsize('spider_768_embeddings.npz') / (1024 * 1024)
    print(f"Original File Size: {orig_size:.2f} MB")
    
    # Estimated reduced size
    est_size = orig_size * ratio
    print(f"Estimated Reduced Data Size: {est_size:.2f} MB")
