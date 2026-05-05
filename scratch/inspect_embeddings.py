import numpy as np
data = np.load('produc_vers/spider_768_embeddings.npz')
print(f"Keys: {list(data.keys())}")
y = data['y']
print(f"Labels shape: {y.shape}")
unique_y = np.unique(y)
print(f"Unique labels: {unique_y}")
if len(y) > 0:
    print(f"Sample label: {y[0]} (Type: {type(y[0])})")
