import numpy as np
from pinecone import Pinecone
import uuid

# 1. Load the 768-dim embeddings
print("Loading 768-dim embeddings...")
data = np.load('spider_768_embeddings.npz')
X = data['X']
y = data['y']

# 2. Connect to Pinecone
pc = Pinecone(api_key="pcsk_4Hj5Sb_4ChZ2R6ooBPbcMfGgXL4mUQNQV4QWD9kMjVF3ggD1gHFMkpVN6UTM356TZSJEao")
# Replace with your actual index host or name
index = pc.Index(host="data-analyst-sandbox-04ia07m.svc.aped-4627-b74a.pinecone.io")

# 3. Prepare data for upsert
# Pinecone expects (id, vector, metadata)
print(f"Preparing {len(X)} vectors for upsert...")
vectors_to_upsert = []
for i in range(len(X)):
    vectors_to_upsert.append({
        "id": str(uuid.uuid4()),
        "values": X[i].tolist(),
        "metadata": {"difficulty": y[i]}
    })

# 4. Upsert in batches (Pinecone recommended batch size is ~100)
batch_size = 100
print("Upserting to Pinecone...")
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)
    print(f"Upserted batch {i//batch_size + 1}")

print("Success! Data is now in Pinecone with 768 dimensions.")
