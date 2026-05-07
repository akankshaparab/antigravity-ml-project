import numpy as np
import os
import json
import zipfile
import pickle
from sklearn.decomposition import PCA
from pinecone import Pinecone

# --- SETTINGS ---
DRY_RUN = False  # Ready for migration
PINECONE_API_KEY = "pcsk_4Hj5Sb_4ChZ2R6ooBPbcMfGgXL4mUQNQV4QWD9kMjVF3ggD1gHFMkpVN6UTM356TZSJEao"
SOURCE_INDEX_HOST = "data-analyst-sandbox-04ia07m.svc.aped-4627-b74a.pinecone.io"
TARGET_INDEX_NAME = "data-analyst-compressed"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = r'C:\Users\Dell1234\Downloads\spider_data.zip'
EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, 'spider_768_embeddings.npz')
ROUTER_ARTIFACTS_PATH = os.path.join(SCRIPT_DIR, 'router_artifacts.pkl')

print(f"--- Phase 1: Gathering All Vectors (Local + Live) ---")

# 1a. Load 9,693 Spider Vectors from Local Files
try:
    data = np.load(EMBEDDINGS_PATH)
    spider_vectors = data['X']
    spider_labels = data['y']
    print(f"Loaded {len(spider_vectors)} Spider vectors from local .npz")
except Exception as e:
    print(f"Error loading local embeddings: {e}")
    exit(1)

# 1b. Load Spider Metadata from ZIP
spider_metadata = []
if os.path.exists(ZIP_PATH):
    print("Extracting Spider metadata from ZIP...")
    query_files = ['train_spider.json', 'train_others.json', 'dev.json']
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for file_name in query_files:
            internal_path = f'spider_data/{file_name}'
            with z.open(internal_path) as f:
                data_part = json.load(f)
                for item in data_part:
                    spider_metadata.append({
                        "question": item.get('question', ''),
                        "query": item.get('query', ''),
                        "difficulty": spider_labels[len(spider_metadata)] if len(spider_metadata) < len(spider_labels) else "N/A"
                    })
else:
    print(f"Error: Zip file not found at {ZIP_PATH}")
    exit(1)

# 1c. Fetch the 1,742 Live Records from Pinecone Sandbox
print(f"Connecting to Sandbox to fetch 1,742 live records...")
pc = Pinecone(api_key=PINECONE_API_KEY)
source_index = pc.Index(host=SOURCE_INDEX_HOST)

# We query for the 1,742 records (using a zero vector to fetch the remaining ones)
# Note: In a production migration, you'd use pagination/list_ids, but based on your
# fetch_and_compare.py, this top_k approach is what you used to define the 'live' set.
res = source_index.query(
    vector=[0.0] * 768,
    top_k=1742,
    include_values=True,
    include_metadata=True
)

live_vectors = np.array([m['values'] for m in res['matches']])
live_metadata = [m['metadata'] for m in res['matches']]
live_ids = [m['id'] for m in res['matches']]

print(f"Fetched {len(live_vectors)} live records from Sandbox.")

# 1d. Combine Datasets
all_vectors = np.vstack([spider_vectors, live_vectors])
all_metadata = spider_metadata + live_metadata
# We use numeric IDs for Spider and original IDs for Live
all_ids = [str(i) for i in range(len(spider_vectors))] + live_ids

print(f"Total Combined Records: {len(all_vectors)}")

# 2. Apply PCA Transformation
print(f"\n--- Phase 2: Dimensionality Reduction ---")
try:
    with open(ROUTER_ARTIFACTS_PATH, 'rb') as f:
        artifacts = pickle.load(f)
        pca = artifacts['pca']
    print(f"Successfully loaded PCA model from artifacts.")
    compressed_vectors = pca.transform(all_vectors)
except Exception as e:
    print(f"Error applying PCA: {e}")
    exit(1)

print(f"Compressed Shape: {compressed_vectors.shape}")

# 3. Upsert to Compressed Index
print(f"\n--- Phase 3: Pinecone Migration ---")
if DRY_RUN:
    print("[DRY RUN] Skipping upsert.")
else:
    target_index = pc.Index(TARGET_INDEX_NAME)
    
    records = []
    for i in range(len(compressed_vectors)):
        records.append({
            "id": all_ids[i],
            "values": compressed_vectors[i].tolist(),
            "metadata": all_metadata[i]
        })

    print(f"Upserting {len(records)} records to {TARGET_INDEX_NAME}...")
    for i in range(0, len(records), 100):
        batch = records[i:i+100]
        target_index.upsert(vectors=batch)
        if i % 1000 == 0:
            print(f"Progress: {i}/{len(records)} upserted...")

    print("SUCCESS: Migration of 11,435 records complete!")