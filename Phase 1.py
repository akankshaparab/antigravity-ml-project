import pandas as pd
import numpy as np
import json
import os
import zipfile
from sentence_transformers import SentenceTransformer

# Load the specific BAAI model mentioned in the sources [1, 3]
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Path to the zip file in your downloads
# Using r'' (raw string) avoids the "invalid unicode escape" error caused by \U
zip_path = r'C:\Users\Dell1234\Downloads\spider_data.zip'

# These are the specific files that contain the questions and labels
query_files = ['train_spider.json', 'train_others.json', 'dev.json']

whole_dataset = []

# Loop through each file to combine them into one 'Observation Set'
# We use zipfile to read directly from the archive
with zipfile.ZipFile(zip_path, 'r') as z:
    for file_name in query_files:
        # Construct the path inside the zip
        internal_path = f'spider_data/{file_name}'
        with z.open(internal_path) as f:
            # json.load turns the file content into a list we can use
            data_part = json.load(f)
            whole_dataset.extend(data_part)

# Verify the count to ensure you have the full corpus (should be >10,000)
print(f"Total query observations loaded: {len(whole_dataset)}")

def get_difficulty(sql_query):
    query = sql_query.upper()
    score = 0
    # Basic heuristics based on SQL keywords
    if "JOIN" in query: score += 1
    if "GROUP BY" in query: score += 1
    if "ORDER BY" in query: score += 1
    if "HAVING" in query: score += 1
    if "INTERSECT" in query or "UNION" in query or "EXCEPT" in query: score += 2
    if query.count("SELECT") > 1: score += 2  # Indicates nested query
    
    if score == 0: return "Easy"
    elif score in [1, 2]: return "Medium"
    elif score in [3, 4]: return "Hard"
    else: return "Extra Hard"

# Extract questions and generate labels
questions = [item['question'] for item in whole_dataset]
labels = [get_difficulty(item['query']) for item in whole_dataset]

# Create Label Vector y
y = np.array(labels)

print("Generating Embeddings (Matrix X)...")
# Vectorization: Map text to 384 dimensions and normalize
# This transforms each sentence into a list of 384 numbers
embeddings = model.encode(questions, show_progress_bar=True, normalize_embeddings=True)
X = np.array(embeddings)

# Geometric Validation: Verify magnitudes are ≈ 1
magnitudes = np.linalg.norm(X, axis=1)
is_normalized = np.allclose(magnitudes, 1.0, atol=1e-3)

print("-" * 30)
print(f"Matrix X Shape: {X.shape}")
print(f"Vector y Shape: {y.shape}")
print(f"Are vectors unit length? {is_normalized}")
print(f"Difficulty Stats:\n{pd.Series(y).value_counts()}")

# Export the processed matrix for Rank Analysis in Phase 2
np.savez('spider_final_embeddings.npz', X=X, y=y)
print("Phase 1 Complete: Data preserved in 'spider_final_embeddings.npz'")
