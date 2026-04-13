import pandas as pd
import numpy as np
import json
import os
import zipfile
from sentence_transformers import SentenceTransformer

# Load the specific BAAI model mentioned in the sources [1, 3]
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Verifying embedding dimension
print(model.get_sentence_embedding_dimension())

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

# Verification of Spider dataset
print(f"Type: {type(whole_dataset)}")
print(f"Length: {len(whole_dataset)}")
if len(whole_dataset) > 0:
    print(f"Keys in first entry: {whole_dataset[0].keys()}")

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
    
   # OLD Version:
# if score == 0: return "Easy"
# elif score in [1, 2]: return "Medium"  <-- This was catching too much
# elif score in [3, 4]: return "Hard"
# else: return "Extra Hard"              <-- This was catching too little

# NEW Version (Try this):
    if score == 0: 
        return "Easy"
    elif score == 1: 
        return "Medium"
    elif score in [2, 3]: 
        return "Hard"
    else: 
        return "Extra Hard"

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

# --- verification for extracting qs and labels ---
print(f"Shape: {embeddings.shape}") 
print(f"Dtype: {embeddings.dtype}")
print(f"First 5 values of first vector: {embeddings[0][:5]}") 
# Checking normalization (should be 1.0)
magnitudes = np.linalg.norm(embeddings, axis=1) 
is_normalized = np.allclose(magnitudes, 1.0, atol=1e-3)
print(f"Geometric Validation - Are all vectors unit length? {is_normalized}")

print(f"Min/Max Magnitudes: {magnitudes.min().round(4)}, {magnitudes.max().round(4)}")

# --- Verify Geometric validation ---
print(f"Magnitudes Check: {magnitudes.min().round(5)}, {magnitudes.max().round(5)}")


# --- X and y verify ---
print(f"Matrix X Shape: {X.shape}")                           
print(f"Vector y Shape: {y.shape}")                       
print(f"Data Type: {X.dtype}")

# This ensures X and y have the same number of rows
assert X.shape[0] == y.shape[0], "Error: Sample size mismatch between X and y!"

# This ensures you have all 4 difficulty levels (Easy, Medium, Hard, Extra Hard)
assert len(set(y)) == 4, f"Error: Expected 4 classes, but found {len(set(y))}!"

# --- Class Distribution Verification ---
counts = pd.Series(y).value_counts()
percentages = (counts / len(y)) * 100

print("\n--- Distribution Validation ---")
for category, pct in percentages.items():
    status = "✅ OK"
    if pct > 50:
        status = "❌ TOO HIGH (>50%)"
    elif pct < 5:
        status = "⚠️ LOW (<5%)"
    
    print(f"{category}: {pct:.2f}% {status}")

# Final Check
if all(5 <= pct <= 50 for pct in percentages):
    print("\nResult: Distribution is balanced and meets target criteria.")
else:
    print("\nResult: Distribution is imbalanced. Review the get_difficulty logic.")


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

print("\n--- Final File Integrity Check ---")
loaded = np.load('spider_final_embeddings.npz') 
print(f"Stored X Shape: {loaded['X'].shape}") 
print(f"Stored y Shape: {loaded['y'].shape}") 
# This compares the file on disk to the data in your memory
matches = np.allclose(loaded['X'], X)
print(f"Does the saved file match the memory? {matches}")