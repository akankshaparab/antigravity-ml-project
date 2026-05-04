import pandas as pd
import numpy as np
import json
import os
import zipfile
from sentence_transformers import SentenceTransformer

# Load the specific BAAI model (768 Dimensions)
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Verifying embedding dimension
print(f"Model Dimension: {model.get_sentence_embedding_dimension()}")

# Path to the zip file in your downloads
zip_path = r'C:\Users\Dell1234\Downloads\spider_data.zip'

# These are the specific files that contain the questions and labels
query_files = ['train_spider.json', 'train_others.json', 'dev.json']

whole_dataset = []

# Loop through each file to combine them into one 'Observation Set'
with zipfile.ZipFile(zip_path, 'r') as z:
    for file_name in query_files:
        internal_path = f'spider_data/{file_name}'
        with z.open(internal_path) as f:
            data_part = json.load(f)
            whole_dataset.extend(data_part)

print(f"Total query observations loaded: {len(whole_dataset)}")

def get_difficulty(sql_query):
    query = sql_query.upper()
    score = 0
    if "JOIN" in query: score += 1
    if "GROUP BY" in query: score += 1
    if "ORDER BY" in query: score += 1
    if "HAVING" in query: score += 1
    if "INTERSECT" in query or "UNION" in query or "EXCEPT" in query: score += 2
    if query.count("SELECT") > 1: score += 2
    
    if score == 0: return "Easy"
    elif score == 1: return "Medium"
    elif score in [2, 3]: return "Hard"
    else: return "Extra Hard"

# Extract questions and generate labels
questions = [item['question'] for item in whole_dataset]
labels = [get_difficulty(item['query']) for item in whole_dataset]
y = np.array(labels)

print("Generating 768-dim Embeddings...")
embeddings = model.encode(questions, show_progress_bar=True, normalize_embeddings=True)
X = np.array(embeddings)

print(f"Matrix X Shape: {X.shape}")                           
print(f"Vector y Shape: {y.shape}")                       

# Export the processed matrix
np.savez('spider_768_embeddings.npz', X=X, y=y)
print("Phase 1 Complete: 768-dim data preserved in 'spider_768_embeddings.npz'")