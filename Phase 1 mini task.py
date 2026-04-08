import pandas as pd
import numpy as np
import json
import zipfile
from sentence_transformers import SentenceTransformer
# Load the specific BAAI model mentioned in the sources [1, 3]
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
# Load the Spider dataset (JSON format) [4]
with zipfile.ZipFile(r"C:\Users\Dell1234\Downloads\spider_data.zip", "r") as z:
    with z.open("spider_data/train_spider.json") as f:
        spider_data = json.load(f)
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

# Isolate the natural language questions (Independent Variable)
questions = [item['question'] for item in spider_data]

# Generate basic difficulty labels based on SQL complexity heuristics
labels = [get_difficulty(item['query']) for item in spider_data]
# Generate embeddings using the production model [1, 3]
# This transforms each sentence into a list of 384 numbers
embeddings = model.encode(questions, show_progress_bar=True, normalize_embeddings=True)
# Create the Data Matrix X [2]
X = np.array(embeddings)

# Create the Label Vector y
y = np.array(labels)

# Check the dimensions: Expected (n_questions, 384) [1]
print(f"Matrix X Shape: {X.shape}") 
print(f"Vector y Shape: {y.shape}")
print(f"Label Distribution: {pd.Series(y).value_counts().to_dict()}")
# Use pandas to find the frequency of each complexity level [2]
df_stats = pd.Series(labels).value_counts()
print("Question Complexity Distribution:")
print(df_stats)
# Geometric Validation: Verify Unit Vectors
# Check if the generated vectors are normalized (magnitude ≈ 1)
magnitudes = np.linalg.norm(X, axis=1)
is_normalized = np.allclose(magnitudes, 1.0, atol=1e-3)
print(f"Geometric Validation - Are all vectors unit length? {is_normalized}")

# Save X and y as a compressed file for the next phase
np.savez('spider_embeddings.npz', X=X, y=y)
print("Phase 1 Complete: Matrix and labels saved.")
