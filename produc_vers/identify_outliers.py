import numpy as np
from pinecone import Pinecone
import sys

# Set encoding to UTF-8 for the terminal to handle emojis/special chars
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Connect
pc = Pinecone(api_key="pcsk_4Hj5Sb_4ChZ2R6ooBPbcMfGgXL4mUQNQV4QWD9kMjVF3ggD1gHFMkpVN6UTM356TZSJEao")
index = pc.Index(host="data-analyst-sandbox-04ia07m.svc.aped-4627-b74a.pinecone.io")

# 2. Fetch all 1,742 records
print("Fetching live data to identify outliers (UTF-8 safe)...")
dummy_vector = [0.0] * 768
query_response = index.query(vector=dummy_vector, top_k=1742, include_metadata=True)

# 3. Print the top outliers
print("\n--- Potential Outlier Queries (Check these for the 'Gaps') ---")
for i, match in enumerate(query_response['matches'][:10]):
    print(f"{i+1}. ID: {match['id']}")
    if 'metadata' in match:
        # Use repr() to safely show metadata even with weird characters
        print(f"   Metadata: {repr(match['metadata'])}")
    print("-" * 20)
