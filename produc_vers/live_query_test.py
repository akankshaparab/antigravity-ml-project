from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import sys

# 1. Setup
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Initializing Final Router (768D)...")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
pc = Pinecone(api_key="pcsk_4Hj5Sb_4ChZ2R6ooBPbcMfGgXL4mUQNQV4QWD9kMjVF3ggD1gHFMkpVN6UTM356TZSJEao")
index = pc.Index(host="data-analyst-sandbox-04ia07m.svc.aped-4627-b74a.pinecone.io")

def test_query(user_query):
    print(f"\nQuery: '{user_query}'")
    vec = model.encode([user_query], normalize_embeddings=True)[0].tolist()
    
    # Search for top 10 (to give us room to filter manually)
    results = index.query(
        vector=vec, 
        top_k=10, 
        include_metadata=True,
        filter={"difficulty": {"$in": ["Easy", "Medium", "Hard", "Extra Hard"]}}
    )
    
    # Manually keep only valid matches
    valid_matches = [m for m in results['matches'] if m['metadata'].get('difficulty') in ["Easy", "Medium", "Hard", "Extra Hard"]]
    
    if valid_matches:
        print("\n--- Top 3 Labeled Matches ---")
        for i, match in enumerate(valid_matches[:3]):
            diff = match['metadata'].get('difficulty')
            print(f"{i+1}. Score: {match['score']:.2%} | Difficulty: {diff}")
        
        best_diff = valid_matches[0]['metadata'].get('difficulty')
        target = "GEMINI FLASH" if best_diff in ["Easy", "Medium"] else "GEMINI PRO"
        print(f"\nDecision: Route to {target} (Based on match: {best_diff})")
    else:
        print("Result: No labeled matches found. Defaulting to GEMINI PRO for safety.")

print("\n--- [System Ready] ---")
while True:
    query = input("\nEnter a SQL question (or 'quit'): ")
    if query.lower() == 'quit': break
    test_query(query)
