from pinecone import Pinecone
import json

PINECONE_API_KEY = "pcsk_4Hj5Sb_4ChZ2R6ooBPbcMfGgXL4mUQNQV4QWD9kMjVF3ggD1gHFMkpVN6UTM356TZSJEao"
SOURCE_INDEX_HOST = "data-analyst-sandbox-04ia07m.svc.aped-4627-b74a.pinecone.io"

pc = Pinecone(api_key=PINECONE_API_KEY)
source_index = pc.Index(host=SOURCE_INDEX_HOST)

print("Fetching all 1742 live records from the Pinecone sandbox...")
res = source_index.query(
    vector=[0.1] * 768,
    top_k=1742,
    include_values=False,
    include_metadata=True
)

total = len(res.matches)
print(f"Total retrieved: {total}")

has_difficulty_count = 0
has_sql_count = 0
sql_and_difficulty_count = 0
unlabeled_sql_count = 0

sample_heuristics_distribution = {
    "Easy": 0,
    "Medium": 0,
    "Hard": 0,
    "Extra Hard": 0
}

def get_difficulty_heuristic(sql_query):
    query = sql_query.upper()
    score = 0
    if "JOIN" in query: score += 1
    if "GROUP BY" in query: score += 1
    if "ORDER BY" in query: score += 1
    if "HAVING" in query: score += 1
    if "INTERSECT" in query or "UNION" in query or "EXCEPT" in query: score += 2
    if query.count("SELECT") > 1: score += 2
    
    if score == 0: return "Easy", score
    elif score == 1: return "Medium", score
    elif score in [2, 3]: return "Hard", score
    else: return "Extra Hard", score

print("\nScanning records...")
for m in res.matches:
    metadata = m.metadata if m.metadata else {}
    has_diff = 'difficulty' in metadata
    
    # Try to extract SQL
    sql = None
    if 'args_json' in metadata:
        try:
            args = json.loads(metadata['args_json'])
            sql = args.get('sql')
        except Exception:
            pass
            
    if has_diff:
        has_difficulty_count += 1
    if sql:
        has_sql_count += 1
        
    if sql and has_diff:
        sql_and_difficulty_count += 1
        
    if sql and not has_diff:
        unlabeled_sql_count += 1
        diff_label, score = get_difficulty_heuristic(sql)
        sample_heuristics_distribution[diff_label] += 1

print("\n--- RESULTS ---")
print(f"Total records checked: {total}")
print(f"Records with 'difficulty' metadata key: {has_difficulty_count}")
print(f"Records with SQL query in metadata (under args_json): {has_sql_count}")
print(f"Records with BOTH SQL and 'difficulty': {sql_and_difficulty_count}")
print(f"Records with SQL but NO 'difficulty': {unlabeled_sql_count}")

if unlabeled_sql_count > 0:
    print("\nHeuristic Difficulty distribution for the unlabeled live SQL queries:")
    for label, count in sample_heuristics_distribution.items():
        print(f"  - {label}: {count} ({count/unlabeled_sql_count:.1%})")
