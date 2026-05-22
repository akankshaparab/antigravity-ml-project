from pinecone import Pinecone
import json

PINECONE_API_KEY = "pcsk_4Hj5Sb_4ChZ2R6ooBPbcMfGgXL4mUQNQV4QWD9kMjVF3ggD1gHFMkpVN6UTM356TZSJEao"
SOURCE_INDEX_HOST = "data-analyst-sandbox-04ia07m.svc.aped-4627-b74a.pinecone.io"

pc = Pinecone(api_key=PINECONE_API_KEY)
source_index = pc.Index(host=SOURCE_INDEX_HOST)

print("Fetching 2000 records from Pinecone...")
res = source_index.query(
    vector=[0.1] * 768,
    top_k=2000,
    include_values=False,
    include_metadata=True
)

metadata_groups = {}
for m in res.matches:
    keys = tuple(sorted(m.metadata.keys())) if m.metadata else ()
    metadata_groups[keys] = metadata_groups.get(keys, 0) + 1

print("\nMetadata patterns found:")
for keys, count in metadata_groups.items():
    print(f"Keys: {keys} -> Count: {count}")
