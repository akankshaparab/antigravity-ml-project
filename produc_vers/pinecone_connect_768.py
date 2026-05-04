from pinecone import Pinecone

# Use the key you just copied from your own dashboard
pc = Pinecone(api_key="pcsk_4Hj5Sb_4ChZ2R6ooBPbcMfGgXL4mUQNQV4QWD9kMjVF3ggD1gHFMkpVN6UTM356TZSJEao")

# Use the host exactly as it appeared in your image
index = pc.Index(host="data-analyst-sandbox-04ia07m.svc.aped-4627-b74a.pinecone.io")

# Verify connection by printing index stats
print("Successfully connected to Pinecone!")
print("Index Stats:", index.describe_index_stats())