import chromadb 
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "code_embs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

collection = client.get_collection(COLLECTION_NAME)

query_results = collection.query(
    query_texts=["how does the project verify a JWT token from an incoming request?"],
    n_results=3,
)

for query in query_results["documents"][0]:
    print(query)
    print()
    print()

print(query_results["distances"][0][::-1])
print(query_results["metadatas"])
print(query_results.keys())