import chromadb
from embed import get_embedding

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("notes")

def retrieve(question, n_results=3):
    question_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    question = "Is python an object oriented language?"
    results = retrieve(question)

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        print(f"[{meta['source']}] (distance: {dist:.4f})")
        print(doc[:150])
        print("---")