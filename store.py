import chromadb
from chunk import chunk_text
from embed import get_embeddings

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("notes")

def store_file(path, filename):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": filename} for _ in chunks]
    )
    print(f"Stored {len(chunks)} chunks from {filename}")

if __name__ == "__main__":
    import os
    for filename in os.listdir("notes"):
        store_file(f"notes/{filename}", filename)