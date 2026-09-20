def chunk_text(text, size = 300, overlap = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start : end])#adds a text chunk from star to end but not including end
        start += size - overlap
    return chunks

if __name__ == "__main__":
    with open("notes/python.txt", "r", encoding="utf-8") as f:
        text = f.read()
        chunks = chunk_text(text)
        
        
        print(f"Split into {len(chunks)} chunks.\n")
        print("First chunk:")
        print(chunks[0])