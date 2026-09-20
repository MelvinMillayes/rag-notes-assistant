import ollama
from query import retrieve

def ask(question):
    results = retrieve(question)
    chunks = results["documents"][0]

    context = "\n\n".join(chunks)

    prompt = f"""Using only the context below, answer the question. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}"""

    response = ollama.chat(model="llama3.2", messages=[
        {"role": "user", "content": prompt}
    ])

    return response["message"]["content"]

if __name__ == "__main__":
    question = "Is python an object oriented language?"
    answer = ask(question)
    print(answer)