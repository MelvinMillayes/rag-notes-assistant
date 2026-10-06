from generate import ask
from agent import app
from agent import app, SYSTEM_PROMPT

def main():
    print("RAG Notes Assistant — ask a question about your notes.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not question:
            continue
 
        result = app.invoke({"messages": [SYSTEM_PROMPT, {"role": "user", "content": question}]})
        answer = result["messages"][-1].content
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":  
    main()