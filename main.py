from generate import ask

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

        answer = ask(question)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()