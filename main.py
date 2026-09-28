from generate import ask
from retrieve_agent import retrival_agent

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
 
        
        retrieve = retrival_agent(question)   

        decision, _, payload = retrieve.partition(":")
        decision = decision.strip().upper()
        payload = payload.strip()

        match decision:
            case "ANSWER":
                print(retrieve)
            case "SEARCH" :
                answer = ask(question)
                print(f"\nAnswer: {answer}\n")
            case _:
                print(f"Unexpected format from model: {retrieve!r}")

if __name__ == "__main__":
    main()