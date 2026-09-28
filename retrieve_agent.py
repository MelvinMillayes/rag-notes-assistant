import ollama

# Multi-line strings require triple quotes (""" or ''')
context = """You are an agent whose job is to decide whether I need to search my notes for the answer, or can answer directly. 
You must respond with exactly one line: either SEARCH: QUERY or ANSWER: <your answer>. Use SEARCH when the question is about the user's own notes, projects, or personal information. Use ANSWER for general knowledge or simple math."""


def retrival_agent(question):
    prompt = f"Context: {context}"
    
    response = ollama.chat(
        model='llama3.2', 
        messages=[
            {
                'role': 'system', 
                'content': prompt
            },
            {
                "role": "user",
                "content": question
            },

        ]
    )
    
    decision = response['message']['content'].strip()
    return decision

if __name__ == "__main__":
    question = "what is 1+2?"
    answer = retrival_agent(question)
    print(answer)
