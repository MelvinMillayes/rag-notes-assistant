
from tools import search_notes, write_note
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from generate import ask

available_tools = [search_notes, write_note]
llm = ChatOllama(model="llama3.2").bind_tools(available_tools)


SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You can call search_notes multiple times if the question covers more than one topic. "
        "For a comparison question, search for each topic separately before answering. "
        "IMPORTANT: Never call write_note in the same turn as search_notes. "
        "Only call write_note after you have already seen the search results and know what to write."
    )
}

def is_system_message(m):
    if isinstance(m, dict):
        return m.get("role") == "system"
    return getattr(m, "type", None) == "system"

def call_model(state: MessagesState):
    messages = state["messages"]

    print(f"\n--- call_model: {len(messages)} messages in state ---")
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
        else:
            role = getattr(m, "type", type(m).__name__)
            content = getattr(m, "content", None)
        print(f"  [{role}] {str(content)[:80]}")

    response = llm.invoke(messages)

    print(f"  → model responded. tool_calls: {response.tool_calls}")
    print(f"  → content: {response.content[:200]!r}")

    return {"messages": [response]}

def should_continue(state: MessagesState):
    last_message= state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

llm = ChatOllama(model="llama3.2").bind_tools(available_tools)
llm_no_tools = ChatOllama(model="llama3.2")  # same model, no tools bound

def finalize_answer(state: MessagesState):
    messages = state["messages"]
    grounding_instruction = {
        "role": "system",
        "content": (
            "Write your final answer using ONLY the information returned by the search_notes tool "
            "in this conversation. Do not add facts from your own general knowledge. "
            "If the retrieved notes don't fully answer the question, say so explicitly rather than filling the gap yourself."
        )
    }
    response = llm_no_tools.invoke(messages + [grounding_instruction])
    print(f"--- finalize ran. content: {response.content[:200]!r} ---")
    return {"messages": [response]}

graph = StateGraph(MessagesState)

graph.add_node("agent", call_model)
graph.add_node("finalize", finalize_answer)
graph.add_node("tools", ToolNode(available_tools))
graph.set_entry_point("agent")

graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: "finalize"})
graph.add_edge("tools", "agent") 
graph.add_edge("finalize", END)

app = graph.compile()



if __name__ == "__main__":
    result = app.invoke({"messages": [{"role": "user", "content": "what is 1+2?"}]})
    print(result["messages"][-1].content)

    result = app.invoke({"messages": [SYSTEM_PROMPT, {"role": "user", "content": "Check what my notes say about use cases for langchain, then do the same for langgraph. Take what you find and add it to the differences.txt file."}]})
    print(result["messages"][-1].content)
