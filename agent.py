# =============================================================================
# LangGraph agent for the RAG Notes Assistant (commented for learning)
#
# Big picture: this program is a small STATE MACHINE (a "graph").
#   - Each box in the graph is a NODE (a plain Python function).
#   - Arrows between boxes are EDGES (rules for what runs next).
#   - A shared STATE (the list of chat messages) is passed from node to node.
#
# The flow this file builds:
#
#        +-------+  has tool calls?   +-------+
#  ----> | agent | -----------------> | tools |
#        +-------+ <----------------- +-------+
#            |        (results go back to the model)
#            | no tool calls
#            v
#        +----------+
#        | finalize | ---> END
#        +----------+
#
# C#/.NET analogy: think of it like a workflow engine / state machine
# (e.g., Stateless or Windows Workflow). Each node is a method; the "state"
# is an object passed through; edges are the transition rules.
# =============================================================================



# Your own functions from tools.py. These are the "tools" the model may request.
from tools import search_notes, write_note

# StateGraph      -> the builder for the graph (nodes + edges)
# MessagesState   -> a ready-made state type: a dict with one key, "messages",
#                    holding the conversation as a list
# END             -> a special marker meaning "stop the graph here"
from langgraph.graph import StateGraph, MessagesState, END

# ToolNode -> a prebuilt node that EXECUTES the tool calls the model asked for.
# (Remember: the model never runs code. It only asks. ToolNode runs it.)
from langgraph.prebuilt import ToolNode

# ChatOllama -> LangChain's wrapper around a local model served by Ollama.
from langchain_ollama import ChatOllama


# --- Model setup -------------------------------------------------------------

# A plain list of the functions the model is allowed to request.
available_tools = [search_notes, write_note]

# Create the chat model, then call .bind_tools(...).
# bind_tools sends each tool's name, description and parameter schema to the
# model with every request, so the model can answer with a structured
# "please call search_notes(query=...)" instead of plain text.
# (Where do the descriptions come from? Usually the function's docstring and
# type hints in tools.py. That's why good docstrings matter for tool calling.)
llm = ChatOllama(model="llama3.2").bind_tools(available_tools)


# --- System prompt -----------------------------------------------------------

# A message dict with role "system". It sets rules for the model's behavior.
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        # Python joins adjacent string literals into one string, so these
        # parenthesized pieces become one long prompt.
        "You can call search_notes multiple times if the question covers more than one topic. "
        "For a comparison question, search for each topic separately before answering. "
        "IMPORTANT: Never call write_note in the same turn as search_notes. "
        "Only call write_note after you have already seen the search results and know what to write."
    )
}


# --- Helper (currently unused) -----------------------------------------------

def is_system_message(m):
    """Return True if message `m` is a system message.

    Messages can be plain dicts ({"role": ...}) or LangChain message objects
    (which have a .type attribute). This handles both shapes.
    NOTE: nothing in this file calls this function right now (dead code).
    """
    if isinstance(m, dict):                 # isinstance ~ C# `is` type check
        return m.get("role") == "system"    # .get returns None if key is missing
    # getattr(obj, "name", default) reads an attribute by name, safely.
    return getattr(m, "type", None) == "system"


# --- Node 1: "agent" (calls the model) ---------------------------------------

def call_model(state: MessagesState):
    """Send the whole conversation to the model and return its reply.

    A node receives the current state and returns a PARTIAL update to it.
    """
    messages = state["messages"]   # the list of messages so far

    # ---- Debug printing: shows what the model is about to see ----
    print(f"\n--- call_model: {len(messages)} messages in state ---")
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
        else:
            # LangChain message object: .type is "human", "ai", "tool", "system"
            role = getattr(m, "type", type(m).__name__)
            content = getattr(m, "content", None)
        # str(content)[:80] -> first 80 characters (slicing), to keep logs short
        print(f"  [{role}] {str(content)[:80]}")

    # ---- The real work: one model call ----
    # .invoke() sends the messages and waits for the full response.
    # The result is an AIMessage that has:
    #   .content    -> text (may be empty if the model only wants a tool call)
    #   .tool_calls -> list of requested tool calls (empty list if none)
    response = llm.invoke(messages)

    response.tool_calls = strip_write_when_searching(response.tool_calls)
    print(f"  → model responded. tool_calls: {response.tool_calls}")
    # !r prints the repr (with quotes), so empty strings are visible in logs
    print(f"  → content: {response.content[:200]!r}")

    # IMPORTANT: return only the NEW message. MessagesState uses a "reducer"
    # (add_messages) that APPENDS it to the list instead of replacing the list.
    return {"messages": [response]}


# --- Routing function (an edge decision) -------------------------------------

def should_continue(state: MessagesState):
    """Decide where to go after the agent node.

    Returns the NAME of the next step (a key in the mapping used below).
    """
    last_message = state["messages"][-1]   # [-1] = last item in a list
    if last_message.tool_calls:            # non-empty list counts as True
        return "tools"                     # model asked for a tool -> run it
    return END                             # no tool request -> we're done looping

def strip_write_when_searching(tool_calls):
    # Set comprehension: collect the tool names requested in this reply.
    # (C# analogy: tool_calls.Select(tc => tc.Name).ToHashSet())
    names = {tc["name"] for tc in tool_calls}

    if "search_notes" in names:
        # List comprehension with a filter.
        # (C# analogy: tool_calls.Where(tc => tc.Name != "write_note").ToList())
        return [tc for tc in tool_calls if tc["name"] != "write_note"]

    return tool_calls   # no search in this reply, so nothing to strip
# --- Node 3: "finalize" (writes the grounded final answer) -------------------

# The SAME model, but WITHOUT tools bound. This one can only write text,
# so it can't start another round of tool calls.
llm_no_tools = ChatOllama(model="llama3.2")


def finalize_answer(state: MessagesState):
    """Rewrite the final answer so it relies only on retrieved notes.

    This is a "grounding" step: a guard against the model filling gaps with
    its own general knowledge (hallucination) in a RAG app.
    """
    messages = state["messages"]

    # An extra instruction added at the very end so it's the freshest thing
    # the model reads before answering.
    grounding_instruction = {
        "role": "system",
        "content": (
            "Write your final answer using ONLY the information returned by the search_notes tool "
            "in this conversation. Do not add facts from your own general knowledge. "
            "If the retrieved notes don't fully answer the question, say so explicitly rather than filling the gap yourself."
        )
    }

    # messages + [x] builds a NEW list (original state is not modified).
    response = llm_no_tools.invoke(messages + [grounding_instruction])

    print(f"--- finalize ran. content: {response.content[:200]!r} ---")
    return {"messages": [response]}


# --- Build the graph ---------------------------------------------------------

# Create the graph, telling it the state shape is MessagesState.
graph = StateGraph(MessagesState)

# Register nodes: add_node("name", function_or_runnable)
graph.add_node("agent", call_model)                  # asks the model
graph.add_node("finalize", finalize_answer)          # writes final answer
graph.add_node("tools", ToolNode(available_tools))   # runs requested tools

# Where execution starts.
graph.set_entry_point("agent")

# Conditional edge: after "agent", run should_continue() and use its return
# value to look up the next node in this mapping:
#   "tools" -> go to the "tools" node
#   END     -> go to "finalize" (instead of stopping immediately)
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: "finalize"})

# Fixed edges (always happen):
graph.add_edge("tools", "agent")   # after tools run, go back to the model
                                   # with the tool results (this is the LOOP)
graph.add_edge("finalize", END)    # after finalize, stop

# compile() validates the graph and returns a runnable app.
# (Roughly like building a pipeline/host before running it in .NET.)
app = graph.compile()


# --- Demo / entry point ------------------------------------------------------

# Python's equivalent of Main(): this block only runs when you execute this
# file directly (python agent.py), NOT when another file imports it.
if __name__ == "__main__":
    # Test 1: simple question, no system prompt.
    # invoke() runs the graph from the entry point to END and returns the final state.
    result = app.invoke({"messages": [{"role": "user", "content": "what is 1+2?"}]})
    # result["messages"][-1] is the last message (the finalize answer).
    print(result["messages"][-1].content)

    # Test 2: multi-step task. Expected flow:
    #   agent -> tools (search langchain) -> agent -> tools (search langgraph)
    #   -> agent -> tools (write_note) -> agent -> finalize -> END
    result = app.invoke({"messages": [
        SYSTEM_PROMPT,
        {"role": "user", "content": "Check what my notes say about use cases for langchain, then do the same for langgraph. Take what you find and add it to the differences.txt file."}
    ]})
    print(result["messages"][-1].content)