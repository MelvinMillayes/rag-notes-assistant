# RAG Notes Assistant

A small Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about your own text notes and get answers grounded in that content, instead of the model's general training knowledge.

Built as a learning project to get hands-on with Python and the core concepts behind modern LLM platforms: chunking, embeddings, vector databases, retrieval, and generation.

## How it works

1. **Chunking** — text files in `notes/` are split into small, overlapping chunks so each piece stays focused on a single idea.
2. **Embedding** — each chunk is converted into a vector (384 numbers representing its meaning) using a local embedding model (`all-MiniLM-L6-v2`, via `sentence-transformers`).
3. **Storage** — chunks and their vectors are stored in [Chroma](https://www.trychroma.com/), a local vector database.
4. **Retrieval** — when you ask a question, it's embedded the same way, and Chroma finds the stored chunks whose vectors are closest to it (i.e., most similar in meaning).
5. **Generation** — the retrieved chunks and your question are sent to a local LLM (via [Ollama](https://ollama.com), running `llama3.2`), which generates an answer using only that retrieved context.

```
notes/*.txt → chunk → embed → store in Chroma
                                     │
your question → embed → retrieve closest chunks → LLM → answer
```

## Project structure

| File | Purpose |
|---|---|
| `chunk.py` | Splits text into overlapping chunks |
| `embed.py` | Turns text into vectors using a local embedding model |
| `store.py` | Embeds and stores all files in `notes/` into Chroma |
| `query.py` | Embeds a question and retrieves the closest matching chunks |
| `generate.py` | Retrieves chunks and asks a local LLM to answer using them |
| `main.py` | Interactive command-line loop to ask questions |
| `test_retrieve.py` | Pytest checks that retrieval returns the expected source file |
| `notes/` | Your own text files to search over |
| `agent.py` | LangGraph agent loop: lets the model decide whether to search notes, write notes, or answer directly, looping until it's done |
| `tools.py` | Defines the `search_notes` and `write_note` tools the agent can call |
## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

You'll also need [Ollama](https://ollama.com) installed, with a model pulled:

```bash
ollama pull llama3.2
```

## Usage

**1. Add your notes** — drop `.txt` files into the `notes/` folder.

**2. Store them in the vector database:**
```bash
python store.py
```

**3. Ask questions:**
```bash
python main.py
```

Type your question and press Enter. Type `quit` or `exit` to stop.

## Running tests

```bash
pytest
```

Checks that a question clearly about one topic actually retrieves chunks from the right source file, and that the best match's similarity distance is reasonably low.

## What I learned

This project was my first hands-on Python work, coming from a C#/.NET background. Building it end-to-end taught me:

- Python fundamentals: virtual environments, `pip`, functions, string slicing, list comprehensions, f-strings
- The core RAG pattern: how chunking, embeddings, vector search, and LLM generation fit together
- Debugging real environment issues on Windows (Python version mismatches, missing C++ runtime dependencies for PyTorch, and the importance of keeping a virtual environment activated)
- Writing basic automated tests with `pytest`
- Learned the core agent loop pattern (think → act → observe → repeat) and how LangGraph expresses it as a state graph: an `agent` node that calls the LLM, a `tools` node that executes whatever the model requests, and a conditional edge that loops back until the model stops requesting tools.
- Learned that tool calls can be issued in parallel within a single model response, which breaks sequential dependencies (e.g. a write that depends on a search result not yet available) — fixed with explicit ordering instructions in the system prompt.
- Learned that a model with tools bound to it can leak tool-call syntax into plain text, especially in a step that's meant to be text-only — fixed by using a separate, tool-free model instance for the final answer-writing step.
- Observed that a small local model (llama3.2) doesn't reliably stay grounded in retrieved note content even when explicitly instructed to use "only" the retrieved context — a known RAG/agent evaluation challenge, not yet solved in this project.
## Example

```
Your question: What are my notes about?

Answer:  Based on the context, it appears that the topic of your notes is not explicitly stated, but the text mentions the following related topics:

- Python programming
- Computer science
- Programming courses
- Curriculum for introductory computer science courses
- Python Enhancement Proposals
- Python style guide (PEP 8)
```
## Known Limitations

- **Grounding isn't fully reliable.** The `finalize` step is explicitly instructed to use only retrieved note content, but `llama3.2` sometimes blends in general knowledge from its own training rather than sticking strictly to what `search_notes` returned. This was observed when asking for a LangChain vs. LangGraph comparison — the written answer included claims not present in either source note. This is a known challenge in RAG systems with smaller local models, and would need a dedicated faithfulness/groundedness check (e.g., a second LLM pass verifying each claim traces back to retrieved content) to catch reliably.
- **Parallel tool calls can break sequential dependencies.** The model sometimes issues multiple tool calls in a single turn (e.g., two searches and a file write at once), before any results are available. This caused `write_note` to run with placeholder content instead of the actual search results. Mitigated with an explicit system-prompt instruction not to call `write_note` in the same turn as `search_notes`, but this relies on prompt compliance rather than a structural guarantee.
- **New notes aren't automatically searchable.** `write_note` saves a `.txt` file to `notes/`, but doesn't re-embed it into Chroma — `store.py` has to be re-run manually before a newly written note can be found by `search_notes`.
- **Tool-call syntax can leak into plain text.** Early in development, a model instance with tools bound to it (`.bind_tools(...)`) sometimes wrote a tool call out as a JSON-like string instead of using the structured tool-calling mechanism, particularly during the final answer-writing step. Fixed by giving that step a separate model instance with no tools bound.