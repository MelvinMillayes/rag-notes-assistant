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

## Example

```
Your question: Is Python an object-oriented language?

Answer: Yes. Python supports multiple programming paradigms, with an
emphasis on object-oriented programming alongside dynamic typing.
```
