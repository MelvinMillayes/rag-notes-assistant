# tools.py
from langchain_core.tools import tool
from query import retrieve
import os


@tool
def search_notes(query: str) -> str:
    """Search the user's personal notes for relevant information. Use this when the question is about the user's own notes, projects, or personal history."""

    results = retrieve(query)
    chunks = results["documents"][0]
    return "\n\n".join(chunks)
 
@tool
def write_note(filename: str, content: str) -> str:
    """Create a new note or overwrite an existing one in the user's notes folder.
    Use this when the user asks you to save, write down, or record something as a note.
    filename should be a simple name like 'meeting_summary.txt' — no folders or special characters."""

    NOTES_DIR = "notes"

    # Guard 1: only allow .txt files
    if not filename.endswith(".txt"):
        filename += ".txt"
    # Guard 2: strip any path info the model might sneak in — keep just the filename
    filename = os.path.basename(filename)

    path = os.path.join(NOTES_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Saved note to {filename} ({len(content)} characters)."