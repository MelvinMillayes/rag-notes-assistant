from query import retrieve

def test_python_question_returns_python_file():
    results = retrieve("Is Python an object-oriented language?", n_results=1)
    top_source = results["metadatas"][0][0]["source"]
    assert top_source == "python.txt"

def test_distance_is_reasonably_low_for_relevant_match():
    results = retrieve("Is Python an object-oriented language?", n_results=1)
    top_distance = results["distances"][0][0]
    assert top_distance < 1.0