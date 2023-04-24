import numpy as np

from outlines.vectors import VectorStore


def test_vector_store():
    def dummy_embedding_model(query: str):
        """We compute a simplistic embedding by converting characters to an int."""
        return np.array([ord(c) for c in query])

    store = VectorStore(dummy_embedding_model)

    store.insert("Test1")
    store.insert("Test2")
    assert len(store.storage) == 2

    result = store.query("Test1")
    assert result[0] == "Test1"

    result = store.query("Test2")
    assert result[0] == "Test2"
