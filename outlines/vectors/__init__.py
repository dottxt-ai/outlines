from collections import deque
from typing import Callable, Dict, List, Tuple

import numpy as np

from outlines.vectors.retrieval import cosine_similarity


class VectorStore:
    """Represents a vector store.

    Vector stores are used to store embeddings and, given a query, retrieve the
    closest entries to this query. This class provides a layer of abstraction
    on top of practical implementations and integrations.

    Attributes
    ----------
    embedding_model
        A function which returns an `numpy.ndarray` of floats when passed a string.
    retrieval_fn
        A function which returns the nearest vector to a given query vector in a list
        of vectors. Defaults to cosine similarity.
    storage
        A list of tuples where text and the corresponding embeddings are stored.

    """

    def __init__(
        self, embedding_model: Callable, retrieval_fn: Callable = cosine_similarity
    ):
        self.embedding_model = embedding_model
        self.retrieval_fn = retrieval_fn
        self.storage: List[Tuple[np.ndarray, str]] = []

    def query(self, query: str, k: int = 1) -> List[str]:
        """Find the store entries that are closest to the query.

        Parameters
        ----------
        query
            A string for which we want to find the closest matches in the store.
        k
            The number of closest matches to return.

        """
        query_embedding = self.embedding_model(query)
        top_k_indices = self.retrieval_fn(
            [elem[0] for elem in self.storage], query_embedding, k
        )
        return [self.storage[i][1] for i in top_k_indices]

    def insert(self, query: str) -> None:
        """Insert the query and its embedding vector in the store.

        Parameters
        ----------
        query
            The string to insert in the store.

        """
        query_embedding = self.embedding_model(query)
        self.storage.append((query_embedding, query))
