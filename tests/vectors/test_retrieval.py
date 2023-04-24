import numpy as np
from numpy.testing import assert_array_equal
from scipy.spatial.transform import Rotation as R

from outlines.vectors.retrieval import cosine_similarity


def test_cosine_similarity():
    query = np.ones(3)
    vectors = [
        R.from_rotvec([0, 0, np.pi / 3]).apply(query),
        query,
        R.from_rotvec([0, 0, np.pi / 4]).apply(query),
        R.from_rotvec([0, 0, np.pi / 5]).apply(query),
        R.from_rotvec([0, 0, np.pi / 6]).apply(query),
    ]

    result_idx = cosine_similarity(vectors, query)
    assert_array_equal(result_idx[0], 1)

    results_idx = cosine_similarity(vectors, query, k=3)
    assert_array_equal(results_idx[0], 1)
    assert_array_equal(results_idx[1], 4)
    assert_array_equal(results_idx[2], 3)
