import logging
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_almost_equal


async def test_store_movie_sunny_day(test_vector_store, test_movie_data):
    result = await test_vector_store.store_movie(
        test_movie_data["title"], test_movie_data["metadata"], test_movie_data["embedding"]
    )
    assert result


async def test_store_movie_rainy_day(test_vector_store, test_movie_data):
    result = await test_vector_store.store_movie(
        test_movie_data["title"], {"mock": "metadata"}, test_movie_data["embedding"]
    )
    assert isinstance(result, bool), "Result should be a boolean"
    assert result is False


async def test_get_movie_by_id(test_vector_store, test_movie_data):
    await test_vector_store.store_movie(
        test_movie_data["title"], test_movie_data["metadata"], test_movie_data["embedding"]
    )
    result = await test_vector_store.get_movie_by_id(test_movie_data["metadata"]["id"])

    assert result["id"] == test_movie_data["metadata"]["id"]
    assert result["document"] == "Inception"
    assert result["metadata"] == test_movie_data["metadata"]
    assert_array_almost_equal(
        np.array(result["embedding"]),
        np.array(test_movie_data["embedding"]),
    )


async def test_find_similar_movies(test_vector_store):
    query_embedding = [0.1, 0.2, 0.3]
    n_results = 1
    result = await test_vector_store.find_similar_movies(query_embedding, n_results)
    assert len(result) == n_results


async def test_store_movie_exception_handling_with_logging(test_vector_store, test_movie_data, caplog):
    with patch.object(test_vector_store.movies_collection, "add", side_effect=Exception("Mocked exception")):
        with caplog.at_level(logging.ERROR):
            result = await test_vector_store.store_movie(
                test_movie_data["title"], test_movie_data["metadata"], test_movie_data["embedding"]
            )
            assert result is False
            assert "Error storing movie Inception: Mocked exception" in caplog.text
