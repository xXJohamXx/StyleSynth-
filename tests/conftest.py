import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from src.review_analyzer.schemas import PersonalReviewStyle
from src.review_analyzer.vector_store import VectorStore


@pytest.fixture
def test_sample_batch():
    return pd.DataFrame(
        {
            "Name": ["Inception", "The Matrix"],
            "Year": [2010, 1999],
            "genres": ["Action,Sci-Fi", "Action,Sci-Fi"],
            "runtimeMinutes": [148, 136],
        }
    )


@pytest.fixture
def test_style_profile():
    return PersonalReviewStyle(
        sentence_patterns=[{"type": "opening", "pattern": "Starts with a quote"}],
        average_length=100,
        sentiment_scores={"positive": 0.5, "negative": 0.3, "neutral": 0.2},
        common_references=["Inception", "The Matrix"],
    )


@pytest.fixture
def test_movie_data():
    unique_id = str(uuid.uuid4())  # Generate a unique ID
    return {
        "title": "Inception",
        "metadata": {
            "id": f"inception-{unique_id}",
            "title": "Inception",
            "year": 2010,
            "genres": "Action,Sci-Fi",
            "runtime": 148,
        },
        "embedding": [0.1, 0.2, 0.3],
    }


@pytest.fixture(scope="session")
def test_vector_store():
    persist_dir = Path("./.test_vectordb")
    store = VectorStore(persist_dir=str(persist_dir))

    yield store

    try:
        shutil.rmtree(persist_dir)
    except Exception as e:
        print(f"Error cleaning up directory: {e}")
