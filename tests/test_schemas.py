import pytest
from pydantic import ValidationError

from review_analyzer.schemas import GeneratedReview, Movie, MovieContext, PersonalReviewStyle


def test_movie_schema_valid_data():
    movie = Movie(
        id="inception-2010",
        title="Inception",
        year=2010,
        genres="Action,Sci-Fi",
        runtime=148,
        context="Inception Action,Sci-Fi",
    )
    assert movie.id == "inception-2010"
    assert movie.title == "Inception"
    assert movie.year == 2010
    assert movie.genres == "Action,Sci-Fi"
    assert movie.runtime == 148
    assert movie.context == "Inception Action,Sci-Fi"


def test_movie_schema_missing_field():
    with pytest.raises(ValidationError):
        Movie(title="Inception", year=2010, genres="Action,Sci-Fi", runtime=148)


def test_movie_schema_invalid_data_type():
    with pytest.raises(ValidationError):
        Movie(id="inception-2010", title="Inception", year="2010", genres="Action,Sci-Fi", runtime=148)


def test_movie_context_valid_data():
    context = MovieContext(title="Inception", year=2010, genres=["Action", "Sci-Fi"], runtime=148)
    assert context.title == "Inception"
    assert context.year == 2010
    assert context.genres == ["Action", "Sci-Fi"]
    assert context.runtime == 148


def test_movie_context_missing_field():
    with pytest.raises(ValidationError):
        MovieContext(title="Inception", year=2010, genres=["Action", "Sci-Fi"])


def test_movie_context_invalid_data_type():
    with pytest.raises(ValidationError):
        MovieContext(title="Inception", year="2010", genres="Action, Sci-Fi", runtime=148)


def test_generated_review_valid_data():
    review = GeneratedReview(
        text="Great movie with stunning visuals.", style_confidence=0.95, key_elements_used=["visuals", "story"]
    )
    assert review.text == "Great movie with stunning visuals."
    assert review.style_confidence == 0.95
    assert review.key_elements_used == ["visuals", "story"]


def test_generated_review_missing_field():
    with pytest.raises(ValidationError):
        GeneratedReview(text="Great movie with stunning visuals.", style_confidence=0.95)


def test_generated_review_invalid_data_type():
    with pytest.raises(ValidationError):
        GeneratedReview(
            text="Great movie with stunning visuals.", style_confidence="high", key_elements_used=["visuals", "story"]
        )


def test_personal_review_style_valid_data():
    style = PersonalReviewStyle(
        sentence_patterns=[{"type": "opening", "pattern": "Starts with a quote"}],
        average_length=100,
        sentiment_scores={"positive": 0.5, "negative": 0.3, "neutral": 0.2},
        common_references=["Inception", "The Matrix"],
    )
    assert style.sentence_patterns == [{"type": "opening", "pattern": "Starts with a quote"}]
    assert style.average_length == 100
    assert style.sentiment_scores == {"positive": 0.5, "negative": 0.3, "neutral": 0.2}
    assert style.common_references == ["Inception", "The Matrix"]


def test_personal_review_style_missing_field():
    with pytest.raises(ValidationError):
        PersonalReviewStyle(
            sentence_patterns=[{"type": "opening", "pattern": "Starts with a quote"}],
            average_length=100,
            sentiment_scores={"positive": 0.5, "negative": 0.3, "neutral": 0.2},
        )


def test_personal_review_style_invalid_data_type():
    with pytest.raises(ValidationError):
        PersonalReviewStyle(
            sentence_patterns="invalid type",
            average_length="100",
            sentiment_scores={"positive": 0.5, "negative": 0.3, "neutral": 0.2},
            common_references=["Inception", "The Matrix"],
        )
