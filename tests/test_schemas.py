import pandas as pd
import pytest
from pydantic import ValidationError

from src.review_analyzer.schemas import (
    GeneratedReview,
    Movie,
    MovieContext,
    PersonalReviewStyle,
    _get_era_description,
    _get_runtime_category,
)


def test_movie_schema_valid_data():
    movie = Movie(
        id='inception-2010',
        title='Inception',
        year=2010,
        genres='Action,Sci-Fi',
        runtime=148,
        context='Inception Action,Sci-Fi 2010s modern film directors_cut',
    )
    assert movie.id == 'inception-2010'
    assert movie.title == 'Inception'
    assert movie.year == 2010
    assert movie.genres == 'Action,Sci-Fi'
    assert movie.runtime == 148
    assert movie.context == 'Inception Action,Sci-Fi 2010s modern film directors_cut'


def test_movie_schema_missing_field():
    with pytest.raises(ValidationError):
        Movie(title='Inception', year=2010, genres='Action,Sci-Fi', runtime=148)


def test_movie_schema_invalid_data_type():
    with pytest.raises(ValidationError):
        Movie(id='inception-2010', title='Inception', year='2010', genres='Action,Sci-Fi', runtime=148)


def test_movie_context_valid_data():
    context = MovieContext(title='Inception', year=2010, genres=['Action', 'Sci-Fi'], runtime=148)
    assert context.title == 'Inception'
    assert context.year == 2010
    assert context.genres == ['Action', 'Sci-Fi']
    assert context.runtime == 148


def test_movie_context_missing_field():
    with pytest.raises(ValidationError):
        MovieContext(title='Inception', year=2010, genres=['Action', 'Sci-Fi'])


def test_movie_context_invalid_data_type():
    with pytest.raises(ValidationError):
        MovieContext(title='Inception', year='2010', genres='Action, Sci-Fi', runtime=148)


def test_movie_context_embedding():
    context = MovieContext(title='Inception', year=2010, genres=['Action', 'Sci-Fi'], runtime=148)
    embedding_context = context.get_embedding_context()
    assert embedding_context == 'Inception Action Sci-Fi 2010s modern film directors_cut'


def test_generated_review_valid_data():
    review = GeneratedReview(
        text='Great movie with stunning visuals.',
        style_confidence={'length': 0.95, 'opening': 0.8, 'transition': 0.7, 'closing': 0.9, 'comparative': 0.85},
        key_elements_used=['visuals', 'story'],
    )

    assert review.text == 'Great movie with stunning visuals.'
    assert isinstance(review.style_confidence, dict)
    assert all(0 <= score <= 1 for score in review.style_confidence.values())
    assert review.key_elements_used == ['visuals', 'story']


def test_generated_review_missing_field():
    with pytest.raises(ValidationError):
        GeneratedReview(text='Great movie with stunning visuals.', style_confidence=0.95)


def test_generated_review_invalid_data_type():
    with pytest.raises(ValidationError):
        GeneratedReview(
            text='Great movie with stunning visuals.', style_confidence='high', key_elements_used=['visuals', 'story']
        )


def test_personal_review_style_valid_data():
    style = PersonalReviewStyle(
        sentence_patterns=[{'type': 'opening', 'pattern': 'Starts with a quote'}],
        average_length=100,
        sentiment_scores={'positive': 0.5, 'negative': 0.3, 'neutral': 0.2},
        common_references=['Inception', 'The Matrix'],
    )
    assert style.sentence_patterns == [{'type': 'opening', 'pattern': 'Starts with a quote'}]
    assert style.average_length == 100
    assert style.sentiment_scores == {'positive': 0.5, 'negative': 0.3, 'neutral': 0.2}
    assert style.common_references == ['Inception', 'The Matrix']


def test_personal_review_style_missing_field():
    with pytest.raises(ValidationError):
        PersonalReviewStyle(
            sentence_patterns=[{'type': 'opening', 'pattern': 'Starts with a quote'}],
            average_length=100,
            sentiment_scores={'positive': 0.5, 'negative': 0.3, 'neutral': 0.2},
        )


def test_personal_review_style_invalid_data_type():
    with pytest.raises(ValidationError):
        PersonalReviewStyle(
            sentence_patterns='invalid type',
            average_length='100',
            sentiment_scores={'positive': 0.5, 'negative': 0.3, 'neutral': 0.2},
            common_references=['Inception', 'The Matrix'],
        )


@pytest.mark.parametrize(
    'year,expected',
    [
        (2023, '2020s contemporary film'),
        (2020, '2020s contemporary film'),
        (2019, '2010s modern film'),
        (2010, '2010s modern film'),
        (2009, '2000s film'),
        (2000, '2000s film'),
        (1990, '1990s film'),
        (1980, '1980s film'),
        (1970, '1970s film'),
        (1960, '1960s film'),
        (1959, 'pre-1960 classic film'),
        # Edge cases
        (0, 'pre-1960 classic film'),
        (1900, 'pre-1960 classic film'),
        (2050, '2020s contemporary film'),
    ],
)
def test_get_era_description(year, expected):
    assert _get_era_description(year) == expected


@pytest.mark.parametrize(
    'minutes,expected',
    [
        (30, 'short_film'),
        (39, 'short_film'),
        (40, 'featurette'),
        (79, 'featurette'),
        (80, 'theatrical_film'),
        (119, 'theatrical_film'),
        (120, 'directors_cut'),
        (159, 'directors_cut'),
        (160, 'cinematic_epic'),
        (200, 'cinematic_epic'),
        # Edge cases
        (0, 'short_film'),
        (1, 'short_film'),
        (500, 'cinematic_epic'),
    ],
)
def test_get_runtime_category(minutes, expected):
    assert _get_runtime_category(minutes) == expected


def test_movie_from_row():
    row = pd.Series({'Name': 'Inception', 'Year': 2010, 'genres': 'Action,Sci-Fi', 'runtimeMinutes': 148})
    movie = Movie.from_row(row, 'inception-2010')

    assert movie.id == 'inception-2010'
    assert movie.title == 'Inception'
    assert movie.year == 2010
    assert movie.genres == 'Action,Sci-Fi'
    assert movie.runtime == 148
    assert movie.context == 'Inception Action,Sci-Fi 2010s modern film directors_cut'


def test_movie_metadata():
    movie = Movie(
        id='inception-2010',
        title='Inception',
        year=2010,
        genres='Action,Sci-Fi',
        runtime=148,
        context='Inception Action,Sci-Fi 2010s modern film directors_cut',
    )
    metadata = movie.to_metadata()

    assert metadata == {
        'id': 'inception-2010',
        'title': 'Inception',
        'year': 2010,
        'genres': 'Action,Sci-Fi',
        'runtime': 148,
        'era': '2010s modern film',
        'length_category': 'directors_cut',
    }
