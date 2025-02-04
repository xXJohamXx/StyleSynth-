from typing import Dict, List

import pandas as pd
from pydantic import BaseModel


def _get_era_description(year: int) -> str:
    """Convert year to meaningful era description."""
    if year >= 2020:
        return '2020s contemporary film'
    elif year >= 2010:
        return '2010s modern film'
    elif year >= 2000:
        return '2000s film'
    elif year >= 1990:
        return '1990s film'
    elif year >= 1980:
        return '1980s film'
    elif year >= 1970:
        return '1970s film'
    elif year >= 1960:
        return '1960s film'
    return 'pre-1960 classic film'


def _get_runtime_category(minutes: int) -> str:
    """Convert runtime to meaningful length description."""
    if minutes < 40:
        return 'short_film'
    elif minutes < 80:
        return 'featurette'
    elif minutes < 120:
        return 'theatrical_film'
    elif minutes < 160:
        return 'directors_cut'
    return 'cinematic_epic'


class Movie(BaseModel):
    """Internal class for processing movies during analysis"""

    id: str
    title: str
    year: int
    genres: str
    runtime: int
    context: str

    @classmethod
    def from_row(cls, row: pd.Series, movie_id: str) -> 'Movie':
        year = row.get('Year', 0)
        runtime = row.get('runtimeMinutes', 0)
        era = _get_era_description(year)
        length_category = _get_runtime_category(runtime)

        context = f"{row.get('Name', '')} {row.get('genres', '')} {era} {length_category}"

        return cls(
            id=movie_id,
            title=row.get('Name', ''),
            year=year,
            genres=row.get('genres', ''),
            runtime=runtime,
            context=context,
        )

    def to_metadata(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'year': self.year,
            'genres': self.genres,
            'runtime': self.runtime,
            'era': _get_era_description(self.year),
            'length_category': _get_runtime_category(self.runtime),
        }


class PersonalReviewStyle(BaseModel):
    sentence_patterns: List[Dict[str, str]]
    average_length: int
    sentiment_scores: Dict[str, float]
    common_references: List[str]


class MovieContext(BaseModel):
    """External class for API interface"""

    title: str
    year: int
    genres: List[str]
    runtime: int

    def get_embedding_context(self) -> str:
        era = _get_era_description(self.year)
        length_category = _get_runtime_category(self.runtime)
        return f"{self.title} {' '.join(self.genres)} {era} {length_category}"


class GeneratedReview(BaseModel):
    text: str
    style_confidence: Dict[str, float]
    key_elements_used: List[str]
