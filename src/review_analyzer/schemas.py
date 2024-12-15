from typing import Dict, List

import pandas as pd
from pydantic import BaseModel


class Movie(BaseModel):
    """Internal class for processing movies during analysis"""

    id: str
    title: str
    year: int
    genres: str
    runtime: int
    context: str

    @classmethod
    def from_row(cls, row: pd.Series, movie_id: str) -> "Movie":
        return cls(
            id=movie_id,
            title=row.get("Name", ""),
            year=row.get("Year", 0),
            genres=row.get("genres", ""),
            runtime=row.get("runtimeMinutes", 0),
            context=f"{row.get('Name', '')} {row.get('genres', '')}",
        )

    def to_metadata(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "genres": self.genres,
            "runtime": self.runtime,
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


class GeneratedReview(BaseModel):
    text: str
    style_confidence: float
    key_elements_used: List[str]
