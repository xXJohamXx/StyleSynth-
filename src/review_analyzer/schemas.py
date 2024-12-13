from typing import Dict, List

from pydantic import BaseModel


class PersonalReviewStyle(BaseModel):
    sentence_patterns: List[Dict[str, str]]
    average_length: int
    sentiment_scores: Dict[str, float]
    common_references: List[str]


class MovieContext(BaseModel):
    title: str
    year: int
    genres: List[str]
    runtime: int


class GeneratedReview(BaseModel):
    text: str
    style_confidence: float
    key_elements_used: List[str]
