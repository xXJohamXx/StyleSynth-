from typing import Dict, List

from langchain.prompts import ChatPromptTemplate

from src.review_analyzer.config import embeddings, llm
from src.review_analyzer.schemas import GeneratedReview, MovieContext, PersonalReviewStyle
from src.review_analyzer.vector_store import VectorStore


class ReviewGenerator:
    def __init__(self, style_profile: PersonalReviewStyle):
        self.style = style_profile
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = VectorStore()

    async def generate_review(self, movie_context: MovieContext, temperature: float = 0.9) -> GeneratedReview:
        """Generate a review based on movie context and similar movies"""

        movie_query = f"{movie_context.title} {' '.join(movie_context.genres)}"
        query_embedding = await self.embeddings.aembed_query(movie_query)

        similar_movies = await self.vector_store.find_similar_movies(
            query_embedding=query_embedding,
            n_results=5,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                # System message: Define the reviewer's characteristics and style
                (
                    "system",
                    """You are a Letterboxd user reviewing movies in my personal style, no need to be formal.

                My writing style characteristics:
                - Sentiment preferences: {sentiment_scores}
                - Common references: {references}
                
                Use these specific sentence patterns:
                1. Opening: {opening_pattern}
                2. Transitions: {transition_pattern}
                3. Comparisons: {comparison_pattern}
                4. Closing: {closing_pattern}""",
                ),
                # User message: Specific review request and requirements
                (
                    "user",
                    """Generate a review for '{title}'
                
                Similar movies I've watched:
                {similar_movies}

                Consider these movies' genres, and themes when writing the review.
                
                The review must:
                - Recreate vibe and feeling of my reviews
                - Use the specified sentence patterns
                - Be approximately {avg_length} words long
                - Consider my experience with similar films
                - Match my sentiment preferences""",
                ),
            ]
        )

        variables = {
            "title": movie_context.title,
            "similar_movies": self._format_similar_movies(similar_movies),
            "sentiment_scores": str(self.style.sentiment_scores),
            "references": ", ".join(self.style.common_references[:5]),
            "opening_pattern": self.style.sentence_patterns[0]["pattern"],
            "transition_pattern": self.style.sentence_patterns[1]["pattern"],
            "comparison_pattern": self.style.sentence_patterns[2]["pattern"],
            "closing_pattern": self.style.sentence_patterns[3]["pattern"],
            "avg_length": self.style.average_length,
            "temperature": temperature,
        }

        response = await (prompt | self.llm).ainvoke(variables)

        review_text = response.content

        return GeneratedReview(
            text=review_text,
            style_confidence=self._calculate_style_confidence(review_text),
            key_elements_used=self._extract_key_elements(review_text),
        )

    def _calculate_style_confidence(self, review_text: str) -> float:
        """Calculate how well the generated review matches the user's style."""
        confidence = 1.0

        # Check length similarity
        target_length = self.style.average_length
        actual_length = len(review_text.split())
        length_diff = abs(target_length - actual_length) / target_length
        confidence *= max(0.5, 1 - length_diff)

        # Check sentence pattern usage
        for pattern in self.style.sentence_patterns:
            if pattern["pattern"].lower() in review_text.lower():
                confidence *= 1.1

        return min(1.0, confidence)

    def _extract_key_elements(self, review_text: str) -> List[str]:
        """Extract key stylistic elements used in the generated review."""
        elements = []

        # Check for sentence patterns
        for pattern in self.style.sentence_patterns:
            if pattern["pattern"].lower() in review_text.lower():
                elements.append(f"Used {pattern['type']} pattern")

        # Check for common references
        for ref in self.style.common_references:
            if ref.lower() in review_text.lower():
                elements.append(f"Referenced {ref}")

        # Check length adherence
        actual_length = len(review_text.split())
        if abs(actual_length - self.style.average_length) <= self.style.average_length * 0.2:
            elements.append("Matched target length")

        return elements

    def _format_similar_movies(self, similar_movies: List[Dict]) -> str:
        """Format similar movies in a clear, structured way for the LLM"""
        formatted = []
        for movie in similar_movies:
            metadata = movie["metadata"]
            # Handle genres that might come as string or list
            genres = metadata.get("genres", "")
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(",")]

            formatted.append(
                f"• {metadata.get('title', 'Unknown')} ({metadata.get('year', 'N/A')})\n"
                f"  Genres: {', '.join(genres)}\n"
                f"  Runtime: {metadata.get('runtime', 'N/A')} minutes"
            )
        return "\n\n".join(formatted)

    def _format_sentence_patterns(self) -> str:
        formatted = []
        for pattern in self.style.sentence_patterns:
            formatted.append(f"- {pattern['type']}: {pattern['pattern']}")
        return "\n".join(formatted)
