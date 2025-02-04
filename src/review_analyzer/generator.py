import json
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
        self._pattern_scores = {}

    async def generate_review(self, movie_context: MovieContext, temperature: float = 0.9) -> GeneratedReview:
        """Generate a review based on movie context and similar movies"""

        movie_query = movie_context.get_embedding_context()
        query_embedding = await self.embeddings.aembed_query(movie_query)

        similar_movies = await self.vector_store.find_similar_movies(
            query_embedding=query_embedding,
            n_results=5,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                # System message: Define the reviewer's characteristics and style
                (
                    'system',
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
                    'user',
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
            'title': movie_context.title,
            'similar_movies': self._format_similar_movies(similar_movies),
            'sentiment_scores': str(self.style.sentiment_scores),
            'references': ', '.join(self.style.common_references[:5]),
            'opening_pattern': self.style.sentence_patterns[0]['pattern'],
            'transition_pattern': self.style.sentence_patterns[1]['pattern'],
            'comparison_pattern': self.style.sentence_patterns[2]['pattern'],
            'closing_pattern': self.style.sentence_patterns[3]['pattern'],
            'avg_length': self.style.average_length,
            'temperature': temperature,
        }

        response = await (prompt | self.llm).ainvoke(variables)

        review_text = response.content

        return GeneratedReview(
            text=review_text,
            style_confidence=await self._calculate_style_confidence(review_text),
            key_elements_used=self._extract_key_elements(review_text),
        )

    async def _calculate_style_confidence(self, review_text: str) -> Dict[str, float]:
        """
        Calculate how well the generated review matches the user's style.
        Returns a dictionary with individual confidence scores for each pattern and length.
        Raises:
            ValueError: If the LLM response cannot be parsed or is not in the expected format.
        """
        confidence_scores = {
            'length': 0.0,
            'opening': 0.0,
            'transition': 0.0,
            'closing': 0.0,
            'comparative': 0.0,
        }

        # Check length similarity
        target_length = self.style.average_length
        actual_length = len(review_text.split())
        length_diff = abs(target_length - actual_length) / target_length
        confidence_scores['length'] = 1 - length_diff

        # Get pattern scores from LLM
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """You are a writing style analyzer. Analyze how well a review matches given patterns.
                You must return a valid JSON object containing ONLY numeric scores between 0 and 1.
                Do not include any additional text, explanations, or formatting.""",
                ),
                (
                    'user',
                    """Review: {review}
                
                Patterns to analyze:
                {patterns}
                
                Return a valid JSON object exactly like this:
                {{"opening": 0.8, "transition": 0.7, "closing": 0.9, "comparative": 0.6}}
                
                Use only numbers between 0 and 1 for scores. Do not include any other text.""",
                ),
            ]
        )

        variables = {
            'review': review_text,
            'patterns': '\n'.join(f"- {p['type']}: {p['pattern']}" for p in self.style.sentence_patterns),
        }

        response = await (prompt | self.llm.with_config({'temperature': 0.1})).ainvoke(variables)
        try:
            pattern_scores = json.loads(response.content.strip())

            for pattern in self.style.sentence_patterns:
                pattern_type = pattern['type']
                score = float(pattern_scores[pattern_type])
                if not 0 <= score <= 1:
                    raise ValueError(f'Invalid score range for {pattern_type}: {score}')
                confidence_scores[pattern_type] = score

        except (ValueError, json.JSONDecodeError) as e:
            print(f'Error parsing response: {str(e)}')
            raise ValueError(f'Failed to analyze review style: {str(e)}') from e

        return confidence_scores

    def _extract_key_elements(self, review_text: str) -> List[str]:
        """Extract key stylistic elements used in the generated review."""
        elements = []

        # Check for common references
        for ref in self.style.common_references:
            if ref.lower() in review_text.lower():
                elements.append(f'Referenced {ref}')

        return elements

    def _format_similar_movies(self, similar_movies: List[Dict]) -> str:
        """Format similar movies in a clear, structured way for the LLM"""
        formatted = []
        for movie in similar_movies:
            metadata = movie['metadata']
            # Handle genres that might come as string or list
            genres = metadata.get('genres', '')
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(',')]

            formatted.append(
                f"• {metadata.get('title', 'Unknown')} ({metadata.get('year', 'N/A')})\n"
                f"  Genres: {', '.join(genres)}\n"
                f"  Runtime: {metadata.get('runtime', 'N/A')} minutes"
            )
        return '\n\n'.join(formatted)

    def _format_sentence_patterns(self) -> str:
        formatted = []
        for pattern in self.style.sentence_patterns:
            formatted.append(f"- {pattern['type']}: {pattern['pattern']}")
        return '\n'.join(formatted)
