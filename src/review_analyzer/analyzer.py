from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import asyncio
from typing import Dict, List

import pandas as pd
from slugify import slugify
from tenacity import retry, stop_after_attempt, wait_exponential

from src.review_analyzer.config import embeddings, llm
from src.review_analyzer.llm import LLMService
from src.review_analyzer.schemas import Movie, PersonalReviewStyle
from src.review_analyzer.vector_store import VectorStore


class ReviewStyleAnalyzer:
    def __init__(self):
        self.llm = llm
        self.embeddings = embeddings
        self.llm_service = LLMService()
        self.vector_store = VectorStore()

    async def learn_style(self, reviews_path: str, watched_path: str) -> PersonalReviewStyle:
        reviews_df = pd.read_csv(reviews_path)
        watched_df = pd.read_csv(watched_path)

        # Check for empty DataFrames
        if reviews_df.empty:
            raise ValueError(f'No data found in the provided reviews CSV file: {reviews_path}')
        if watched_df.empty:
            raise ValueError(f'No data found in the provided watched movies CSV file: {watched_path}')

        print('Processing watched movies...')
        batch_size = 50
        total_movies = len(watched_df)

        for start_idx in range(0, total_movies, batch_size):
            batch = watched_df.iloc[start_idx : min(start_idx + batch_size, total_movies)]
            print(f'Processing batch {start_idx//batch_size + 1}/{(total_movies + batch_size - 1)//batch_size}')

            try:
                await self._process_batch(batch)
            except Exception as e:
                print(f'Error processing batch: {e}')
                continue

        style_components = await asyncio.gather(
            self._analyze_vocabulary(reviews_df),
            self._analyze_sentences(reviews_df),
        )

        return self._compile_style_profile(style_components)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
    )
    async def _process_batch(self, batch: pd.DataFrame):
        """Process a batch of movies with rate limiting"""

        # Create Movie objects for new movies
        new_movies = [
            Movie.from_row(row, slugify(f"{row.get('Name', '')}-{row.get('Year', '')}"))
            for _, row in batch.iterrows()
            if not await self.vector_store.get_movie_by_id(slugify(f"{row.get('Name', '')}-{row.get('Year', '')}"))
        ]

        if new_movies:
            print(f'Generating embeddings for {len(new_movies)} new movies...')

            # Generate embeddings in parallel
            embeddings = await asyncio.gather(*(self.embeddings.aembed_query(movie.context) for movie in new_movies))

            # Store movies with their embeddings
            await asyncio.gather(
                *(
                    self.vector_store.store_movie(
                        movie_title=movie.title, metadata=movie.to_metadata(), embedding=embedding
                    )
                    for movie, embedding in zip(new_movies, embeddings)
                )
            )
        else:
            print('All movies in batch already exist in database.')

    async def _analyze_vocabulary(self, reviews_df: pd.DataFrame) -> Dict:
        """Analyze vocabulary patterns in reviews"""
        all_reviews = ' '.join(reviews_df['Review'].tolist())

        average_length = int(reviews_df['Review'].str.split().str.len().mean())

        # Use the tools in parallel
        results = await asyncio.gather(
            self.llm_service._analyze_sentiment(all_reviews),
            self.llm_service._extract_references(all_reviews),
        )

        return {
            'sentiment': results[0],
            'references': results[1],
            'average_length': average_length,
        }

    async def _analyze_sentences(self, reviews_df: pd.DataFrame) -> List[Dict[str, str]]:
        """Analyze common sentence structures and patterns in reviews"""

        all_reviews = ' '.join(reviews_df['Review'].tolist())
        results = await self.llm_service._analyze_sentence_patterns(all_reviews)

        return results

    def _compile_style_profile(self, style_components: List[Dict]) -> PersonalReviewStyle:
        """Compile analyzed components into a PersonalReviewStyle object"""

        vocabulary_data = style_components[0]

        return PersonalReviewStyle(
            sentence_patterns=style_components[1],
            average_length=vocabulary_data['average_length'],
            sentiment_scores=vocabulary_data['sentiment'],
            common_references=vocabulary_data['references'],
        )
