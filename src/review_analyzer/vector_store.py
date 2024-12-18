import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, persist_dir: str = './.vectordb'):
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))

        # Create collections with cosine similarity
        self.movies_collection = self.client.get_or_create_collection(
            name='watched_movies', metadata={'hnsw:space': 'cosine'}
        )

    async def store_movie(self, movie_title: str, metadata: dict, embedding: List[float]) -> bool:
        """Store a movie with its embedding and metadata."""
        movie_id = metadata.get('id')
        if not movie_id:
            logger.error(f'No ID provided for movie: {movie_title}')
            return False

        try:
            self.movies_collection.add(
                documents=[movie_title], metadatas=[metadata], embeddings=[embedding], ids=[movie_id]
            )
            logger.info(f'Successfully stored movie: {movie_title}')
            return True

        except Exception as e:
            logger.error(f'Error storing movie {movie_title}: {e}')
            return False

    async def get_movie_by_id(self, movie_id: str) -> Optional[Dict]:
        """Retrieve a specific movie by ID"""
        try:
            results = self.movies_collection.get(ids=[movie_id], include=['documents', 'metadatas', 'embeddings'])
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0],
                    'embedding': results['embeddings'][0],
                }
        except Exception as e:
            logger.error(f'Error retrieving movie {movie_id}: {e}')
        return None

    async def find_similar_movies(
        self, query_embedding: List[float], n_results: int = 5, filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Find similar movies using semantic similarity and optional metadata filters"""
        try:
            results = self.movies_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
            )

            return [
                {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                }
                for i in range(len(results['ids'][0]))
            ]

        except Exception as e:
            logger.error(f'Error querying similar movies: {e}')
            return []

    async def get_movie_count(self) -> int:
        """Get total number of stored movies"""
        return self.movies_collection.count()
