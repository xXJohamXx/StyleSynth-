import asyncio

from src.review_analyzer.analyzer import ReviewStyleAnalyzer
from src.review_analyzer.schemas import MovieContext


async def simple_example(movie: MovieContext):
    # Initialize the analyzer
    analyzer = ReviewStyleAnalyzer()

    style_profile = await analyzer.learn_style(
        reviews_path="data/letterboxd/reviews.csv", watched_path="data/letterboxd/watched.csv"
    )

    print("\nYour Review Style:")
    print(f"Average Length: {style_profile.average_length} words")

    # Initialize generator with your style
    from src.review_analyzer.generator import ReviewGenerator

    generator = ReviewGenerator(style_profile)

    review = await generator.generate_review(movie)

    print(f"\nGenerated Review for {movie.title}:")
    print("\nReview:")
    print(review.text)

    print(f"\nKey elements used: {review.key_elements_used}")


# Run the example
if __name__ == "__main__":
    # Very simple usage example

    # Enter Movie Details
    movie = MovieContext(title="Gattaca", year=1997, genres=["Drama", "Sci-Fi", "Thriller"], runtime=106)

    asyncio.run(simple_example(movie))
