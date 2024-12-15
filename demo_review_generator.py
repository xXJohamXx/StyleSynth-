import asyncio
import logging
from datetime import datetime
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from src.review_analyzer.analyzer import ReviewStyleAnalyzer
from src.review_analyzer.config import embeddings
from src.review_analyzer.generator import ReviewGenerator
from src.review_analyzer.schemas import MovieContext
from src.review_analyzer.vector_store import VectorStore

logging.basicConfig(level=logging.ERROR)
console = Console()


async def demo_review_generation(movie: MovieContext):
    """Demonstrate the review generation pipeline"""

    # 1. Show startup banner
    console.print(
        Panel.fit(
            "[bold blue]Movie Review Generator Demo[/bold blue]\n"
            "Analyzing your review style and generating personalized movie reviews",
            border_style="blue",
        )
    )

    # 2. Initialize components
    console.print("\n[yellow]Initializing components...[/yellow]")
    analyzer = ReviewStyleAnalyzer()
    style_profile = await analyzer.learn_style(
        reviews_path="data/letterboxd/reviews.csv", watched_path="data/letterboxd/watched.csv"
    )

    # 3. Create a table for movie details
    movie_table = Table(title="Movie Details", show_header=True)
    movie_table.add_column("Property", style="cyan")
    movie_table.add_column("Value", style="green")
    movie_table.add_row("Title", movie.title)
    movie_table.add_row("Year", str(movie.year))
    movie_table.add_row("Genres", ", ".join(movie.genres))
    movie_table.add_row("Runtime", f"{movie.runtime} minutes")
    console.print(movie_table)

    # 4. Show similar movies
    console.print("\n[yellow]Finding similar movies you have watched...[/yellow]\n")
    vector_store = VectorStore()
    movie_query = f"{movie.title} {' '.join(movie.genres)}"
    query_embedding = await embeddings.aembed_query(movie_query)

    similar_movies = await vector_store.find_similar_movies(query_embedding=query_embedding, n_results=5)

    similar_table = Table(title="Most Similar Movies You Have Watched", show_header=True)
    similar_table.add_column("Title", style="cyan")
    similar_table.add_column("Year", style="green")
    similar_table.add_column("Genres", style="blue")
    similar_table.add_column("Runtimw", style="green")

    for m in similar_movies:
        metadata = m["metadata"]
        similar_table.add_row(metadata["title"], str(metadata["year"]), metadata["genres"], str(metadata["runtime"]))
    console.print(similar_table)

    # 5. Generate review
    console.print("\n[yellow]Generating personalized review...[/yellow]")
    generator = ReviewGenerator(style_profile)
    review = await generator.generate_review(movie)

    # 6. Show final review
    review_panel = Panel(
        f"[italic]{review.text}[/italic]\n\n" f"[dim]Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        title="Generated Review",
        border_style="green",
    )
    console.print("\n", review_panel)

    # 7. Show statistics
    stats_table = Table(title="Generation Statistics", show_header=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Word Count", str(len(review.text.split())))
    stats_table.add_row("Similar Movies Used", str(len(similar_movies)))
    stats_table.add_row("Key Elements Used", ", ".join(review.key_elements_used))
    console.print("\n", stats_table)


class MovieInputHandler:
    def __init__(self):
        self.console = Console()

    def get_title(self) -> str:
        return Prompt.ask("Enter movie title")

    def get_year(self) -> int:
        while True:
            year = IntPrompt.ask("Enter movie year")
            if 1900 <= year <= 2024:
                return year
            self.console.print("[red]Please enter a valid year between 1900 and 2024[/red]")

    def get_genres(self) -> List[str]:
        genres_input = Prompt.ask("Enter up to 3 genres", default="Action, Drama", show_default=True)
        return [genre.strip() for genre in genres_input.split(",")]

    def get_runtime(self) -> int:
        while True:
            runtime = IntPrompt.ask("Enter movie runtime (in minutes)")
            if 0 < runtime < 500:
                return runtime
            self.console.print("[red]Please enter a valid runtime between 1 and 500 minutes[/red]")

    def get_movie_context(self) -> MovieContext:
        """Get all movie details and return MovieContext object"""
        return MovieContext(
            title=self.get_title(), year=self.get_year(), genres=self.get_genres(), runtime=self.get_runtime()
        )


if __name__ == "__main__":
    input_handler = MovieInputHandler()
    movie = input_handler.get_movie_context()

    asyncio.run(demo_review_generation(movie))
