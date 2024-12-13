# StyleSynth 

A Python tool to analyze personal Letterboxd review styles and generate LLM-powered movie reviews that match your writing style.

## Features

- Analyzes Letterboxd reviews to understand writing style
- Identifies common patterns in reviews
- Uses OpenAI's GPT models for personalized review generation
- Vector similarity search for finding similar movies that you've watched

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Poetry (Python dependency management tool)
- OpenAI API key
- Download Letterboxd user data and store in `data/letterboxd` (reviews.csv and watched.csv)

### Installation

1. **Install Poetry** (if not already installed):
   ```bash
   pip install poetry
   ```

2. **Clone and Set Up the Project**:
   ```bash
   # Clone the repository
   git clone https://github.com/xXJohamXx/StyleSynth.git
   cd stylesynth

   # Install dependencies using Poetry
   poetry install
   ```

3. **Set Up Environment Variables**:
   ```bash
   # Option 1: Create or overwrite .env file
   echo "OPENAI_API_KEY=your-api-key-here" > .env

   # Option 2: Manually create .env file
   # Create a new file named .env and add your API key:
   OPENAI_API_KEY=your-api-key-here
   ```

   Replace `your-api-key-here` with your actual OpenAI API key.

### Simple Usage Example

1. **Check out the example script** in [`examples/simple_example.py`](examples/simple_example.py) which demonstrates basic usage of the library.

2. **Run the script using Poetry**:
   ```bash
   poetry run python examples/simple_example.py
   ```

Note: Make sure you have set up your OpenAI API key in the `.env` file before running the example.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.