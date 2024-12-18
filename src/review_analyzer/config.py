import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Get the project root directory (2 levels up from this file)
project_root = Path(__file__).parent.parent.parent

dotenv_path = project_root / '.env'
load_dotenv(dotenv_path)

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPENAI_API_KEY not found in environment variables')

# Create shared instances
llm = ChatOpenAI(api_key=api_key, model='gpt-4o')
embeddings = OpenAIEmbeddings(api_key=api_key)
