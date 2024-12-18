import json
from typing import Dict, List, Union

from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool

from src.review_analyzer.config import llm


class LLMService:
    def __init__(self):
        self.llm = llm
        self.tools = self._initialize_tools()

    async def analyze_text(
        self, text: str, prompt: ChatPromptTemplate, temperature: float = 0.7
    ) -> Union[Dict, List, str]:
        """Generic method for text analysis with temperature control"""

        configured_llm = self.llm.with_config({'temperature': temperature})

        # Invoke the chain
        response = await (prompt | configured_llm).ainvoke({'text': text})
        return response.content

    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """You are a sentiment analyzer that returns only JSON.
            The scores must sum to 1.0 and include: positive, negative, and neutral.""",
                ),
                ('user', 'Analyze the sentiment in this text: {text}'),
            ]
        )

        response = await self.analyze_text(text, prompt, temperature=0.3)
        return await self._parse_response(response, 'json')

    async def _extract_references(self, text: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """You are a movie reference extractor that returns only a comma-separated list.
            Include direct mentions, director references, and clear film allusions.""",
                ),
                (
                    'user',
                    """Extract movie references from this text.
            Return ONLY the comma-separated list, no explanatory text.
            
            Text: {text}""",
                ),
            ]
        )

        response = await self.analyze_text(text, prompt, temperature=0.2)
        return await self._parse_response(response, 'list')

    async def _analyze_sentence_patterns(self, text: str) -> List[Dict[str, str]]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """You analyze writing patterns in movie reviews.
            Return EXACTLY 4 patterns (one per category) in a JSON array.""",
                ),
                (
                    'user',
                    """Analyze these reviews and identify common patterns for:
            1. Opening sentences
            2. Transition phrases
            3. Closing statements
            4. Comparative structures

            Format must be:
            [
                {{"type": "opening", "pattern": "pattern description"}},
                {{"type": "transition", "pattern": "pattern description"}},
                {{"type": "closing", "pattern": "pattern description"}},
                {{"type": "comparative", "pattern": "pattern description"}}
            ]

            Reviews: {text}""",
                ),
            ]
        )

        response = await self.analyze_text(text, prompt, temperature=0.3)
        patterns = await self._parse_response(response, 'json')

        # Validate pattern count
        if not patterns or len(patterns) != 4:
            return [
                {'type': 'opening', 'pattern': 'Starts with director mention'},
                {'type': 'transition', 'pattern': 'However, despite the'},
                {'type': 'closing', 'pattern': 'Ends with rating justification'},
                {'type': 'comparative', 'pattern': 'Reminds me of...'},
            ]

        return patterns

    async def _parse_response(self, response: str, output_type: str) -> Union[Dict, List, str]:
        """Parse LLM response into structured data"""
        try:
            # Clean the response: remove markdown formatting if present
            cleaned_response = response.strip('```json\n').strip('```').strip()

            if output_type == 'json':
                return json.loads(cleaned_response)
            elif output_type == 'list':
                # Remove all unwanted characters and split into clean items
                chars_to_remove = '[]"\'\n'
                return [
                    item.strip()
                    for item in cleaned_response.translate(str.maketrans('', '', chars_to_remove)).split(',')
                    if item.strip()
                ]
            else:
                return cleaned_response.replace('\n', '')

        except json.JSONDecodeError as e:
            print(f'Warning: Failed to parse JSON response: {e}')
            print(f'Raw response: {response}')
            return {} if output_type == 'json' else []
        except Exception as e:
            print(f'Warning: Error parsing response: {e}')
            print(f'Raw response: {response}')
            return {} if output_type == 'json' else []

    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name='analyze_sentiment',
                func=self._analyze_sentiment,
                description='Analyze the sentiment of review text',
            ),
            Tool(
                name='extract_movie_references',
                func=self._extract_references,
                description='Extract movie references from review text',
            ),
            Tool(
                name='analyze_sentence_patterns',
                func=self._analyze_sentence_patterns,
                description='Analyze sentence patterns in movie reviews',
            ),
        ]
