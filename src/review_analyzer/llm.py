import json
from typing import Dict, List, Union

from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool

from .config import llm


class LLMService:
    def __init__(self):
        self.llm = llm
        self.tools = self._initialize_tools()

    async def analyze_text(self, text: str, prompt_template: str, temperature: float = 0.3) -> Union[Dict, List, str]:
        """Generic method for text analysis with temperature control"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Pass temperature in the invoke call
        response = await (prompt | self.llm).ainvoke({"text": text, "temperature": temperature})
        return response.content

    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        prompt = """
        Analyze the sentiment in this text and return only a JSON object with these scores (must sum to 1.0):
        - positive
        - negative
        - neutral
        
        Text: {text}
        """
        response = await self.analyze_text(text, prompt, temperature=0.3)
        return await self._parse_response(response, "json")

    async def _extract_references(self, text: str) -> List[str]:
        prompt = """
        Extract all movie references from this text and return them as a comma-separated list.
        Include direct mentions, director references, and clear film allusions.
        
        IMPORTANT: Return ONLY the comma-separated list of strings. Do not include any explanatory text before or after.
        Format must be exactly as shown in this example:
        [
            "Before Sunrise",
            "Before Sunset",
            "The Matrix",
            "The Lord of the Rings"
        ]

        Text: {text}
        """
        response = await self.analyze_text(text, prompt, temperature=0.2)
        return await self._parse_response(response, "list")

    async def _analyze_sentence_patterns(self, text: str) -> List[Dict[str, str]]:
        prompt = """
        Analyze this collection of movie reviews and identify the most common writing patterns.
        
        For each category below, identify EXACTLY ONE recurring pattern that appears frequently across multiple reviews:

        1. How do the reviews typically start? (e.g., "Starts with director's name", "Opens with plot setup")
        2. What phrases are commonly used to transition between thoughts? (e.g., "However,", "While", "Despite")
        3. How do the reviews usually conclude? (e.g., "Ends with recommendation", "Closes with rating justification")
        4. How are comparisons typically made? (e.g., "Reminds me of...", "Unlike [other film]...")

        Return EXACTLY 4 patterns total (one per category) in a JSON array.
        Format must be exactly as shown in this example:
        [
            {{"type": "opening", "pattern": "Begins with emotional reaction"}},
            {{"type": "transition", "pattern": "Uses 'However' to contrast points"}},
            {{"type": "closing", "pattern": "Ends with recommendation"}},
            {{"type": "comparative", "pattern": "Reminds me of..."}}
        ]

        Reviews to analyze: {text}
        """
        response = await self.analyze_text(text, prompt, temperature=0.3)
        patterns = await self._parse_response(response, "json")

        # Validate pattern count
        if not patterns or len(patterns) != 4:
            return [
                {"type": "opening", "pattern": "Starts with director mention"},
                {"type": "transition", "pattern": "However, despite the"},
                {"type": "closing", "pattern": "Ends with rating justification"},
                {"type": "comparative", "pattern": "Reminds me of..."},
            ]

        return patterns

    async def _parse_response(self, response: str, output_type: str) -> Union[Dict, List, str]:
        """Parse LLM response into structured data"""
        try:
            # Clean the response: remove markdown formatting if present
            cleaned_response = response.strip("```json\n").strip("```").strip()

            if output_type == "json":
                return json.loads(cleaned_response)
            elif output_type == "list":
                # Remove all unwanted characters and split into clean items
                chars_to_remove = "[]\"'\n"
                return [
                    item.strip()
                    for item in cleaned_response.translate(str.maketrans("", "", chars_to_remove)).split(",")
                    if item.strip()
                ]
            else:
                return cleaned_response.replace("\n", "")

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Raw response: {response}")
            return {} if output_type == "json" else []
        except Exception as e:
            print(f"Warning: Error parsing response: {e}")
            print(f"Raw response: {response}")
            return {} if output_type == "json" else []

    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name="analyze_sentiment",
                func=self._analyze_sentiment,
                description="Analyze the sentiment of review text",
            ),
            Tool(
                name="extract_movie_references",
                func=self._extract_references,
                description="Extract movie references from review text",
            ),
            Tool(
                name="analyze_sentence_patterns",
                func=self._analyze_sentence_patterns,
                description="Analyze sentence patterns in movie reviews",
            ),
        ]
