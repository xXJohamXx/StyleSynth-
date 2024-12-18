from unittest.mock import patch

import pandas as pd
import pytest

with patch('src.review_analyzer.config.llm'), patch('src.review_analyzer.config.embeddings'):
    from src.review_analyzer.analyzer import ReviewStyleAnalyzer


@pytest.fixture(autouse=True)
def set_openai_api_key(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test_key')


@pytest.fixture
async def mock_analyzer():
    with (
        patch('src.review_analyzer.analyzer.LLMService') as MockLLMService,
        patch('src.review_analyzer.analyzer.VectorStore') as MockVectorStore,
        patch('src.review_analyzer.config.llm') as mock_llm,
        patch('src.review_analyzer.config.embeddings') as mock_embeddings,
        patch('src.review_analyzer.analyzer.pd.read_csv') as mock_read_csv,
    ):
        _mock_llm_service = MockLLMService.return_value
        _mock_vector_store = MockVectorStore.return_value
        _mock_llm = mock_llm.return_value
        _mock_embeddings = mock_embeddings.return_value

        analyzer = ReviewStyleAnalyzer()

        with (
            patch.object(analyzer, '_analyze_vocabulary') as mock_analyze_vocabulary,
            patch.object(analyzer, '_analyze_sentences') as mock_analyze_sentences,
            patch.object(analyzer, '_process_batch') as mock_process_batch,
        ):
            yield {
                'analyzer': analyzer,
                'mock_read_csv': mock_read_csv,
                'mock_analyze_vocabulary': mock_analyze_vocabulary,
                'mock_analyze_sentences': mock_analyze_sentences,
                'mock_process_batch': mock_process_batch,
            }


async def test_learn_style_valid_data(mock_analyzer):
    mock_analyzer['mock_read_csv'].return_value = pd.DataFrame({'Review': ['Great movie!', 'Not bad']})
    mock_analyzer['mock_analyze_vocabulary'].return_value = {
        'sentiment': {'positive': 0.5, 'negative': 0.3, 'neutral': 0.2},
        'references': ['Inception', 'The Matrix'],
        'average_length': 100,
    }
    mock_analyzer['mock_analyze_sentences'].return_value = [
        {'type': 'opening', 'pattern': 'Starts with a quote'},
        {'type': 'transition', 'pattern': 'However, despite the'},
        {'type': 'closing', 'pattern': 'Ends with rating justification'},
        {'type': 'comparative', 'pattern': 'Reminds me of...'},
    ]

    style_profile = await mock_analyzer['analyzer'].learn_style('path/to/reviews.csv', 'path/to/watched.csv')

    assert style_profile is not None
    assert style_profile.sentiment_scores == {'positive': 0.5, 'negative': 0.3, 'neutral': 0.2}
    assert style_profile.common_references == ['Inception', 'The Matrix']
    assert style_profile.sentence_patterns[0]['pattern'] == 'Starts with a quote'
    assert style_profile.sentence_patterns[1]['pattern'] == 'However, despite the'
    assert style_profile.sentence_patterns[2]['pattern'] == 'Ends with rating justification'
    assert style_profile.sentence_patterns[3]['pattern'] == 'Reminds me of...'
    assert mock_analyzer['mock_read_csv'].call_count == 2, 'Expected 2 calls to pd.read_csv'
    assert mock_analyzer['mock_analyze_vocabulary'].call_count == 1, 'Expected 1 call to _analyze_vocabulary'
    assert mock_analyzer['mock_analyze_sentences'].call_count == 1, 'Expected 1 call to _analyze_sentences'


async def test_learn_style_invalid_path(mock_analyzer):
    mock_analyzer['mock_read_csv'].side_effect = FileNotFoundError

    with pytest.raises(FileNotFoundError):
        await mock_analyzer['analyzer'].learn_style('fake/path/to/reviews.csv', 'fake/path/to/watched.csv')


async def test_learn_style_empty_data(mock_analyzer):
    mock_analyzer['mock_read_csv'].return_value = pd.DataFrame()

    with pytest.raises(ValueError):
        await mock_analyzer['analyzer'].learn_style('path/to/reviews.csv', 'path/to/watched.csv')
