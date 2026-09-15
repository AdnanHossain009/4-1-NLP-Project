"""Unit tests for Part 1: Dataset and Text Preprocessing Pipeline."""

import os
import pandas as pd
import pytest
from src.preprocessing import (
    TextPreprocessor,
    load_dataset,
    stratified_train_test_split,
    VALID_CATEGORIES,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bbc_news.csv")


@pytest.fixture
def preprocessor():
    """Fixture to provide a clean TextPreprocessor instance."""
    return TextPreprocessor()


def test_html_stripping(preprocessor):
    raw = "<p>This is a <b>bold</b> statement with <a href='https://example.com'>links</a>.</p>"
    cleaned = preprocessor.clean_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "</a>" not in cleaned
    assert "bold statement with links" in cleaned


def test_lowercasing_and_punctuation_filtering(preprocessor):
    raw = "Breaking: INFLATION Rises by 3.5% in London, UK! What's next? #Economy2026"
    cleaned = preprocessor.clean_text(raw)
    assert cleaned == "breaking inflation rises by in london uk what s next economy"
    assert not any(char.isupper() for char in cleaned)
    assert not any(char.isdigit() for char in cleaned)
    assert "#" not in cleaned and "!" not in cleaned and "%" not in cleaned


def test_tokenization(preprocessor):
    text = "The quick brown fox jumps over the lazy dog."
    tokens = preprocessor.tokenize(text)
    assert isinstance(tokens, list)
    assert "quick" in tokens
    assert "brown" in tokens
    assert "dog" in tokens


def test_stopword_removal(preprocessor):
    tokens = ["the", "government", "and", "the", "minister", "are", "in", "talks"]
    filtered = preprocessor.remove_stopwords(tokens)
    assert "the" not in filtered
    assert "and" not in filtered
    assert "are" not in filtered
    assert "in" not in filtered
    assert "government" in filtered
    assert "minister" in filtered
    assert "talks" in filtered


def test_lemmatization(preprocessor):
    tokens = ["running", "players", "studying", "matches", "corporations"]
    lemmas = preprocessor.lemmatize(tokens)
    assert isinstance(lemmas, list)
    # Verbs reduced to base (run, study) and plural nouns to singular (player, match, corporation)
    assert "player" in lemmas
    assert "match" in lemmas
    assert "corporation" in lemmas
    assert ("run" in lemmas or "running" in lemmas)


def test_end_to_end_preprocess(preprocessor):
    text = "<h1>Technology News</h1> <p>Engineers are building fast computers!</p>"
    tokens = preprocessor.preprocess(text, remove_stopwords=True, lemmatize=True, return_tokens=True)
    assert isinstance(tokens, list)
    assert "technology" in tokens
    assert "computer" in tokens
    assert "are" not in tokens

    joined_text = preprocessor.preprocess(text, return_tokens=False)
    assert isinstance(joined_text, str)
    assert "technology" in joined_text


def test_load_dataset_and_validate_categories():
    assert os.path.exists(DATA_PATH), f"Dataset {DATA_PATH} must exist before running test."
    df = load_dataset(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2225
    assert "category" in df.columns
    assert "text" in df.columns

    categories = set(df["category"].unique())
    assert categories == VALID_CATEGORIES
    for cat in VALID_CATEGORIES:
        assert (df["category"] == cat).sum() > 0


def test_load_dataset_missing_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file.csv")


def test_stratified_train_test_split():
    df = load_dataset(DATA_PATH)
    train_df, test_df = stratified_train_test_split(df, test_size=0.2, random_state=42)

    # Size verification
    total_len = len(df)
    assert len(train_df) + len(test_df) == total_len
    assert len(test_df) == pytest.approx(int(total_len * 0.2), abs=2)

    # Class balance verification: proportion of each category in train and test should match full dataset
    for cat in VALID_CATEGORIES:
        orig_ratio = (df["category"] == cat).mean()
        train_ratio = (train_df["category"] == cat).mean()
        test_ratio = (test_df["category"] == cat).mean()

        assert train_ratio == pytest.approx(orig_ratio, abs=0.02)
        assert test_ratio == pytest.approx(orig_ratio, abs=0.02)


def test_train_test_split_reproducibility():
    df = load_dataset(DATA_PATH)
    train_df1, test_df1 = stratified_train_test_split(df, test_size=0.2, random_state=42)
    train_df2, test_df2 = stratified_train_test_split(df, test_size=0.2, random_state=42)

    assert train_df1["text"].iloc[0] == train_df2["text"].iloc[0]
    assert test_df1["text"].iloc[0] == test_df2["text"].iloc[0]
