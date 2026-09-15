"""Text Preprocessing and Dataset Management Module.

Provides comprehensive text cleaning, tokenization, stop-word elimination,
WordNet lemmatization, dataset loading, class validation, and stratified
train/test splitting.
"""

import os
import re
from typing import List, Tuple, Union, Optional
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Expected universal categories
VALID_CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}


_NLTK_RESOURCES_CHECKED = False


def ensure_nltk_resources():
    """Ensure all required NLTK tokenizers and lexicons are available with clear messages."""
    global _NLTK_RESOURCES_CHECKED
    if _NLTK_RESOURCES_CHECKED:
        return
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[NLTK Info] Resource '{pkg}' not found locally. Downloading '{pkg}'...")
            nltk.download(pkg, quiet=True)
    _NLTK_RESOURCES_CHECKED = True


class TextPreprocessor:
    """Production-grade text preprocessor implementing formula-based cleaning."""

    def __init__(self, custom_stopwords: Optional[set] = None):
        ensure_nltk_resources()
        try:
            default_stops = set(stopwords.words("english"))
        except Exception:
            # Fallback essential English stopwords if offline
            default_stops = {
                "a", "about", "above", "after", "again", "against", "all", "am", "an",
                "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
                "before", "being", "below", "between", "both", "but", "by", "can't",
                "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
                "doing", "don't", "down", "during", "each", "few", "for", "from",
                "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
                "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
                "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
                "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
                "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
                "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
                "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
                "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
                "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
                "then", "there", "there's", "these", "they", "they'd", "they'll",
                "they're", "they've", "this", "those", "through", "to", "too", "under",
                "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
                "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
                "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
                "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
                "your", "yours", "yourself", "yourselves"
            }
        self.stopwords = default_stops if custom_stopwords is None else custom_stopwords
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    @staticmethod
    def strip_html(text: str) -> str:
        """Strip HTML markup tags from raw text."""
        if not isinstance(text, str):
            return ""
        clean = re.compile(r"<[^>]+?>")
        return clean.sub(" ", text)

    def clean_text(self, text: str) -> str:
        """Perform lowercasing, HTML stripping, non-alpha filtering, and whitespace normalization."""
        if not isinstance(text, str):
            return ""

        # 1. Remove HTML tags
        text = self.strip_html(text)

        # 2. Convert to lowercase
        text = text.lower()

        # 3. Strip URL patterns
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 4. Filter non-alphabetic characters (replace punctuation/numbers with spaces)
        text = re.sub(r"[^a-z\s]", " ", text)

        # 5. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into distinct word tokens."""
        clean = self.clean_text(text)
        if not clean:
            return []
        try:
            return word_tokenize(clean)
        except Exception:
            # Regex word tokenization fallback
            return re.findall(r"\b[a-z]+\b", clean)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out stop-words from token sequence."""
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Reduce words to their lemma forms using WordNet."""
        if self.lemmatizer is None:
            return tokens
        try:
            # Lemmatize both noun and verb forms for thorough reduction
            return [self.lemmatizer.lemmatize(t, pos="v") if self.lemmatizer.lemmatize(t, pos="v") != t
                    else self.lemmatizer.lemmatize(t, pos="n") for t in tokens]
        except Exception:
            return tokens

    def preprocess(
        self,
        text: str,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        return_tokens: bool = True,
    ) -> Union[List[str], str]:
        """Execute full preprocessing pipeline.

        Args:
            text: Raw input text string.
            remove_stopwords: Whether to eliminate common stopwords.
            lemmatize: Whether to apply WordNet lemmatization.
            return_tokens: If True, returns List[str]; else returns joined string.

        Returns:
            Preprocessed tokens or space-separated string.
        """
        tokens = self.tokenize(text)

        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        if lemmatize:
            tokens = self.lemmatize(tokens)

        if return_tokens:
            return tokens
        return " ".join(tokens)


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load BBC News CSV dataset and validate categories and integrity.

    Args:
        filepath: Path to the dataset CSV file.

    Returns:
        Validated pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If categories or schema do not match expectations.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    if "category" not in df.columns or "text" not in df.columns:
        raise ValueError(f"Dataset missing required columns ('category', 'text'). Found: {list(df.columns)}")

    # Clean nulls or empty strings
    df = df.dropna(subset=["category", "text"]).copy()
    df["category"] = df["category"].str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    found_categories = set(df["category"].unique())
    if not found_categories.issubset(VALID_CATEGORIES) or len(found_categories) == 0:
        raise ValueError(
            f"Dataset categories {found_categories} do not match expected subset of {VALID_CATEGORIES}"
        )

    return df


def stratified_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a reproducible, stratified train/test split without data leakage.

    Args:
        df: Input DataFrame containing 'category' and 'text'.
        test_size: Proportion of dataset allocated to the test partition (default: 0.2).
        random_state: Seed for reproducible random splitting (default: 42).

    Returns:
        Tuple of (train_df, test_df) with reset indices.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["category"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

