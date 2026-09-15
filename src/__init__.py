"""News Category Classification and Semantic Analysis Package.

This package provides foundational, formula-based NLP modules:
- Text Preprocessing (HTML stripping, tokenization, lemmatization)
- Dataset Loading & Stratified Splitting
- Generalized N-Gram Language Modeling (MLE, Laplace smoothing, Next-word prediction)
"""

from src.preprocessing import (
    TextPreprocessor,
    load_dataset,
    stratified_train_test_split,
)
from src.ngram_model import NGramLanguageModel

__all__ = [
    "TextPreprocessor",
    "load_dataset",
    "stratified_train_test_split",
    "NGramLanguageModel",
]
