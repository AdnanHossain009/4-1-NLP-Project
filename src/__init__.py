"""News Category Classification and Semantic Analysis Package.

This package provides foundational, formula-based NLP modules:
- Text Preprocessing (HTML stripping, tokenization, lemmatization)
- Dataset Loading & Stratified Splitting
- Generalized N-Gram Language Modeling (MLE, Laplace smoothing, Next-word prediction)
- Multinomial Naive Bayes Classifier (from scratch using pure NumPy)
- Custom Word2Vec Skip-Gram Architecture (PyTorch from scratch)
- Offline Pretrained GloVe Embeddings & Document Vector Aggregation
- One-vs-Rest Logistic Regression (from scratch using pure NumPy)
"""

from src.preprocessing import (
    TextPreprocessor,
    load_dataset,
    stratified_train_test_split,
)
from src.ngram_model import NGramLanguageModel

try:
    from src.naive_bayes import MultinomialNaiveBayes
except ImportError:
    MultinomialNaiveBayes = None

try:
    from src.word2vec_scratch import CustomWord2Vec, SkipGramModel
except ImportError:
    CustomWord2Vec, SkipGramModel = None, None

try:
    from src.embeddings import GloVeEmbeddings, compute_training_idf, aggregate_document_mean, aggregate_document_tfidf
except ImportError:
    GloVeEmbeddings = None

try:
    from src.logistic_regression import OneVsRestLogisticRegression, BinaryLogisticRegression, sigmoid
except ImportError:
    OneVsRestLogisticRegression = None

__all__ = [
    "TextPreprocessor",
    "load_dataset",
    "stratified_train_test_split",
    "NGramLanguageModel",
    "MultinomialNaiveBayes",
    "CustomWord2Vec",
    "SkipGramModel",
    "GloVeEmbeddings",
    "OneVsRestLogisticRegression",
    "BinaryLogisticRegression",
    "sigmoid",
]
