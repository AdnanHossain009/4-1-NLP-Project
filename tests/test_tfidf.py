"""Unit Tests for TF-IDF Feature Extractor and Logistic Regression on N-Gram Features.

Covers:
- Unigram extraction
- Bigram extraction
- Training DF and IDF calculation accuracy
- Zero test data leakage (test terms not seen in training do not affect IDF or break transform)
- L2 vector normalization
- OvR Logistic Regression training on TF-IDF features
- Probability normalization across categories
- Model serialization and deserialization
"""

import math
import tempfile
import os
import numpy as np
import pytest

from src.tfidf_extractor import TfidfNGramFeatureExtractor
from src.logistic_regression import OneVsRestLogisticRegression, BinaryLogisticRegression


def test_unigram_extraction():
    """Verify that unigram extraction retains exact token sequences."""
    extractor = TfidfNGramFeatureExtractor(ngram_range=(1, 1))
    tokens = ["football", "match", "final"]
    extracted = extractor.extract_ngrams(tokens)
    assert extracted == ["football", "match", "final"]


def test_unigram_and_bigram_extraction():
    """Verify that (1, 2) ngram extraction includes both unigrams and consecutive bigrams."""
    extractor = TfidfNGramFeatureExtractor(ngram_range=(1, 2))
    tokens = ["world", "cup", "final"]
    extracted = extractor.extract_ngrams(tokens)
    expected = ["world", "cup", "final", "world_cup", "cup_final"]
    assert extracted == expected


def test_idf_exact_mathematical_formula():
    """Verify IDF values match the smoothed textbook formula: ln((N+1)/(DF+1)) + 1."""
    corpus = [
        ["stock", "market", "profit"],
        ["stock", "market", "shares"],
        ["football", "match", "cup"],
        ["football", "cup"],
    ]
    # N = 4
    # DF("stock") = 2, DF("cup") = 2, DF("market") = 2, DF("profit") = 1
    extractor = TfidfNGramFeatureExtractor(ngram_range=(1, 1), min_df=1, max_features=None)
    extractor.fit(corpus)

    n = 4.0
    # Expected IDF for "stock": ln((4+1)/(2+1)) + 1 = ln(5/3) + 1
    expected_idf_stock = math.log(5.0 / 3.0) + 1.0
    idx_stock = extractor.vocabulary_["stock"]
    actual_idf_stock = extractor.idf_diag_[idx_stock]

    assert pytest.approx(actual_idf_stock, rel=1e-5) == expected_idf_stock


def test_zero_test_data_leakage_and_unseen_terms():
    """Verify that transforming unseen words does not alter IDF or crash."""
    train_corpus = [
        ["economy", "growth", "inflation"],
        ["government", "minister", "election"],
    ]
    extractor = TfidfNGramFeatureExtractor(ngram_range=(1, 2), min_df=1)
    extractor.fit(train_corpus)

    test_corpus = [
        ["economy", "unknown_term", "alien_word"],
        ["nonexistent_bigram_1", "nonexistent_bigram_2"],
    ]
    X_test = extractor.transform(test_corpus)

    assert X_test.shape == (2, extractor.vocab_size)
    # The second document has only unseen terms; normalized vector should be all zeros
    assert np.all(X_test[1] == 0.0)
    # The first document has "economy", so its feature at "economy" should be > 0
    econ_idx = extractor.vocabulary_["economy"]
    assert X_test[0, econ_idx] > 0.0


def test_l2_vector_normalization():
    """Verify that rows of non-empty documents have unit L2 norm."""
    corpus = [
        ["premier", "league", "striker", "goal"],
        ["prime", "minister", "parliament", "tax"],
    ]
    extractor = TfidfNGramFeatureExtractor(ngram_range=(1, 2), min_df=1, norm="l2")
    X = extractor.fit_transform(corpus)

    for i in range(X.shape[0]):
        norm = np.linalg.norm(X[i])
        assert pytest.approx(norm, abs=1e-5) == 1.0


def test_min_df_and_max_features():
    """Verify that min_df and max_features filter the vocabulary properly."""
    corpus = [
        ["rare_word", "common_word"],
        ["another_rare", "common_word"],
        ["third_rare", "common_word"],
    ]
    extractor = TfidfNGramFeatureExtractor(ngram_range=(1, 1), min_df=2, max_features=1)
    extractor.fit(corpus)

    # Only "common_word" appears in >= 2 docs
    assert extractor.vocab_size == 1
    assert "common_word" in extractor.vocabulary_


def test_ovr_logistic_regression_fit_on_tfidf():
    """Verify that OneVsRestLogisticRegression trains on TF-IDF features and outputs valid probabilities."""
    corpus = [
        ["football", "match", "goal", "striker"],
        ["world", "cup", "tournament", "football"],
        ["market", "shares", "investor", "profit"],
        ["company", "shares", "growth", "market"],
        ["minister", "election", "parliament", "vote"],
        ["government", "minister", "prime", "parliament"],
        ["film", "director", "actor", "cinema"],
        ["movie", "actor", "festival", "film"],
        ["software", "computer", "algorithm", "code"],
        ["technology", "internet", "software", "computer"],
    ]
    labels = [
        "sport", "sport",
        "business", "business",
        "politics", "politics",
        "entertainment", "entertainment",
        "tech", "tech",
    ]

    extractor = TfidfNGramFeatureExtractor(ngram_range=(1, 2), min_df=1, max_features=100)
    X = extractor.fit_transform(corpus)

    clf = OneVsRestLogisticRegression(
        classes=["business", "entertainment", "politics", "sport", "tech"],
        learning_rate=0.5,
        n_iterations=200,
        seed=42,
    )
    losses = clf.fit(X, labels)

    assert len(clf.classifiers) == 5
    for cat in clf.classes:
        assert cat in losses
        # Loss should decrease
        assert losses[cat][-1] < losses[cat][0]

    # Predict probabilities on training samples
    probs = clf.predict_proba(X)
    assert probs.shape == (len(corpus), 5)
    # Each row must sum to 1.0
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)

    # Check predictions
    preds = clf.predict(X)
    assert len(preds) == len(labels)
    assert all(p in clf.classes for p in preds)


def test_tfidf_extractor_save_and_load():
    """Verify serialization and deserialization of the extractor."""
    corpus = [
        ["tech", "software", "development"],
        ["market", "economy", "growth"],
    ]
    extractor = TfidfNGramFeatureExtractor(ngram_range=(1, 2), min_df=1)
    extractor.fit(corpus)
    X_orig = extractor.transform(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_tfidf.pkl")
        extractor.save(save_path)

        loaded = TfidfNGramFeatureExtractor.load(save_path)
        assert loaded.vocabulary_ == extractor.vocabulary_
        assert loaded.ngram_range == extractor.ngram_range
        X_loaded = loaded.transform(corpus)
        assert np.allclose(X_orig, X_loaded)
