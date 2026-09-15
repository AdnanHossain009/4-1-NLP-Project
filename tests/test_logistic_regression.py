"""Unit tests for Part 6: One-vs-Rest (OvR) Logistic Regression from Scratch."""

import os
import pickle
import numpy as np
import pytest
from src.logistic_regression import (
    sigmoid,
    BinaryLogisticRegression,
    OneVsRestLogisticRegression,
    VALID_CATEGORIES,
)


def test_sigmoid_numerical_stability():
    # Standard values
    assert sigmoid(0.0) == pytest.approx(0.5)

    # Large positive and negative values (clipping verification)
    sig_pos = sigmoid(1000.0)
    sig_neg = sigmoid(-1000.0)

    assert 0.0 < sig_pos <= 1.0
    assert 0.0 <= sig_neg < 1.0
    assert sig_pos > 0.9999
    assert sig_neg < 0.0001
    assert not np.isnan(sig_pos)
    assert not np.isnan(sig_neg)

    # Array input
    arr = np.array([-500.0, 0.0, 500.0])
    res = sigmoid(arr)
    assert res.shape == (3,)
    assert (res >= 0.0).all() and (res <= 1.0).all()


def test_binary_logistic_regression_fit():
    rng = np.random.RandomState(42)
    # Linearly separable 2D dataset
    X_pos = rng.randn(40, 4) + 2.0
    X_neg = rng.randn(40, 4) - 2.0
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 40 + [0] * 40)

    clf = BinaryLogisticRegression(learning_rate=0.5, n_iterations=300, seed=42)
    losses = clf.fit(X, y)

    assert len(losses) > 0
    assert losses[-1] < losses[0]  # Loss must decrease
    assert clf.w.shape == (4,)

    preds = clf.predict_proba(X)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()
    # High accuracy on linearly separable data
    pred_labels = (preds >= 0.5).astype(int)
    acc = np.mean(pred_labels == y)
    assert acc > 0.90


def test_ovr_five_binary_classifiers_exist():
    clf = OneVsRestLogisticRegression(classes=VALID_CATEGORIES, n_iterations=50)
    rng = np.random.RandomState(42)
    X = rng.randn(25, 10)
    y = [VALID_CATEGORIES[i % 5] for i in range(25)]

    clf.fit(X, y)

    # Verify exactly 5 binary classifiers exist
    assert len(clf.classifiers) == 5
    assert len(clf.weights) == 5
    assert len(clf.biases) == 5

    for cat in VALID_CATEGORIES:
        assert cat in clf.classifiers
        assert clf.weights[cat].shape == (10,)
        assert isinstance(clf.biases[cat], float)


def test_ovr_predict_proba_normalized():
    clf = OneVsRestLogisticRegression(classes=VALID_CATEGORIES, n_iterations=50)
    rng = np.random.RandomState(42)
    X = rng.randn(15, 8)
    y = [VALID_CATEGORIES[i % 5] for i in range(15)]

    clf.fit(X, y)
    probs = clf.predict_proba(X)

    # Output dimensions (15, 5)
    assert probs.shape == (15, 5)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Probabilities across all 5 classes must sum to 1.0 for each row
    row_sums = np.sum(probs, axis=1)
    assert row_sums == pytest.approx(np.ones(15), rel=1e-5)


def test_ovr_predict_returns_valid_category():
    clf = OneVsRestLogisticRegression(classes=VALID_CATEGORIES, n_iterations=50)
    rng = np.random.RandomState(42)
    X = rng.randn(10, 6)
    y = [VALID_CATEGORIES[i % 5] for i in range(10)]

    clf.fit(X, y)
    predictions = clf.predict(X)

    assert len(predictions) == 10
    for pred in predictions:
        assert pred in VALID_CATEGORIES


def test_ovr_model_save_and_load_persistence(tmp_path):
    save_file = str(tmp_path / "test_ovr_lr.pkl")
    clf = OneVsRestLogisticRegression(classes=VALID_CATEGORIES, n_iterations=100)
    rng = np.random.RandomState(42)
    X = rng.randn(20, 5)
    y = [VALID_CATEGORIES[i % 5] for i in range(20)]

    clf.fit(X, y)
    orig_probs = clf.predict_proba(X)
    orig_preds = clf.predict(X)

    clf.save(save_file)
    assert os.path.exists(save_file)

    # Reload model
    loaded_clf = OneVsRestLogisticRegression.load(save_file)
    loaded_probs = loaded_clf.predict_proba(X)
    loaded_preds = loaded_clf.predict(X)

    assert np.allclose(orig_probs, loaded_probs, atol=1e-5)
    assert orig_preds == loaded_preds


def test_ovr_on_toy_document_vectors():
    # 5 clusters in 10D space corresponding to 5 categories
    rng = np.random.RandomState(42)
    cluster_centers = {cat: rng.randn(10) * 3.0 for cat in VALID_CATEGORIES}

    X_list, y_list = [], []
    for cat in VALID_CATEGORIES:
        samples = cluster_centers[cat] + rng.randn(20, 10) * 0.5
        X_list.append(samples)
        y_list.extend([cat] * 20)

    X = np.vstack(X_list)
    y = y_list

    clf = OneVsRestLogisticRegression(classes=VALID_CATEGORIES, learning_rate=0.5, n_iterations=300)
    clf.fit(X, y)

    acc = clf.score(X, y)
    assert acc > 0.85  # Clearly separable clusters should yield high accuracy

