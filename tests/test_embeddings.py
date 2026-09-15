"""Unit and Integration Tests for Document Vector Aggregation and Offline GloVe Embeddings.

Validates:
- Correctness of GloVeEmbeddings storage, indexing, and vector lookup.
- Mathematical precision of Mean document-vector aggregation:
    X_mean = (1 / N) * sum_{i=1}^N vector(w_i)
- Mathematical precision of TF-IDF weighted document-vector aggregation:
    X_tfidf = sum [TF(w_i) * IDF(w_i) * vector(w_i)] / sum [TF(w_i) * IDF(w_i)]
- Zero Data Leakage: IDF computation strictly from training partition.
- Graceful handling of Out-Of-Vocabulary (OOV) tokens, empty inputs, and partial OOV.
- Universal compatibility with BOTH Custom Word2Vec (Step 4) and GloVe (Step 5).
- Batch corpus vectorization matching matrix dimensionality expectations.
"""

import os
import math
import pytest
import numpy as np

from src.embeddings import (
    GloVeEmbeddings,
    compute_training_idf,
    aggregate_document_mean,
    aggregate_document_tfidf,
    vectorize_corpus_mean,
    vectorize_corpus_tfidf,
)
from src.word2vec_scratch import CustomWord2Vec


@pytest.fixture
def toy_embedding_dict():
    """Toy 2-dimensional word embedding dictionary for deterministic manual proofs."""
    return {
        "apple": np.array([2.0, 4.0], dtype=np.float32),
        "banana": np.array([4.0, 6.0], dtype=np.float32),
        "cherry": np.array([6.0, 8.0], dtype=np.float32),
    }


def test_mean_aggregation_exact_math(toy_embedding_dict):
    """Verify mean aggregation against hand-calculated analytical result."""
    # Document: ["apple", "banana"]
    # Mean = ([2, 4] + [4, 6]) / 2 = [3.0, 5.0]
    doc = ["apple", "banana"]
    mean_vec = aggregate_document_mean(doc, toy_embedding_dict)

    assert mean_vec.shape == (2,)
    assert np.allclose(mean_vec, np.array([3.0, 5.0], dtype=np.float32))


def test_mean_aggregation_oov_and_empty(toy_embedding_dict):
    """Verify graceful handling of OOV words, empty documents, and mixed OOV."""
    # 1. Completely empty document -> zero vector
    empty_vec = aggregate_document_mean([], toy_embedding_dict)
    assert np.all(empty_vec == 0.0)
    assert empty_vec.shape == (2,)

    # 2. Document with only OOV words -> zero vector
    all_oov = ["alien", "martian", "ufo"]
    oov_vec = aggregate_document_mean(all_oov, toy_embedding_dict)
    assert np.all(oov_vec == 0.0)
    assert oov_vec.shape == (2,)

    # 3. Document with mixed known and OOV words -> OOV ignored, mean over valid
    mixed_doc = ["apple", "unknownword"]
    mixed_vec = aggregate_document_mean(mixed_doc, toy_embedding_dict)
    assert np.allclose(mixed_vec, np.array([2.0, 4.0], dtype=np.float32))


def test_tfidf_aggregation_exact_math(toy_embedding_dict):
    """Verify TF-IDF weighted aggregation against hand-calculated analytical result."""
    # Vocabulary vectors:
    #   apple:  [2.0, 4.0]
    #   banana: [4.0, 6.0]
    #
    # Training IDF weights:
    #   apple:  3.0
    #   banana: 1.0
    #
    # Document: ["apple", "banana", "banana"]
    # TF(apple) = 1, TF(banana) = 2
    #
    # Weight(apple)  = 1 * 3.0 = 3.0
    # Weight(banana) = 2 * 1.0 = 2.0
    # Sum of weights = 3.0 + 2.0 = 5.0
    #
    # Numerator = 3.0 * [2.0, 4.0] + 2.0 * [4.0, 6.0]
    #           = [6.0, 12.0] + [8.0, 12.0] = [14.0, 24.0]
    #
    # X_tfidf = [14.0 / 5.0, 24.0 / 5.0] = [2.8, 4.8]
    doc = ["apple", "banana", "banana"]
    idf_weights = {"apple": 3.0, "banana": 1.0}

    tfidf_vec = aggregate_document_tfidf(doc, toy_embedding_dict, idf_weights)
    assert tfidf_vec.shape == (2,)
    assert np.allclose(tfidf_vec, np.array([2.8, 4.8], dtype=np.float32))


def test_tfidf_aggregation_oov_and_empty(toy_embedding_dict):
    """Verify TF-IDF aggregation handles empty docs, pure OOV, and partial OOV."""
    idf_weights = {"apple": 2.0, "banana": 1.5}

    # Empty document -> zero vector
    vec_empty = aggregate_document_tfidf([], toy_embedding_dict, idf_weights)
    assert np.all(vec_empty == 0.0)

    # Pure OOV document -> zero vector
    vec_oov = aggregate_document_tfidf(["unknownone", "unknowntwo"], toy_embedding_dict, idf_weights)
    assert np.all(vec_oov == 0.0)

    # Partial OOV document: OOV ignored, known word used
    vec_mixed = aggregate_document_tfidf(["banana", "unknownone"], toy_embedding_dict, idf_weights)
    assert np.allclose(vec_mixed, np.array([4.0, 6.0], dtype=np.float32))


def test_compute_training_idf_zero_leakage():
    """Verify IDF formula and strict zero leakage computation on training corpus."""
    # Training corpus of 3 documents:
    # Doc 0: ["market", "share"]
    # Doc 1: ["market", "profit"]
    # Doc 2: ["market", "economy"]
    #
    # N_train = 3
    # DF("market")  = 3 (appears in all 3 docs)
    # DF("share")   = 1
    # DF("profit")  = 1
    # DF("economy") = 1
    #
    # Smooth IDF formula: ln((N + 1) / (DF + 1)) + 1
    # IDF("market") = ln((3 + 1) / (3 + 1)) + 1 = ln(1) + 1 = 1.0
    # IDF("share")  = ln((3 + 1) / (1 + 1)) + 1 = ln(2) + 1 = ~1.69315
    train_corpus = [
        ["market", "share"],
        ["market", "profit"],
        ["market", "economy"],
    ]
    idf_table, default_idf = compute_training_idf(train_corpus, smooth=True)

    assert idf_table["market"] == pytest.approx(1.0)
    assert idf_table["share"] == pytest.approx(math.log(2.0) + 1.0)
    assert idf_table["profit"] == pytest.approx(math.log(2.0) + 1.0)
    assert idf_table["economy"] == pytest.approx(math.log(2.0) + 1.0)

    # Common word "market" has strictly lower IDF than rare word "profit"
    assert idf_table["market"] < idf_table["profit"]

    # Fallback default IDF for unseen words
    assert default_idf == pytest.approx(math.log(4.0) + 1.0)


def test_glove_cache_load_and_retrieval():
    """Verify GloVeEmbeddings container loads pretrained_embeddings.npy successfully."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npy_path = os.path.join(project_root, "models", "pretrained_embeddings.npy")
    vocab_path = os.path.join(project_root, "models", "pretrained_embeddings_vocab.json")

    assert os.path.exists(npy_path), "models/pretrained_embeddings.npy must exist"
    assert os.path.exists(vocab_path), "models/pretrained_embeddings_vocab.json must exist"

    glove = GloVeEmbeddings.load(npy_path, vocab_path)
    assert glove.vocab_size > 1000
    assert glove.embedding_dim == 100

    # Retrieve valid in-vocabulary word
    sample_word = glove.idx_to_word[0]
    vec = glove.get_embedding(sample_word)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (100,)
    assert not np.isnan(vec).any()

    # Retrieve OOV word
    assert glove.get_embedding("supercalifragilistic12345") is None


def test_dual_compatibility_w2v_and_glove():
    """Verify both aggregation functions work identically with CustomWord2Vec and GloVe."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    w2v_model_path = os.path.join(project_root, "models", "custom_word2vec.pt")
    w2v_vocab_path = os.path.join(project_root, "models", "vocab.json")
    npy_path = os.path.join(project_root, "models", "pretrained_embeddings.npy")
    vocab_path = os.path.join(project_root, "models", "pretrained_embeddings_vocab.json")

    # 1. Load both models
    w2v = CustomWord2Vec(embedding_dim=100)
    w2v.load(model_path=w2v_model_path, vocab_path=w2v_vocab_path)
    glove = GloVeEmbeddings.load(npy_path, vocab_path)

    # 2. Sample sentence
    sample_doc = "The football team won the championship match after late goals"
    fake_idf = {"football": 2.5, "team": 1.2, "won": 1.8, "championship": 3.0, "match": 2.0, "goal": 2.2}

    # Mean Aggregation on both
    w2v_mean = aggregate_document_mean(sample_doc, w2v)
    glove_mean = aggregate_document_mean(sample_doc, glove)

    assert w2v_mean.shape == (100,)
    assert glove_mean.shape == (100,)
    assert np.linalg.norm(w2v_mean) > 0.0
    assert np.linalg.norm(glove_mean) > 0.0

    # TF-IDF Aggregation on both
    w2v_tfidf = aggregate_document_tfidf(sample_doc, w2v, fake_idf)
    glove_tfidf = aggregate_document_tfidf(sample_doc, glove, fake_idf)

    assert w2v_tfidf.shape == (100,)
    assert glove_tfidf.shape == (100,)
    assert np.linalg.norm(w2v_tfidf) > 0.0
    assert np.linalg.norm(glove_tfidf) > 0.0


def test_batch_corpus_vectorization():
    """Verify vectorize_corpus_mean and vectorize_corpus_tfidf create correct 2D matrices."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npy_path = os.path.join(project_root, "models", "pretrained_embeddings.npy")
    vocab_path = os.path.join(project_root, "models", "pretrained_embeddings_vocab.json")
    glove = GloVeEmbeddings.load(npy_path, vocab_path)

    corpus = [
        "Economy growth accelerates as central bank weighs rate cuts",
        "Film festival honors independent cinema with awards",
        "Striker scored goal in premier league championship match",
    ]
    idf_weights = {"economy": 2.0, "bank": 1.5, "film": 2.2, "goal": 2.1}

    # Vectorize batch via mean
    X_mean = vectorize_corpus_mean(corpus, glove)
    assert X_mean.shape == (3, 100)
    assert isinstance(X_mean, np.ndarray)

    # Vectorize batch via TF-IDF
    X_tfidf = vectorize_corpus_tfidf(corpus, glove, idf_weights)
    assert X_tfidf.shape == (3, 100)
    assert isinstance(X_tfidf, np.ndarray)

