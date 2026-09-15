"""Unit tests for Part 4: Custom Word2Vec Skip-Gram Architecture."""

import os
import math
import numpy as np
import pytest
import torch
from src.word2vec_scratch import SkipGramModel, CustomWord2Vec


@pytest.fixture
def toy_tokenized_corpus():
    return [
        ["the", "football", "team", "won", "the", "match"],
        ["the", "football", "team", "lost", "the", "game"],
        ["every", "football", "player", "trains", "daily"],
        ["arsenal", "football", "club", "won", "trophy"],
    ]


def test_skipgram_model_architecture():
    vocab_size = 50
    embedding_dim = 16
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim)

    # Verify layer instances
    assert isinstance(model.embedding, torch.nn.Embedding)
    assert isinstance(model.output_layer, torch.nn.Linear)
    assert model.embedding.weight.shape == (vocab_size, embedding_dim)
    assert model.output_layer.weight.shape == (vocab_size, embedding_dim)
    assert model.output_layer.bias is None


def test_forward_pass_dimensions():
    vocab_size = 30
    embedding_dim = 8
    batch_size = 4
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim)

    center_indices = torch.tensor([0, 5, 12, 29], dtype=torch.long)
    logits = model(center_indices)

    # Output dimensions must be (batch_size, vocab_size)
    assert logits.shape == (batch_size, vocab_size)
    assert not torch.isnan(logits).any()


def test_vocab_creation(toy_tokenized_corpus):
    w2v = CustomWord2Vec(embedding_dim=10, window_size=2, min_count=2)
    vocab_dict = w2v.build_vocab(toy_tokenized_corpus)

    assert isinstance(vocab_dict, dict)
    assert len(vocab_dict) > 0
    # "football" appears in all 4 sentences, so it must be in the vocabulary
    assert "football" in vocab_dict
    assert "team" in vocab_dict
    assert w2v.vocab_size == len(vocab_dict)
    assert len(w2v.idx_to_word) == w2v.vocab_size


def test_center_context_pair_generation():
    w2v = CustomWord2Vec(embedding_dim=10, window_size=2, min_count=1)
    corpus = [["a", "b", "c", "d"]]
    w2v.build_vocab(corpus)
    pairs = w2v.generate_training_pairs(corpus)

    # For sequence ["a", "b", "c", "d"] with window=2:
    # center "a" (idx 0): context "b" (idx 1), "c" (idx 2) -> 2 pairs
    # center "b" (idx 1): context "a" (0), "c" (2), "d" (3) -> 3 pairs
    # center "c" (idx 2): context "a" (0), "b" (1), "d" (3) -> 3 pairs
    # center "d" (idx 3): context "b" (1), "c" (2) -> 2 pairs
    # Total pairs = 2 + 3 + 3 + 2 = 10 pairs
    assert len(pairs) == 10

    # Ensure all pairs are valid tuples of integer indices
    for center, context in pairs:
        assert isinstance(center, int)
        assert isinstance(context, int)
        assert center != context
        assert 0 <= center < w2v.vocab_size
        assert 0 <= context < w2v.vocab_size


def test_training_executes_without_error(toy_tokenized_corpus):
    w2v = CustomWord2Vec(embedding_dim=16, window_size=2, min_count=1, lr=0.01)
    losses = w2v.train(toy_tokenized_corpus, epochs=3, batch_size=4, verbose=False)

    assert len(losses) == 3
    for loss in losses:
        assert isinstance(loss, float)
        assert loss > 0.0
        assert not math.isnan(loss)

    # Verify embedding matrix shape
    assert w2v.embeddings is not None
    assert w2v.embeddings.shape == (w2v.vocab_size, 16)


def test_cosine_similarity_computation():
    # Identical vectors -> 1.0
    v1 = np.array([1.0, 2.0, 3.0])
    assert CustomWord2Vec.cosine_similarity(v1, v1) == pytest.approx(1.0, rel=1e-5)

    # Orthogonal vectors -> 0.0
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    assert CustomWord2Vec.cosine_similarity(v2, v3) == pytest.approx(0.0, abs=1e-6)

    # Opposite vectors -> -1.0
    assert CustomWord2Vec.cosine_similarity(v2, -v2) == pytest.approx(-1.0, rel=1e-5)

    # Zero-vector guard
    v_zero = np.zeros(3)
    assert CustomWord2Vec.cosine_similarity(v1, v_zero) == 0.0


def test_nearest_neighbors_and_oov(toy_tokenized_corpus):
    w2v = CustomWord2Vec(embedding_dim=16, window_size=2, min_count=1)
    w2v.train(toy_tokenized_corpus, epochs=2, batch_size=4, verbose=False)

    # Valid in-vocabulary query
    neighbors = w2v.find_nearest_neighbors("football", top_k=3)
    assert isinstance(neighbors, list)
    assert len(neighbors) <= 3
    for word, score in neighbors:
        assert isinstance(word, str)
        assert word != "football"  # query word must not be returned
        assert -1.0 <= score <= 1.0

    # Out-of-vocabulary query must handle gracefully without crashing
    oov_neighbors = w2v.find_nearest_neighbors("nonexistent_word_xyz", top_k=5)
    assert oov_neighbors == []


def test_model_serialization(tmp_path, toy_tokenized_corpus):
    model_path = str(tmp_path / "custom_word2vec.pt")
    vocab_path = str(tmp_path / "vocab.json")

    w2v = CustomWord2Vec(embedding_dim=12, window_size=2, min_count=1)
    w2v.train(toy_tokenized_corpus, epochs=2, batch_size=4, verbose=False)
    w2v.save(model_path=model_path, vocab_path=vocab_path)

    assert os.path.exists(model_path)
    assert os.path.exists(vocab_path)

    # Load into new instance
    w2v_loaded = CustomWord2Vec()
    w2v_loaded.load(model_path=model_path, vocab_path=vocab_path)

    assert w2v_loaded.vocab_size == w2v.vocab_size
    assert w2v_loaded.embedding_dim == 12
    assert w2v_loaded.embeddings.shape == w2v.embeddings.shape
    assert np.allclose(w2v_loaded.embeddings, w2v.embeddings, atol=1e-5)

