"""Unit tests for Part 2: Generalized N-Gram Language Model."""

import math
import pytest
from src.ngram_model import NGramLanguageModel


@pytest.fixture
def toy_corpus():
    return [
        ["the", "football", "team", "won", "the", "match"],
        ["the", "football", "team", "lost", "the", "game"],
        ["every", "football", "player", "trains", "daily"],
    ]


def test_ngram_initialization_invalid_n():
    with pytest.raises(ValueError):
        NGramLanguageModel(n=0)


def test_sentence_padding_bigram():
    model = NGramLanguageModel(n=2)
    tokens = ["hello", "world"]
    padded = model._pad_tokens(tokens)
    assert padded == ["<s>", "hello", "world", "</s>"]


def test_sentence_padding_trigram():
    model = NGramLanguageModel(n=3)
    tokens = ["the", "cat", "sat"]
    padded = model._pad_tokens(tokens)
    assert padded == ["<s>", "<s>", "the", "cat", "sat", "</s>"]


def test_sentence_padding_5gram():
    model = NGramLanguageModel(n=5)
    tokens = ["one", "two"]
    padded = model._pad_tokens(tokens)
    assert padded == ["<s>", "<s>", "<s>", "<s>", "one", "two", "</s>"]


def test_extract_ngrams():
    model = NGramLanguageModel(n=3)
    tokens = ["a", "b", "c", "d"]
    trigrams = model.extract_ngrams(tokens, n=3)
    assert trigrams == [("a", "b", "c"), ("b", "c", "d")]

    bigrams = model.extract_ngrams(tokens, n=2)
    assert bigrams == [("a", "b"), ("b", "c"), ("c", "d")]


def test_mle_probability(toy_corpus):
    # Train Bigram MLE (smoothing=False)
    model = NGramLanguageModel(n=2, laplace_smoothing=False)
    model.train(toy_corpus)

    # In toy_corpus:
    # "football" is followed by:
    # "team" (sentence 1), "team" (sentence 2), "player" (sentence 3)
    # Total context count for ("football",): 3
    # Count for ("football",) -> "team": 2
    # Count for ("football",) -> "player": 1
    p_team = model.get_probability("team", ("football",), smoothing=False)
    p_player = model.get_probability("player", ("football",), smoothing=False)
    p_won = model.get_probability("won", ("football",), smoothing=False)

    assert p_team == pytest.approx(2.0 / 3.0)
    assert p_player == pytest.approx(1.0 / 3.0)
    assert p_won == 0.0


def test_laplace_smoothing_probability_sum_to_one(toy_corpus):
    model = NGramLanguageModel(n=2, laplace_smoothing=True)
    model.train(toy_corpus)

    context = ("football",)
    v_size = model.vocab_size
    assert v_size > 0

    # The sum of smoothed probabilities across all words in vocab must equal exactly 1.0
    total_prob = sum(model.get_probability(w, context, smoothing=True) for w in model.vocab)
    assert total_prob == pytest.approx(1.0, rel=1e-5)

    # Unseen word should have non-zero probability
    unseen_prob = model.get_probability("unseen_word", context, smoothing=True)
    assert unseen_prob > 0.0


def test_trigram_language_model():
    corpus = [
        ["government", "announces", "tax", "cuts"],
        ["government", "announces", "spending", "review"],
    ]
    model = NGramLanguageModel(n=3, laplace_smoothing=True)
    model.train(corpus)

    context = ("government", "announces")
    p_tax = model.get_probability("tax", context, smoothing=True)
    p_spending = model.get_probability("spending", context, smoothing=True)
    assert p_tax == p_spending
    assert p_tax > 0.0


def test_sequence_log_likelihood_and_perplexity():
    corpus = [
        ["arsenal", "secured", "victory"],
        ["chelsea", "secured", "victory"],
    ]
    model = NGramLanguageModel(n=2, laplace_smoothing=True)
    model.train(corpus)

    test_seq = ["arsenal", "secured", "victory"]
    ll = model.calculate_log_likelihood(test_seq)
    assert isinstance(ll, float)
    assert ll < 0.0  # Log of probabilities in (0, 1) is negative

    perp = model.calculate_perplexity(test_seq)
    assert isinstance(perp, float)
    assert perp > 1.0  # Perplexity is >= 1


def test_predict_next_words(toy_corpus):
    model = NGramLanguageModel(n=2, laplace_smoothing=True)
    model.train(toy_corpus)

    # Predictions for "football"
    predictions = model.predict_next_words(("football",), top_k=3)
    assert len(predictions) == 3
    top_word, top_prob = predictions[0]

    # "team" occurred twice after "football", so it should be the top prediction
    assert top_word == "team"
    assert top_prob > 0.0

    # Verify descending ordering of probabilities
    probs = [p for _, p in predictions]
    assert probs == sorted(probs, reverse=True)


def test_higher_order_ngrams():
    # Verify 4-gram and 5-gram models work identically
    sentences = [["the", "central", "bank", "raised", "interest", "rates"]]
    for n in [4, 5]:
        model = NGramLanguageModel(n=n, laplace_smoothing=True)
        model.train(sentences)
        assert model.vocab_size > 0

        # Predict next word given appropriate context
        context = ["the", "central", "bank", "raised"][- (n - 1) :]
        preds = model.predict_next_words(context, top_k=2)
        assert len(preds) > 0
        assert preds[0][0] in model.vocab
