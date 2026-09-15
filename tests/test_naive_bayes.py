"""Unit and Integration Tests for Multinomial Naive Bayes Classifier (from Scratch).

Validates:
- Mathematical correctness of Class Prior P(c) = N_c / N
- Add-1 (Laplace) smoothed word likelihoods:
    P(w|c) = (count(w,c) + 1) / (total words in class c + |V|)
- Sum of probabilities over vocabulary per class equals 1.0
- Document scoring in log space: log P(c|D) = log P(c) + sum log P(w_i|c)
- Normalized probabilities via Log-Sum-Exp in [0, 1] summing to 1.0
- Out-of-vocabulary (OOV) and empty document robustness
- Save and load serialization persistence
- End-to-end training and evaluation on preprocessed BBC News training split.
"""

import os
import math
import numpy as np
import pytest

from src.naive_bayes import BagOfWords, MultinomialNaiveBayes, logsumexp
from src.preprocessing import TextPreprocessor, load_dataset, stratified_train_test_split


@pytest.fixture
def toy_corpus():
    """Deterministic, manually verifiable 2-class toy corpus."""
    documents = [
        ["football", "match", "goal", "victory"],      # Doc 0: sport (4 words)
        ["team", "football", "game"],                  # Doc 1: sport (3 words)
        ["computer", "software", "code"],              # Doc 2: tech  (3 words)
        ["software", "code", "computer", "algorithm"], # Doc 3: tech  (4 words)
    ]
    labels = ["sport", "sport", "tech", "tech"]
    return documents, labels


def test_bag_of_words_vectorizer(toy_corpus):
    """Verify BagOfWords vectorizer builds correct vocabulary and count matrices."""
    docs, _ = toy_corpus
    bow = BagOfWords()
    X = bow.fit_transform(docs)

    # 10 unique tokens across the 4 documents
    expected_vocab = sorted([
        "algorithm", "code", "computer", "football",
        "game", "goal", "match", "software", "team", "victory"
    ])
    assert sorted(list(bow.vocab.keys())) == expected_vocab
    assert bow.vocab_size == 10
    assert X.shape == (4, 10)

    # Total word count across all docs = 4 + 3 + 3 + 4 = 14
    assert np.sum(X) == 14

    # Check specific counts
    football_idx = bow.vocab["football"]
    assert X[0, football_idx] == 1
    assert X[1, football_idx] == 1
    assert X[2, football_idx] == 0


def test_prior_probabilities_toy(toy_corpus):
    """Verify Class Prior formula P(c) = N_c / N on balanced and imbalanced sets."""
    docs, labels = toy_corpus
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(docs, labels)

    # 2 sport, 2 tech -> Total = 4 -> P(sport) = 0.5, P(tech) = 0.5
    assert model.total_docs_ == 4
    assert len(model.classes_) == 2
    assert "sport" in model.class_to_idx_
    assert "tech" in model.class_to_idx_

    sport_idx = model.class_to_idx_["sport"]
    tech_idx = model.class_to_idx_["tech"]

    assert model.class_priors_[sport_idx] == pytest.approx(0.5)
    assert model.class_priors_[tech_idx] == pytest.approx(0.5)
    assert model.log_class_priors_[sport_idx] == pytest.approx(math.log(0.5))
    assert model.log_class_priors_[tech_idx] == pytest.approx(math.log(0.5))

    # Test imbalanced priors: 3 sport, 1 tech
    imbalanced_docs = docs[:3] + [["football", "stadium"]]
    imbalanced_labels = ["sport", "sport", "tech", "sport"]
    model_imb = MultinomialNaiveBayes(alpha=1.0)
    model_imb.fit(imbalanced_docs, imbalanced_labels)

    s_idx = model_imb.class_to_idx_["sport"]
    t_idx = model_imb.class_to_idx_["tech"]
    assert model_imb.class_priors_[s_idx] == pytest.approx(0.75)
    assert model_imb.class_priors_[t_idx] == pytest.approx(0.25)


def test_add_one_smoothing_word_likelihood_toy(toy_corpus):
    """Verify word likelihood with Laplace (Add-1) smoothing:
    P(w|c) = (count(w,c) + 1) / (total words in class c + |V|)
    """
    docs, labels = toy_corpus
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(docs, labels)

    sport_idx = model.class_to_idx_["sport"]
    tech_idx = model.class_to_idx_["tech"]

    # sport total words: 4 + 3 = 7 words
    # tech total words: 3 + 4 = 7 words
    # |V| = 10
    # Smoothed denominator for both classes = 7 + 10 = 17
    assert model.class_word_totals_[sport_idx] == 7
    assert model.class_word_totals_[tech_idx] == 7
    assert model.vocab_size_ == 10

    # "football" appears 2 times in sport, 0 times in tech
    football_idx = model.vocab_["football"]
    expected_p_football_sport = (2.0 + 1.0) / (7.0 + 10.0)  # 3 / 17
    expected_p_football_tech = (0.0 + 1.0) / (7.0 + 10.0)   # 1 / 17

    # Exponentiate stored log likelihoods to check exact probabilities
    p_football_sport = math.exp(model.feature_log_prob_[sport_idx, football_idx])
    p_football_tech = math.exp(model.feature_log_prob_[tech_idx, football_idx])

    assert p_football_sport == pytest.approx(expected_p_football_sport)
    assert p_football_tech == pytest.approx(expected_p_football_tech)

    # Sum of likelihoods across all words in vocabulary must equal 1.0 for each class
    prob_matrix = np.exp(model.feature_log_prob_)
    assert np.sum(prob_matrix[sport_idx, :]) == pytest.approx(1.0, rel=1e-5)
    assert np.sum(prob_matrix[tech_idx, :]) == pytest.approx(1.0, rel=1e-5)


def test_log_space_scoring_and_logsumexp_normalization(toy_corpus):
    """Verify document scoring in log space and exact normalized probability via log-sum-exp."""
    docs, labels = toy_corpus
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(docs, labels)

    sport_idx = model.class_to_idx_["sport"]
    tech_idx = model.class_to_idx_["tech"]

    # Test document: ["football", "match"]
    # In sport:
    # count("football") = 2 -> P = 3/17
    # count("match") = 1 -> P = 2/17
    # log_score(sport) = log(0.5) + log(3/17) + log(2/17) = log(0.5 * 6 / 289)
    #
    # In tech:
    # count("football") = 0 -> P = 1/17
    # count("match") = 0 -> P = 1/17
    # log_score(tech) = log(0.5) + log(1/17) + log(1/17) = log(0.5 * 1 / 289)
    test_doc = [["football", "match"]]
    log_scores = model.predict_log_proba(test_doc)[0]

    expected_log_sport = math.log(0.5) + math.log(3.0 / 17.0) + math.log(2.0 / 17.0)
    expected_log_tech = math.log(0.5) + math.log(1.0 / 17.0) + math.log(1.0 / 17.0)

    assert log_scores[sport_idx] == pytest.approx(expected_log_sport)
    assert log_scores[tech_idx] == pytest.approx(expected_log_tech)

    # Normalized probabilities by Bayes theorem:
    # P(sport | D) = (3 * 2) / (3 * 2 + 1 * 1) = 6 / 7
    # P(tech | D) = (1 * 1) / (3 * 2 + 1 * 1) = 1 / 7
    expected_prob_sport = 6.0 / 7.0
    expected_prob_tech = 1.0 / 7.0

    probs = model.predict_proba(test_doc)[0]
    assert probs[sport_idx] == pytest.approx(expected_prob_sport)
    assert probs[tech_idx] == pytest.approx(expected_prob_tech)

    # Probabilities must be strictly between 0 and 1, and sum to 1.0
    assert 0.0 <= probs[sport_idx] <= 1.0
    assert 0.0 <= probs[tech_idx] <= 1.0
    assert np.sum(probs) == pytest.approx(1.0)


def test_toy_predictions(toy_corpus):
    """Verify class predictions on distinct sports and technology queries."""
    docs, labels = toy_corpus
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(docs, labels)

    queries = [
        ["football", "team", "victory"],      # Should predict "sport"
        ["computer", "algorithm", "software"], # Should predict "tech"
    ]
    preds = model.predict(queries)
    assert preds[0] == "sport"
    assert preds[1] == "tech"


def test_out_of_vocabulary_words_handling(toy_corpus):
    """Verify that unseen words do not cause crashes and leave predictions valid."""
    docs, labels = toy_corpus
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(docs, labels)

    # Query contains known word "football" plus unknown words "alien", "galaxy"
    oov_query = [["football", "alien", "galaxy"]]
    probs = model.predict_proba(oov_query)[0]
    pred = model.predict(oov_query)[0]

    assert pred == "sport"
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0
    assert np.sum(probs) == pytest.approx(1.0)


def test_empty_or_all_oov_document(toy_corpus):
    """Verify behavior on an empty document or a document with only OOV words.
    When all words are OOV, log likelihood sum is 0, so posteriors equal class priors.
    """
    docs, labels = toy_corpus
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(docs, labels)

    all_oov = [["unknownwordone", "unknownwordtwo"]]
    probs = model.predict_proba(all_oov)[0]

    # Posteriors must fall back exactly to class priors (0.5, 0.5)
    sport_idx = model.class_to_idx_["sport"]
    tech_idx = model.class_to_idx_["tech"]
    assert probs[sport_idx] == pytest.approx(0.5)
    assert probs[tech_idx] == pytest.approx(0.5)
    assert np.sum(probs) == pytest.approx(1.0)


def test_model_save_and_load_persistence(toy_corpus, tmp_path):
    """Verify model weights can be serialized to disk and restored without loss."""
    docs, labels = toy_corpus
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(docs, labels)

    save_path = str(tmp_path / "test_nb_weights.pkl")
    model.save(save_path)
    assert os.path.exists(save_path)

    loaded_model = MultinomialNaiveBayes.load(save_path)
    assert loaded_model.alpha == model.alpha
    assert np.array_equal(loaded_model.classes_, model.classes_)
    assert np.allclose(loaded_model.class_priors_, model.class_priors_)
    assert np.allclose(loaded_model.feature_log_prob_, model.feature_log_prob_)

    # Verify identical prediction and probability outputs
    test_query = [["computer", "code"]]
    orig_prob = model.predict_proba(test_query)
    loaded_prob = loaded_model.predict_proba(test_query)
    assert np.allclose(orig_prob, loaded_prob)


def test_logsumexp_numerical_stability():
    """Verify logsumexp handles extreme positive and negative log values without underflow/overflow."""
    # Large magnitude scores that would overflow standard exp(s)
    scores = np.array([[-1000.0, -1002.0], [500.0, 502.0]])
    lse = logsumexp(scores, axis=1, keepdims=True)
    assert not np.isnan(lse).any()
    assert not np.isinf(lse).any()

    # Probability after normalization
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), [1.0, 1.0])


def test_naive_bayes_on_bbc_train_split():
    """Integration test: Train on the real BBC preprocessed training split and verify test accuracy."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "bbc_news.csv")
    if not os.path.exists(data_path):
        pytest.skip(f"BBC News dataset not found at {data_path}")

    df = load_dataset(data_path)
    train_df, test_df = stratified_train_test_split(df, test_size=0.2, random_state=42)

    preprocessor = TextPreprocessor()
    train_tokens = [preprocessor.preprocess(t, return_tokens=True) for t in train_df["text"]]
    test_tokens = [preprocessor.preprocess(t, return_tokens=True) for t in test_df["text"]]

    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(train_tokens, train_df["category"].values)

    # 1. Model trains without error
    assert model.vocab_size_ > 10000
    assert len(model.classes_) == 5
    expected_categories = {"business", "entertainment", "politics", "sport", "tech"}
    assert set(model.classes_) == expected_categories

    # 2. Training and Test Accuracy are high (> 95% train, > 90% test)
    train_acc = model.score(train_tokens, train_df["category"].values)
    test_acc = model.score(test_tokens, test_df["category"].values)
    assert train_acc >= 0.95
    assert test_acc >= 0.90

    # 3. Predictions are valid category labels
    test_sample = test_tokens[:20]
    preds = model.predict(test_sample)
    assert len(preds) == 20
    for p in preds:
        assert p in expected_categories

    # 4. Probabilities are strictly between 0 and 1, and sum to 1.0
    probs = model.predict_proba(test_sample)
    assert probs.shape == (20, 5)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_saved_bbc_weights_file():
    """Verify that models/naive_bayes_weights.pkl exists and can make realistic predictions."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "naive_bayes_weights.pkl")

    if not os.path.exists(model_path):
        pytest.skip("models/naive_bayes_weights.pkl not yet generated.")

    model = MultinomialNaiveBayes.load(model_path)
    assert model.vocab_size_ > 10000
    assert len(model.classes_) == 5

    # Test realistic raw sentence inputs
    sample_texts = [
        "The football team won the championship match after a late goal in injury time",
        "Prime minister delivered a speech in parliament discussing the government budget",
        "Software engineers developed a new algorithm using neural network computer chips",
    ]
    preprocessor = TextPreprocessor()
    tokenized = [preprocessor.preprocess(t, return_tokens=True) for t in sample_texts]

    preds = model.predict(tokenized)
    assert preds[0] == "sport"
    assert preds[1] == "politics"
    assert preds[2] == "tech"

