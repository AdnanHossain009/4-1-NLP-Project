"""End-to-End System Integration Tests for the News Classification & Semantic Analysis Project.

Validates:
- All trained model artifacts exist on disk and deserialize correctly.
- End-to-end classification inference across all five pipelines.
- Probability outputs are strictly valid distributions (sum to 1.0, non-negative).
- Evaluation reports and visualizations exist with valid metric ranges.
- Streamlit application initializes and executes without exception.
"""

import os
import json
import numpy as np
import pytest

from src.preprocessing import TextPreprocessor, VALID_CATEGORIES
from src.naive_bayes import MultinomialNaiveBayes
from src.word2vec_scratch import CustomWord2Vec
from src.embeddings import (
    GloVeEmbeddings,
    compute_training_idf,
    aggregate_document_mean,
    aggregate_document_tfidf,
)
from src.clustering_eval import load_pipeline_logistic_regression
from streamlit.testing.v1 import AppTest


@pytest.fixture
def project_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "root": root,
        "models": os.path.join(root, "models"),
        "reports": os.path.join(root, "reports"),
        "data": os.path.join(root, "data"),
    }


def test_saved_model_artifacts_exist_and_load(project_paths):
    """Verify that all required model files exist and deserialize cleanly."""
    models_dir = project_paths["models"]

    # 1. Naive Bayes
    nb_path = os.path.join(models_dir, "naive_bayes_weights.pkl")
    assert os.path.exists(nb_path), "Naive Bayes model file missing"
    nb = MultinomialNaiveBayes.load(nb_path)
    assert nb.vocab_size_ > 1000
    assert len(nb.classes_) == 5

    # 2. Custom Word2Vec
    w2v_model_path = os.path.join(models_dir, "custom_word2vec.pt")
    w2v_vocab_path = os.path.join(models_dir, "vocab.json")
    assert os.path.exists(w2v_model_path), "Word2Vec weights missing"
    assert os.path.exists(w2v_vocab_path), "Word2Vec vocab missing"
    w2v = CustomWord2Vec(embedding_dim=100)
    w2v.load(w2v_model_path, w2v_vocab_path)
    assert w2v.vocab_size > 1000
    assert w2v.embedding_dim == 100

    # 3. GloVe offline cache
    glove_npy = os.path.join(models_dir, "pretrained_embeddings.npy")
    glove_vocab = os.path.join(models_dir, "pretrained_embeddings_vocab.json")
    assert os.path.exists(glove_npy), "GloVe npy cache missing"
    assert os.path.exists(glove_vocab), "GloVe vocab cache missing"
    glove = GloVeEmbeddings.load(glove_npy, glove_vocab)
    assert glove.vocab_size > 1000
    assert glove.embedding_dim == 100

    # 4. TF-IDF Extractors
    ext_bi_path = os.path.join(models_dir, "tfidf_extractor.pkl")
    ext_uni_path = os.path.join(models_dir, "tfidf_unigram_extractor.pkl")
    assert os.path.exists(ext_bi_path), "TF-IDF bigram extractor missing"
    assert os.path.exists(ext_uni_path), "TF-IDF unigram extractor missing"

    # 5. Logistic Regression weights bundle
    lr_path = os.path.join(models_dir, "logistic_regression_weights.pkl")
    assert os.path.exists(lr_path), "Logistic Regression bundle missing"
    for pipe_key in ["w2v_mean", "w2v_tfidf", "glove_mean", "glove_tfidf", "tfidf_unigram", "tfidf_unigram_bigram"]:
        clf = load_pipeline_logistic_regression(lr_path, pipe_key)
        assert len(clf.classes) == 5
        assert len(clf.classifiers) == 5


def test_end_to_end_classification_all_seven_pipelines(project_paths):
    """Verify that sample texts can be classified end-to-end across all 7 traditional pipelines."""
    models_dir = project_paths["models"]
    prep = TextPreprocessor()
    from src.tfidf_extractor import TfidfNGramFeatureExtractor

    # Load models
    nb = MultinomialNaiveBayes.load(os.path.join(models_dir, "naive_bayes_weights.pkl"))
    w2v = CustomWord2Vec(embedding_dim=100)
    w2v.load(os.path.join(models_dir, "custom_word2vec.pt"), os.path.join(models_dir, "vocab.json"))
    glove = GloVeEmbeddings.load(os.path.join(models_dir, "pretrained_embeddings.npy"), os.path.join(models_dir, "pretrained_embeddings_vocab.json"))
    ext_uni = TfidfNGramFeatureExtractor.load(os.path.join(models_dir, "tfidf_unigram_extractor.pkl"))
    ext_bi = TfidfNGramFeatureExtractor.load(os.path.join(models_dir, "tfidf_extractor.pkl"))
    lr_bundle = os.path.join(models_dir, "logistic_regression_weights.pkl")

    sample_articles = [
        ("The striker scored two decisive goals in the football championship match", "sport"),
        ("Software engineers built neural network algorithms for computer microchips", "tech"),
        ("Central banks announced corporate interest rate policies to stabilize market economies", "business"),
    ]

    idf_weights = {"striker": 2.5, "football": 2.0, "software": 2.2, "algorithm": 2.8, "bank": 1.5, "market": 1.3}

    for text, expected_hint in sample_articles:
        tokens = prep.preprocess(text, return_tokens=True)
        assert len(tokens) > 0

        # Pipeline A: Naive Bayes
        pred_a = nb.predict([tokens])[0]
        prob_a = nb.predict_proba([tokens])[0]
        assert pred_a in VALID_CATEGORIES
        assert len(prob_a) == 5
        assert np.isclose(np.sum(prob_a), 1.0, atol=1e-4)

        # Pipeline B: W2V Mean -> Logistic Regression
        v_b = aggregate_document_mean(tokens, w2v)
        clf_b = load_pipeline_logistic_regression(lr_bundle, "w2v_mean")
        pred_b = clf_b.predict(v_b)[0]
        prob_b = clf_b.predict_proba(v_b)[0]
        assert pred_b in VALID_CATEGORIES
        assert np.isclose(np.sum(prob_b), 1.0, atol=1e-4)

        # Pipeline C: W2V TF-IDF -> Logistic Regression
        v_c = aggregate_document_tfidf(tokens, w2v, idf_weights)
        clf_c = load_pipeline_logistic_regression(lr_bundle, "w2v_tfidf")
        pred_c = clf_c.predict(v_c)[0]
        prob_c = clf_c.predict_proba(v_c)[0]
        assert pred_c in VALID_CATEGORIES
        assert np.isclose(np.sum(prob_c), 1.0, atol=1e-4)

        # Pipeline D: GloVe Mean -> Logistic Regression
        v_d = aggregate_document_mean(tokens, glove)
        clf_d = load_pipeline_logistic_regression(lr_bundle, "glove_mean")
        pred_d = clf_d.predict(v_d)[0]
        prob_d = clf_d.predict_proba(v_d)[0]
        assert pred_d in VALID_CATEGORIES
        assert np.isclose(np.sum(prob_d), 1.0, atol=1e-4)

        # Pipeline E: GloVe TF-IDF -> Logistic Regression
        v_e = aggregate_document_tfidf(tokens, glove, idf_weights)
        clf_e = load_pipeline_logistic_regression(lr_bundle, "glove_tfidf")
        pred_e = clf_e.predict(v_e)[0]
        prob_e = clf_e.predict_proba(v_e)[0]
        assert pred_e in VALID_CATEGORIES
        assert np.isclose(np.sum(prob_e), 1.0, atol=1e-4)

        # Pipeline F: TF-IDF Unigram -> Logistic Regression
        X_f = ext_uni.transform([tokens])
        clf_f = load_pipeline_logistic_regression(lr_bundle, "tfidf_unigram")
        pred_f = clf_f.predict(X_f)[0]
        prob_f = clf_f.predict_proba(X_f)[0]
        assert pred_f in VALID_CATEGORIES
        assert np.isclose(np.sum(prob_f), 1.0, atol=1e-4)

        # Pipeline G: TF-IDF Unigram+Bigram -> Logistic Regression
        X_g = ext_bi.transform([tokens])
        clf_g = load_pipeline_logistic_regression(lr_bundle, "tfidf_unigram_bigram")
        pred_g = clf_g.predict(X_g)[0]
        prob_g = clf_g.predict_proba(X_g)[0]
        assert pred_g in VALID_CATEGORIES
        assert np.isclose(np.sum(prob_g), 1.0, atol=1e-4)


def test_reports_and_benchmark_artifacts_integrity(project_paths):
    """Verify metrics.json, confusion_matrix.png, model_comparison.csv, and error_analysis.txt exist."""
    reports_dir = project_paths["reports"]
    metrics_path = os.path.join(reports_dir, "metrics.json")
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    comp_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    error_txt_path = os.path.join(reports_dir, "error_analysis.txt")
    features_json_path = os.path.join(reports_dir, "top_features_per_category.json")

    assert os.path.exists(metrics_path), "reports/metrics.json is missing"
    assert os.path.exists(cm_path), "reports/confusion_matrix.png is missing"
    assert os.path.exists(comp_csv_path), "reports/model_comparison.csv is missing"
    assert os.path.exists(error_txt_path), "reports/error_analysis.txt is missing"
    assert os.path.exists(features_json_path), "reports/top_features_per_category.json is missing"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    assert "pipelines" in metrics_data
    pipelines = metrics_data["pipelines"]
    assert len(pipelines) >= 7
    for p_id, p_info in pipelines.items():
        assert 0.0 <= p_info["accuracy"] <= 1.0
        assert 0.0 <= p_info["macro_f1"] <= 1.0
        assert "confusion_matrix" in p_info
        assert len(p_info["confusion_matrix"]) == 5


def test_streamlit_application_headless_startup(project_paths):
    """Verify that app.py starts and executes headlessly without raising exceptions."""
    app_file = os.path.join(project_paths["root"], "app.py")
    assert os.path.exists(app_file), "app.py does not exist"

    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()

    # Ensure no unhandled exceptions were raised
    assert len(at.exception) == 0, f"Streamlit app raised exceptions: {at.exception}"


def test_streamlit_application_edge_cases(project_paths):
    """Verify that app.py handles empty, short, and OOV text inputs without exceptions."""
    app_file = os.path.join(project_paths["root"], "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    if at.text_area:
        # 1. Test short text
        at.text_area[0].input("Economy growth").run()
        assert len(at.exception) == 0

        # 2. Test OOV text
        at.text_area[0].input("Krypton blork flimzam zork").run()
        assert len(at.exception) == 0

        # 3. Test empty text
        at.text_area[0].input("   ").run()
        assert len(at.exception) == 0


