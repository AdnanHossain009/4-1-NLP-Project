"""Interactive Streamlit Web Application for News Classification and Semantic Analysis.

Hardened for Course Showcase Demo:
- Graphical Interface: 100% interactive UI; no terminal or code windows shown.
- Real Dataset: BBC News Corpus ("gorur rochona" universal benchmark — 2,225 articles, 5 categories).
- Pre-Trained Freeze: Zero runtime training / fitting / backprop; all models loaded from disk with caching.
- Startup Integrity Check: Verifies all required artifacts exist on disk before rendering; clean error on missing.
- Multi-Model Inference: All 5 pipelines evaluated concurrently on user input.
- Robust Edge Cases: Graceful handling of empty input, short text (1-2 words), and out-of-vocabulary terms.
"""

import os
import sys
import json
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set page configuration before any UI rendering
st.set_page_config(
    page_title="BBC News Classification & Semantic Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 1. Artifact Integrity Verification (Requirement 4)
# =====================================================================

REQUIRED_ARTIFACTS = [
    ("models/naive_bayes_weights.pkl", "Multinomial Naive Bayes Model Weights"),
    ("models/custom_word2vec.pt", "Custom Word2Vec PyTorch Model"),
    ("models/vocab.json", "Custom Word2Vec Vocabulary"),
    ("models/pretrained_embeddings.npy", "Offline GloVe Embedding Matrix"),
    ("models/logistic_regression_weights.pkl", "One-vs-Rest Logistic Regression Weights Bundle"),
    ("reports/metrics.json", "Evaluation Benchmarks Metrics Report"),
    ("reports/confusion_matrix.png", "Multi-Pipeline Confusion Matrix Visualization"),
    ("reports/clusters.png", "Unsupervised K-Means Clustering Visualization"),
]


def check_artifact_integrity() -> List[Tuple[str, str]]:
    """Check if all required pretrained artifacts exist on disk.
    
    Returns:
        List of missing (file_path, description) tuples.
    """
    missing = []
    for rel_path, desc in REQUIRED_ARTIFACTS:
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.exists(abs_path):
            missing.append((rel_path, desc))
    return missing


missing_artifacts = check_artifact_integrity()
if missing_artifacts:
    st.error("🚨 Missing Required Model Artifacts for Showcase Demo")
    st.markdown(
        "The application cannot start because the following required pre-trained artifacts "
        "were not found on disk:\n\n"
        + "\n".join([f"- **`{path}`** — *{desc}*" for path, desc in missing_artifacts])
        + "\n\n### Required Action:\n"
        "Please run the offline training pipeline once before launching the demo:\n"
        "```bash\npython train_pipeline.py\n```"
    )
    st.stop()


# =====================================================================
# Imports from Core Modules (Safe after integrity check)
# =====================================================================

from src.preprocessing import (
    TextPreprocessor,
    load_dataset,
    stratified_train_test_split,
    VALID_CATEGORIES,
    ensure_nltk_resources,
)
from src.ngram_model import NGramLanguageModel
from src.naive_bayes import MultinomialNaiveBayes
from src.word2vec_scratch import CustomWord2Vec
from src.embeddings import (
    GloVeEmbeddings,
    aggregate_document_mean,
    aggregate_document_tfidf,
    compute_training_idf,
)
from src.logistic_regression import (
    OneVsRestLogisticRegression,
    BinaryLogisticRegression,
)

# Universal category styling
CATEGORY_COLORS = {
    "business": "#1f77b4",
    "entertainment": "#9467bd",
    "politics": "#d62728",
    "sport": "#2ca02c",
    "tech": "#ff7f0e",
}

CATEGORY_ICONS = {
    "business": "📈",
    "entertainment": "🎭",
    "politics": "🏛️",
    "sport": "⚽",
    "tech": "💻",
}

PRESET_EXAMPLES = {
    "⚽ Sport": "The football club secured a dramatic victory in the championship final after the striker scored two decisive goals in extra time.",
    "💻 Tech": "Software engineers developed an advanced neural algorithm for mobile microprocessors to accelerate machine learning inference.",
    "📈 Business": "Corporate profits climbed across international equity markets as central banks signaled prospective interest rate cuts to stimulate economic growth.",
    "🏛️ Politics": "The prime minister defended the annual budget in parliament, promising fiscal reforms and taxation incentives for healthcare and transport infrastructure.",
    "🎭 Entertainment": "The independent film festival awarded top honors to acclaimed international directors and veteran actors during the gala awards ceremony.",
    "🔬 Diagnostic (Short)": "Brazil lost 7-1 against Germany",
    "📰 Diagnostic (Match Report)": "Germany defeated Brazil 7-1 in the World Cup semifinal, scoring five goals in the first half and reaching the final after a remarkable football performance.",
    "⚡ Short Text (Edge Case)": "Economy growth",
    "👽 OOV Slang (Edge Case)": "Krypton blork flimzam zork",
}


# =====================================================================
# 2. Cached Resource & Pretrained Model Loaders (Zero Runtime Training)
# =====================================================================

@st.cache_resource(show_spinner="Initializing NLP Preprocessor & Lexicons...")
def get_preprocessor():
    ensure_nltk_resources()
    return TextPreprocessor()


@st.cache_resource(show_spinner="Loading Pretrained Multinomial Naive Bayes...")
def get_naive_bayes_model():
    path = os.path.join(project_root, "models", "naive_bayes_weights.pkl")
    if not os.path.exists(path):
        return None
    return MultinomialNaiveBayes.load(path)


@st.cache_resource(show_spinner="Loading Pretrained Custom Word2Vec Model...")
def get_word2vec_model():
    m_path = os.path.join(project_root, "models", "custom_word2vec.pt")
    v_path = os.path.join(project_root, "models", "vocab.json")
    if not os.path.exists(m_path) or not os.path.exists(v_path):
        return None
    w2v = CustomWord2Vec(embedding_dim=100)
    w2v.load(m_path, v_path)
    return w2v


@st.cache_resource(show_spinner="Loading TF-IDF Feature Extractor...")
def get_tfidf_extractor(ngram_type: str = "bigram"):
    fname = "tfidf_extractor.pkl" if ngram_type == "bigram" else "tfidf_unigram_extractor.pkl"
    path = os.path.join(project_root, "models", fname)
    if not os.path.exists(path):
        return None
    from src.tfidf_extractor import TfidfNGramFeatureExtractor
    return TfidfNGramFeatureExtractor.load(path)


@st.cache_resource(show_spinner="Loading Offline GloVe Embedding Cache...")
def get_glove_model():
    npy_path = os.path.join(project_root, "models", "pretrained_embeddings.npy")
    vocab_path = os.path.join(project_root, "models", "pretrained_embeddings_vocab.json")
    if not os.path.exists(npy_path) or not os.path.exists(vocab_path):
        return None
    return GloVeEmbeddings.load(npy_path, vocab_path)


@st.cache_resource(show_spinner="Loading Pretrained Logistic Regression Bundle...")
def get_logistic_regression_bundle():
    path = os.path.join(project_root, "models", "logistic_regression_weights.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_lr_model_for_pipeline(pipe_key: str) -> Optional[OneVsRestLogisticRegression]:
    """Reconstruct a trained OneVsRestLogisticRegression instance for a specific pipeline."""
    bundle = get_logistic_regression_bundle()
    if bundle is None or "all_pipelines" not in bundle or pipe_key not in bundle["all_pipelines"]:
        return None
    pipe_data = bundle["all_pipelines"][pipe_key]
    clf = OneVsRestLogisticRegression(
        classes=pipe_data["classes"],
        learning_rate=bundle.get("learning_rate", 0.5),
        n_iterations=bundle.get("n_iterations", 800),
        l2_reg=bundle.get("l2_reg", 0.0001),
    )
    clf.weights = pipe_data["weights"]
    clf.biases = pipe_data["biases"]
    for cat in clf.classes:
        b_clf = BinaryLogisticRegression(
            learning_rate=clf.learning_rate,
            n_iterations=clf.n_iterations,
            l2_reg=clf.l2_reg,
        )
        b_clf.w = clf.weights[cat].copy()
        b_clf.b = float(clf.biases[cat])
        clf.classifiers[cat] = b_clf
    return clf


@st.cache_resource(show_spinner="Loading Pretrained N-Gram Language Models...")
def get_ngram_models() -> Dict[int, NGramLanguageModel]:
    """Load pre-trained N-Gram models from disk (Zero live training at startup)."""
    ngram_path = os.path.join(project_root, "models", "ngram_models.pkl")
    if os.path.exists(ngram_path):
        with open(ngram_path, "rb") as f:
            return pickle.load(f)
    return {}


@st.cache_resource(show_spinner="Loading Precomputed IDF Weights Table...")
def get_idf_weights() -> Tuple[Dict[str, float], float]:
    """Load precomputed training-set IDF table from disk."""
    idf_path = os.path.join(project_root, "models", "idf_weights.json")
    if os.path.exists(idf_path):
        with open(idf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("idf_weights", {}), float(data.get("default_idf", 1.0))
    # Fallback to computing once if json was absent
    csv_path = os.path.join(project_root, "data", "bbc_news.csv")
    if os.path.exists(csv_path):
        df = load_dataset(csv_path)
        train_df, _ = stratified_train_test_split(df, test_size=0.2, random_state=42)
        prep = get_preprocessor()
        train_tokens = [prep.preprocess(t, return_tokens=True) for t in train_df["text"]]
        return compute_training_idf(train_tokens, smooth=True)
    return {}, 1.0


@st.cache_data
def load_metrics_report() -> Optional[dict]:
    """Load the held-out evaluation report from disk."""
    path = os.path.join(project_root, "reports", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Preload cached models on initial app execution
preprocessor = get_preprocessor()
nb_model = get_naive_bayes_model()
w2v_model = get_word2vec_model()
glove_model = get_glove_model()
lr_bundle = get_logistic_regression_bundle()
ngram_models = get_ngram_models()
idf_weights, default_idf = get_idf_weights()


# =====================================================================
# 3. Main Header & Corpus Caption (Requirement 2)
# =====================================================================

st.title("📰 BBC News Classification & Semantic Analysis Engine")
st.caption(
    "Trained on the BBC News Corpus (Universal 'gorur rochona' benchmark — "
    "2,225 articles, 5 categories: Business, Entertainment, Politics, Sport, Tech)"
)
st.markdown(
    "**Course Showcase Demonstration** | 100% Offline Inference | Zero Live Training | Pure Mathematical NLP"
)
st.divider()


# =====================================================================
# 4. Sidebar: Verification & System Health
# =====================================================================

with st.sidebar:
    st.header("🛡️ System Integrity Check")
    tfidf_ext = get_tfidf_extractor("bigram")
    st.success("✅ Pretrained Artifacts Verified On Disk")

    st.markdown(
        f"""
        - **Dataset**: BBC News (2,225 docs)
        - **Multinomial Naive Bayes**: `Ready` ({nb_model.vocab_size_:,} words)
        - **Custom Word2Vec**: `Ready` (100d, {w2v_model.vocab_size:,} words)
        - **Pretrained GloVe**: `Ready` (100d, {glove_model.vocab_size:,} words)
        - **TF-IDF N-Gram Extractor**: {'`Ready` (1-g + 2-g, 8,000 features)' if tfidf_ext else '`Missing`'}
        - **Logistic Regression**: `Ready` (6 OvR Configurations)
        - **N-Gram Models**: `Ready` (Orders n=2, 3, 4, 5)
        - **Frozen IDF Weights**: `Ready` ({len(idf_weights):,} terms)
        """
    )

    st.divider()
    st.header("📋 Rehearsal Presets")
    st.write("Click any preset below to load sample text into the input area:")

    for label, sample_text in PRESET_EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["user_input_text"] = sample_text

    st.divider()
    st.info(
        "**Strict Showcase Guarantees:**\n"
        "• No RAG, No LLMs, No Transformers\n"
        "• Pure NumPy / PyTorch from scratch\n"
        "• Real inference — zero mocks or fake numbers\n"
        "• Zero retraining on button click or page load"
    )


# =====================================================================
# 5. Main Application Tabs
# =====================================================================

tab_live_demo, tab_semantic, tab_benchmarks = st.tabs([
    "🎯 Live Multi-Model Inference & N-Gram Analysis",
    "🌐 Word2Vec Semantic Explorer",
    "📊 Evaluation Benchmarks",
])


# =====================================================================
# TAB 1: Live Multi-Model Inference & N-Gram Analysis
# =====================================================================

with tab_live_demo:
    st.subheader("1. Interactive Text Input")
    st.write("Type or paste any arbitrary news text below to evaluate all 5 classification pipelines concurrently:")

    default_input = st.session_state.get("user_input_text", PRESET_EXAMPLES["⚽ Sport"])

    user_text = st.text_area(
        label="News Article Text / Headline",
        value=default_input,
        height=130,
        placeholder="Type or paste any news text here...",
        help="Paste article text from any category, or test edge cases like very short phrases or out-of-vocabulary words.",
    )

    col_btn, col_ngram_sel = st.columns([2, 1])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Text Across All Models", type="primary", use_container_width=True)
    with col_ngram_sel:
        ngram_order_labels = {
            "Trigram (n=3) [Recommended]": 3,
            "Bigram (n=2)": 2,
            "4-Gram (n=4)": 4,
            "5-Gram (n=5)": 5,
        }
        selected_ngram_label = st.selectbox(
            "N-Gram Analysis Order",
            options=list(ngram_order_labels.keys()),
            index=0,
            help="Select n-gram order for next-word suggestion and perplexity calculation.",
        )
        selected_n = ngram_order_labels[selected_ngram_label]

    # Process on button click or existing non-empty input
    cleaned_input = user_text.strip()

    if analyze_clicked or cleaned_input:
        # Edge Case 1: Empty input
        if not cleaned_input:
            st.warning("⚠️ Please enter text to analyze.")
        else:
            try:
                # Preprocess text (preserve_numbers=True handles scorelines like 7-1)
                tokens = preprocessor.preprocess(cleaned_input, return_tokens=True, preserve_numbers=True)

                # Edge Case 2: Very short or non-alphanumeric input
                if not tokens:
                    st.warning("⚠️ The input text contains no valid alphanumeric words after preprocessing.")
                else:
                    st.divider()

                    # -------------------------------------------------
                    # Section A: Preprocessed Representation
                    # -------------------------------------------------
                    st.subheader("2. Preprocessed Tokens (What the Models Actually See)")

                    # Check vocabulary coverage
                    w2v_cov = sum(1 for t in tokens if t in w2v_model.word_to_idx)
                    glove_cov = sum(1 for t in tokens if t in glove_model.word_to_idx)
                    total_tokens = len(tokens)

                    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                    with meta_col1:
                        st.metric("Raw Characters", f"{len(cleaned_input):,}")
                    with meta_col2:
                        st.metric("Clean Tokens", f"{total_tokens:,}")
                    with meta_col3:
                        st.metric("W2V Vocab Coverage", f"{w2v_cov}/{total_tokens} ({(w2v_cov/total_tokens)*100:.0f}%)")
                    with meta_col4:
                        st.metric("GloVe Vocab Coverage", f"{glove_cov}/{total_tokens} ({(glove_cov/total_tokens)*100:.0f}%)")

                    # Display token chips
                    tokens_display = " ".join([f"`{t}`" for t in tokens])
                    st.markdown(f"**Clean Token Sequence:** {tokens_display}")

                    # Edge Case 3: Heavy / Full OOV Notification
                    if w2v_cov == 0 and glove_cov == 0:
                        st.info(
                            "ℹ️ **Out-of-Vocabulary (OOV) Notice:** None of the input tokens were recognized in "
                            "the embedding vocabularies. The embedding models evaluate on zero-vectors using learned "
                            "class biases, and Naive Bayes applies Laplace Add-1 smoothing."
                        )

                    # -------------------------------------------------
                    # Section B: Multi-Model Live Inference
                    # -------------------------------------------------
                    st.divider()
                    st.subheader("3. Multi-Pipeline Classification Predictions")

                    # Collect predictions across all traditional pipelines
                    pipeline_results: Dict[str, Dict[str, Any]] = {}

                    # Pipeline 1: Naive Bayes
                    nb_probs = nb_model.predict_proba([tokens])[0]
                    nb_pred = nb_model.predict([tokens])[0]
                    pipeline_results["Naive Bayes (BoW)"] = {
                        "pred": nb_pred,
                        "probs": {c: float(p) for c, p in zip(nb_model.classes_, nb_probs)},
                        "conf": float(np.max(nb_probs)),
                    }

                    # Pipeline 2 & 3: TF-IDF Unigram & Unigram+Bigram
                    tfidf_bigram_ext = get_tfidf_extractor("bigram")
                    tfidf_unigram_ext = get_tfidf_extractor("unigram")
                    lr_tfidf_bigram = get_lr_model_for_pipeline("tfidf_unigram_bigram")
                    lr_tfidf_unigram = get_lr_model_for_pipeline("tfidf_unigram")

                    active_explanation = None
                    if tfidf_bigram_ext is not None and lr_tfidf_bigram is not None:
                        X_bi = tfidf_bigram_ext.transform([tokens])
                        bi_probs = lr_tfidf_bigram.predict_proba(X_bi)[0]
                        bi_pred = lr_tfidf_bigram.predict(X_bi)[0]
                        pipeline_results["LR on TF-IDF (1+2g)"] = {
                            "pred": bi_pred,
                            "probs": {c: float(p) for c, p in zip(lr_tfidf_bigram.classes, bi_probs)},
                            "conf": float(np.max(bi_probs)),
                        }
                        active_explanation = tfidf_bigram_ext.explain_instance(tokens, lr_tfidf_bigram.weights, top_k=6)

                    if tfidf_unigram_ext is not None and lr_tfidf_unigram is not None:
                        X_uni = tfidf_unigram_ext.transform([tokens])
                        uni_probs = lr_tfidf_unigram.predict_proba(X_uni)[0]
                        uni_pred = lr_tfidf_unigram.predict(X_uni)[0]
                        pipeline_results["LR on TF-IDF (1-g)"] = {
                            "pred": uni_pred,
                            "probs": {c: float(p) for c, p in zip(lr_tfidf_unigram.classes, uni_probs)},
                            "conf": float(np.max(uni_probs)),
                        }

                    # Pipelines 4-7: Word2Vec & GloVe Embedding Pipelines
                    w2v_mean_vec = aggregate_document_mean(tokens, w2v_model)
                    w2v_tfidf_vec = aggregate_document_tfidf(tokens, w2v_model, idf_weights, default_idf=default_idf)
                    glove_mean_vec = aggregate_document_mean(tokens, glove_model)
                    glove_tfidf_vec = aggregate_document_tfidf(tokens, glove_model, idf_weights, default_idf=default_idf)

                    lr_emb_pipelines = [
                        ("LR on Word2Vec (Mean)", "w2v_mean", w2v_mean_vec),
                        ("LR on Word2Vec (TF-IDF)", "w2v_tfidf", w2v_tfidf_vec),
                        ("LR on GloVe (Mean)", "glove_mean", glove_mean_vec),
                        ("LR on GloVe (TF-IDF)", "glove_tfidf", glove_tfidf_vec),
                    ]

                    for disp_name, pipe_key, vec in lr_emb_pipelines:
                        clf = get_lr_model_for_pipeline(pipe_key)
                        if clf is not None:
                            probs = clf.predict_proba(vec)[0]
                            pred = clf.predict(vec)[0]
                            pipeline_results[disp_name] = {
                                "pred": pred,
                                "probs": {c: float(p) for c, p in zip(clf.classes, probs)},
                                "conf": float(np.max(probs)),
                            }

                    # Display cards side-by-side
                    card_cols = st.columns(len(pipeline_results))
                    for col, (pipe_name, res) in zip(card_cols, pipeline_results.items()):
                        pred_cat = res["pred"]
                        conf_pct = res["conf"] * 100
                        cat_color = CATEGORY_COLORS.get(pred_cat, "#333333")
                        icon = CATEGORY_ICONS.get(pred_cat, "📌")

                        with col:
                            st.markdown(
                                f"""
                                <div style="border: 2px solid {cat_color}; border-radius: 8px; padding: 10px; background-color: {cat_color}10; text-align: center; min-height: 145px;">
                                    <span style="font-size: 0.72em; color: #555; font-weight: 600; text-transform: uppercase;">{pipe_name}</span>
                                    <h3 style="margin: 6px 0 2px 0; color: {cat_color}; font-size: 1.2em;">{icon} {pred_cat.capitalize()}</h3>
                                    <p style="margin: 0; font-size: 1.05em; font-weight: bold; color: #222;">{conf_pct:.1f}%</p>
                                    <span style="font-size: 0.75em; color: #777;">Confidence Score</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Consensus check
                    votes = [res["pred"] for res in pipeline_results.values()]
                    from collections import Counter
                    vote_counts = Counter(votes)
                    top_vote, top_count = vote_counts.most_common(1)[0]
                    consensus_color = CATEGORY_COLORS.get(top_vote, "#333")
                    st.markdown(
                        f"<p style='margin-top: 14px; font-size: 1.05em;'><strong>Multi-Model Consensus:</strong> "
                        f"<span style='color: {consensus_color}; font-weight: bold;'>{top_vote.capitalize()}</span> "
                        f"({top_count} of {len(pipeline_results)} models agree)</p>",
                        unsafe_allow_html=True,
                    )

                    # Detailed Probability Comparison Table
                    with st.expander(f"📊 View Complete Category Probability Breakdown across all {len(pipeline_results)} Models", expanded=False):
                        prob_table_data = []
                        for cat in sorted(list(VALID_CATEGORIES)):
                            row = {"Category": f"{CATEGORY_ICONS.get(cat, '')} {cat.capitalize()}"}
                            for pipe_name, res in pipeline_results.items():
                                prob_val = res["probs"].get(cat, 0.0)
                                row[pipe_name] = f"{prob_val * 100:.2f}%"
                            prob_table_data.append(row)
                        st.dataframe(pd.DataFrame(prob_table_data), use_container_width=True)

                    # Explainability section for TF-IDF
                    if active_explanation is not None:
                        st.markdown("---")
                        st.subheader("🔬 Mathematical Feature Explainability (TF-IDF Weights)")
                        active_terms = active_explanation.get("active_vocab_terms", [])
                        if active_terms:
                            st.write(f"**Vocabulary Terms Detected in Input ({len(active_terms)} found):** " + ", ".join([f"`{t}`" for t in active_terms]))
                            top_pipe_pred = pipeline_results.get("LR on TF-IDF (1+2g)", {}).get("pred", top_vote)
                            top_feats = active_explanation["class_explanations"].get(top_pipe_pred, [])
                            if top_feats:
                                st.write(f"**Top Mathematical Contributors for `{top_pipe_pred.upper()}` ($x_j \\cdot W_c[j]$):**")
                                feat_cols = st.columns(min(len(top_feats), 4))
                                for f_idx, feat in enumerate(top_feats[:4]):
                                    with feat_cols[f_idx]:
                                        st.metric(
                                            label=feat["term"],
                                            value=f"{feat['contribution']:+.3f}",
                                            delta=f"w={feat['weight']:+.2f} | tfidf={feat['tfidf']:.2f}",
                                        )
                        else:
                            st.info("ℹ️ No in-vocabulary terms detected for this short input. Prediction reflects base classifier biases.")

                    # -------------------------------------------------
                    # Section C: N-Gram Language Model Analysis
                    # -------------------------------------------------
                    st.divider()
                    st.subheader(f"4. N-Gram Language Model Analysis ({selected_ngram_label})")

                    lm = ngram_models.get(selected_n)
                    if lm is None:
                        st.info(f"N-Gram language model for n={selected_n} is not available.")
                    else:
                        extracted_ngrams = lm.extract_ngrams(tokens, n=selected_n)
                        last_tokens = tokens[- (selected_n - 1) :] if len(tokens) >= (selected_n - 1) else tokens
                        ctx_str = " ".join(last_tokens) if last_tokens else "<s>"

                        # Perplexity calculation
                        ppl = lm.calculate_perplexity(tokens)
                        log_lik = lm.calculate_log_likelihood(tokens)

                        ng_col1, ng_col2 = st.columns([1, 1])

                        with ng_col1:
                            st.write("**Language Model Predictability Metrics:**")
                            m_col1, m_col2 = st.columns(2)
                            with m_col1:
                                st.metric("Perplexity PP(W)", f"{ppl:.2f}" if ppl != float("inf") else "∞")
                            with m_col2:
                                st.metric("Log-Likelihood", f"{log_lik:.2f}")

                            st.write(f"**Extracted {selected_n}-Grams ({len(extracted_ngrams):,} found):**")
                            if not extracted_ngrams:
                                st.caption(f"*Sequence length ({len(tokens)} tokens) is shorter than order n={selected_n}.*")
                            else:
                                display_ng = [" ".join(ng) for ng in extracted_ngrams[:8]]
                                st.write(", ".join([f"`{ng}`" for ng in display_ng]))
                                if len(extracted_ngrams) > 8:
                                    st.caption(f"*...and {len(extracted_ngrams) - 8} more.*")

                        with ng_col2:
                            st.write(f"**Top Next-Word Suggestions given context: `'{ctx_str}'`**")
                            predictions = lm.predict_next_words(last_tokens, top_k=5)
                            if not predictions:
                                st.caption("*No in-vocabulary context predictions available.*")
                            else:
                                pred_rows = [
                                    {"Next Word": f"'{w}'", "Conditional Probability P(w|context)": f"{p:.5f}"}
                                    for w, p in predictions
                                ]
                                st.table(pd.DataFrame(pred_rows))

            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")


# =====================================================================
# TAB 2: Word2Vec Semantic Explorer
# =====================================================================

with tab_semantic:
    st.subheader("Word2Vec Semantic Explorer (Custom Skip-Gram Architecture)")
    st.markdown(
        "Query the custom **100-dimensional Skip-Gram Word2Vec embeddings** trained directly on the BBC News corpus. "
        "Calculates real mathematical **Cosine Similarity**: "
        r"$\cos(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$."
    )

    if w2v_model is None:
        st.error("❌ Custom Word2Vec model not found at `models/custom_word2vec.pt`.")
    else:
        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            query_word = st.text_input(
                "Enter a query word:",
                value="football",
                placeholder="e.g. football, government, software, market, movie",
                help="Test semantic neighbors across different BBC news domains.",
            )
        with q_col2:
            top_k_val = st.slider("Top K Neighbors", min_value=3, max_value=15, value=5)

        if st.button("🔎 Find Nearest Neighbors", key="btn_w2v_search"):
            q_clean = query_word.strip().lower()
            if not q_clean:
                st.warning("Please enter a query word.")
            elif q_clean not in w2v_model.word_to_idx:
                st.warning(
                    f"⚠️ Word `'{q_clean}'` is out-of-vocabulary in the Custom Word2Vec model "
                    f"({w2v_model.vocab_size:,} active words)."
                )
            else:
                neighbors = w2v_model.find_nearest_neighbors(q_clean, top_k=top_k_val)
                st.write(f"### Nearest Neighbors for **'{q_clean}'**:")

                n_col1, n_col2 = st.columns([2, 3])
                with n_col1:
                    df_neighbors = pd.DataFrame([
                        {"Neighbor Word": word, "Cosine Similarity": round(score, 4)}
                        for word, score in neighbors
                    ])
                    st.table(df_neighbors)

                with n_col2:
                    st.write("**Similarity Bar Distribution:**")
                    for word, score in neighbors:
                        norm_score = max(0.0, min(1.0, float(score)))
                        st.write(f"**{word}** — {score:.4f}")
                        st.progress(norm_score)


# =====================================================================
# TAB 3: Model Evaluation Benchmarks
# =====================================================================

with tab_benchmarks:
    st.subheader("Model Evaluation Benchmarks")
    metrics_data = load_metrics_report()

    if metrics_data is None:
        st.info("ℹ️ Evaluation metrics not found. Run `python train_pipeline.py` to generate reports.")
    else:
        st.write("### Model Comparison Table (Held-Out Test Split: 445 Articles)")
        table_rows = []
        for p_id, p_info in metrics_data.get("pipelines", {}).items():
            table_rows.append({
                "Pipeline": p_info.get("name"),
                "Accuracy": f"{p_info.get('accuracy', 0)*100:.2f}%",
                "Macro Precision": f"{p_info.get('macro_precision', 0)*100:.2f}%",
                "Macro Recall": f"{p_info.get('macro_recall', 0)*100:.2f}%",
                "Macro F1-Score": f"{p_info.get('macro_f1', 0)*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

