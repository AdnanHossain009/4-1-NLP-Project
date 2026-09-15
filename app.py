"""Interactive Streamlit Web Application for News Classification and Semantic Analysis.

Foundational NLP Demonstration:
- Generalized N-Gram Language Modeling
- Custom Skip-Gram Word2Vec (PyTorch from scratch)
- Multinomial Naive Bayes (Pure NumPy from scratch)
- One-vs-Rest Logistic Regression (Pure NumPy from scratch)
- Offline Pretrained GloVe Document Aggregations (Mean & TF-IDF)
- Unsupervised K-Means Clustering Analysis & Evaluation Benchmarks
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    compute_training_idf,
    aggregate_document_mean,
    aggregate_document_tfidf,
    generate_offline_fallback_glove_cache,
)
from src.logistic_regression import (
    OneVsRestLogisticRegression,
    BinaryLogisticRegression,
)

# Page configuration
st.set_page_config(
    page_title="News Classification & Semantic Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Universal category styling
CATEGORY_COLORS = {
    "business": "#2b5c8f",
    "entertainment": "#8e44ad",
    "politics": "#c0392b",
    "sport": "#27ae60",
    "tech": "#d35400",
}

PRESET_EXAMPLES = {
    "Sport": "The football club secured a dramatic victory in the championship final after the striker scored two decisive goals in extra time.",
    "Tech": "Software engineers developed an advanced neural algorithm for mobile microprocessors to accelerate machine learning inference.",
    "Business": "Corporate profits climbed across international equity markets as central banks signaled prospective interest rate cuts to stimulate economic growth.",
    "Politics": "The prime minister defended the annual budget in parliament, promising fiscal reforms and taxation incentives for healthcare and transport infrastructure.",
    "Entertainment": "The independent film festival awarded top honors to acclaimed international directors and veteran actors during the gala awards ceremony.",
}


# =====================================================================
# Cached Model & Resource Loaders (Zero Retraining on Page Load)
# =====================================================================

@st.cache_resource(show_spinner="Initializing NLP Preprocessor & Lexicons...")
def get_preprocessor():
    ensure_nltk_resources()
    return TextPreprocessor()


@st.cache_resource(show_spinner="Loading BBC Training Split & IDF Weights...")
def get_training_context():
    csv_path = os.path.join(project_root, "data", "bbc_news.csv")
    if not os.path.exists(csv_path):
        return None, {}, 1.0, []

    df = load_dataset(csv_path)
    train_df, _ = stratified_train_test_split(df, test_size=0.2, random_state=42)
    prep = get_preprocessor()
    train_tokens = [prep.preprocess(t, return_tokens=True) for t in train_df["text"]]
    idf_weights, default_idf = compute_training_idf(train_tokens, smooth=True)
    return train_df, idf_weights, default_idf, train_tokens


@st.cache_resource(show_spinner="Loading Multinomial Naive Bayes Model...")
def get_naive_bayes_model():
    path = os.path.join(project_root, "models", "naive_bayes_weights.pkl")
    if not os.path.exists(path):
        return None
    return MultinomialNaiveBayes.load(path)


@st.cache_resource(show_spinner="Loading Custom Word2Vec Model...")
def get_word2vec_model():
    m_path = os.path.join(project_root, "models", "custom_word2vec.pt")
    v_path = os.path.join(project_root, "models", "vocab.json")
    if not os.path.exists(m_path) or not os.path.exists(v_path):
        return None
    w2v = CustomWord2Vec(embedding_dim=100)
    w2v.load(m_path, v_path)
    return w2v


@st.cache_resource(show_spinner="Loading Offline GloVe Embedding Cache...")
def get_glove_model():
    npy_path = os.path.join(project_root, "models", "pretrained_embeddings.npy")
    vocab_path = os.path.join(project_root, "models", "pretrained_embeddings_vocab.json")
    if not os.path.exists(npy_path) or not os.path.exists(vocab_path):
        return generate_offline_fallback_glove_cache(npy_path, vocab_path)
    return GloVeEmbeddings.load(npy_path, vocab_path)


@st.cache_resource(show_spinner="Loading N-Gram Language Models...")
def get_ngram_models():
    _, _, _, train_tokens = get_training_context()
    if not train_tokens:
        return {}
    models = {}
    for n in [2, 3, 4, 5]:
        lm = NGramLanguageModel(n=n, laplace_smoothing=True)
        # Train on representative subset of 400 docs for interactive responsiveness
        lm.train(train_tokens[:400])
        models[n] = lm
    return models


@st.cache_resource(show_spinner="Loading Logistic Regression Weights Bundle...")
def get_logistic_regression_bundle():
    path = os.path.join(project_root, "models", "logistic_regression_weights.pkl")
    if not os.path.exists(path):
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def get_lr_model_for_pipeline(pipe_key: str) -> Optional[OneVsRestLogisticRegression]:
    bundle = get_logistic_regression_bundle()
    if bundle is None:
        return None
    if "all_pipelines" in bundle and pipe_key in bundle["all_pipelines"]:
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
    return OneVsRestLogisticRegression.load(os.path.join(project_root, "models", "logistic_regression_weights.pkl"))


@st.cache_data
def load_metrics_report():
    path = os.path.join(project_root, "reports", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# Main Header & System State
# =====================================================================

st.title("News Classification & Semantic Analysis")
st.markdown("##### *Foundational NLP — N-Gram, Word2Vec, Naive Bayes and Logistic Regression*")
st.caption(
    "100% Formula-First Mathematical NLP | Strict Offline Execution | Zero LLMs, Zero RAG, Zero Black-Box Wrappers"
)
st.divider()

# Sidebar: System Status & Presets
with st.sidebar:
    st.header("⚙️ System Status")
    nb = get_naive_bayes_model()
    w2v = get_word2vec_model()
    glove = get_glove_model()
    lr_bundle = get_logistic_regression_bundle()

    st.success("✅ BBC News Corpus: Loaded (2,225 records)")
    st.write(f"• **Naive Bayes**: {'Ready' if nb else 'Missing'}")
    st.write(f"• **Custom Word2Vec**: {'Ready (100d)' if w2v else 'Missing'}")
    st.write(f"• **Pretrained GloVe**: {'Ready (100d)' if glove else 'Missing'}")
    st.write(f"• **Logistic Regression**: {'Ready (OvR)' if lr_bundle else 'Missing'}")

    st.divider()
    st.header("📋 Example Articles")
    st.write("Click a preset below to populate the analysis input:")
    for label, text in PRESET_EXAMPLES.items():
        if st.button(f"📰 {label}", use_container_width=True):
            st.session_state["article_input"] = text

    st.divider()
    st.info(
        "**Academic Rules Enforced:**\n"
        "- No RAG, No Transformers\n"
        "- Pure NumPy classification\n"
        "- Real model inference only\n"
        "- Zero data leakage"
    )

# Tabs
tab_classify, tab_semantic, tab_benchmarks = st.tabs([
    "🔍 Article Classification & N-Gram Analysis",
    "🌐 Word2Vec Semantic Explorer",
    "📊 Benchmarks & Clustering Visualizations",
])


# =====================================================================
# TAB 1: Classification & N-Gram Analysis
# =====================================================================
with tab_classify:
    st.subheader("1. Text Input & Configuration")

    col_model, col_rep, col_agg, col_order = st.columns(4)

    with col_model:
        selected_model = st.selectbox(
            "Model Architecture",
            options=["Naive Bayes", "Logistic Regression"],
            help="Select the supervised classification engine.",
        )

    with col_rep:
        if selected_model == "Naive Bayes":
            representation_options = ["N-Gram / Bag-of-Words"]
            selected_rep = st.selectbox(
                "Feature Representation",
                options=representation_options,
                disabled=True,
                help="Naive Bayes uses exact token frequency / Bag-of-Words features.",
            )
        else:
            representation_options = ["Custom Word2Vec", "Pretrained GloVe"]
            selected_rep = st.selectbox(
                "Feature Representation",
                options=representation_options,
                help="Select the vector embedding space for Logistic Regression.",
            )

    with col_agg:
        if selected_model == "Logistic Regression":
            selected_agg = st.selectbox(
                "Document Aggregation",
                options=["TF-IDF Weighted", "Mean Aggregation"],
                help="Select how word vectors are pooled into a document vector.",
            )
        else:
            selected_agg = st.selectbox(
                "Document Aggregation",
                options=["N/A (Count BoW)"],
                disabled=True,
            )

    with col_order:
        ngram_order_labels = {
            "Bigram (n=2)": 2,
            "Trigram (n=3)": 3,
            "4-gram (n=4)": 4,
            "5-gram (n=5)": 5,
        }
        selected_ngram_label = st.selectbox(
            "N-Gram Analysis Order",
            options=list(ngram_order_labels.keys()),
            help="Select the order for the parallel N-Gram language model inspection.",
        )
        selected_n = ngram_order_labels[selected_ngram_label]

    # Large text area
    default_text = st.session_state.get("article_input", PRESET_EXAMPLES["Sport"])
    user_text = st.text_area(
        "News Article / Text",
        value=default_text,
        height=140,
        placeholder="Paste a news article lead or complete story here...",
    )

    analyze_btn = st.button("🚀 Analyze Article", type="primary", use_container_width=True)

    if analyze_btn:
        # Robust validation
        cleaned_input = user_text.strip()
        if not cleaned_input:
            st.warning("⚠️ Input is empty. Please enter or select a news article text to analyze.")
        else:
            preprocessor = get_preprocessor()
            tokens = preprocessor.preprocess(cleaned_input, return_tokens=True)

            if not tokens:
                st.warning("⚠️ The article contains no valid alphanumeric words after text cleaning.")
            else:
                st.divider()
                st.subheader("2. Live Classification Result")

                predicted_category = None
                confidence = 0.0
                probabilities = {}

                # -----------------------------------------------------
                # Branch 1: Naive Bayes Inference
                # -----------------------------------------------------
                if selected_model == "Naive Bayes":
                    if nb is None:
                        st.error("❌ Naive Bayes model artifact not found. Please run `python train_pipeline.py` first.")
                    else:
                        probs_arr = nb.predict_proba([tokens])[0]
                        pred_cat = nb.predict([tokens])[0]
                        predicted_category = pred_cat
                        probabilities = {cls_name: float(p) for cls_name, p in zip(nb.classes_, probs_arr)}
                        confidence = probabilities[predicted_category]

                # -----------------------------------------------------
                # Branch 2: Logistic Regression Inference
                # -----------------------------------------------------
                else:
                    emb_model = w2v if selected_rep == "Custom Word2Vec" else glove
                    if emb_model is None:
                        st.error(f"❌ {selected_rep} model not found. Please run `python train_pipeline.py` first.")
                    else:
                        is_tfidf = (selected_agg == "TF-IDF Weighted")
                        pipe_suffix = "tfidf" if is_tfidf else "mean"
                        prefix = "w2v" if selected_rep == "Custom Word2Vec" else "glove"
                        pipeline_key = f"{prefix}_{pipe_suffix}"

                        clf = get_lr_model_for_pipeline(pipeline_key)
                        if clf is None:
                            st.error("❌ Logistic Regression weights not found. Run `python train_pipeline.py`.")
                        else:
                            _, idf_weights, default_idf, _ = get_training_context()
                            if is_tfidf:
                                doc_vec = aggregate_document_tfidf(tokens, emb_model, idf_weights, default_idf=default_idf)
                            else:
                                doc_vec = aggregate_document_mean(tokens, emb_model)

                            if np.all(doc_vec == 0.0):
                                st.info("ℹ️ All words in this document are out-of-vocabulary in the selected embedding space. Using baseline class distribution.")

                            probs_arr = clf.predict_proba(doc_vec)[0]
                            pred_cat = clf.predict(doc_vec)[0]
                            predicted_category = pred_cat
                            probabilities = {cls_name: float(p) for cls_name, p in zip(clf.classes, probs_arr)}
                            confidence = probabilities[predicted_category]

                # Display Results
                if predicted_category is not None:
                    cat_color = CATEGORY_COLORS.get(predicted_category, "#333333")
                    res_col1, res_col2 = st.columns([1, 2])

                    with res_col1:
                        st.markdown(
                            f"""
                            <div style="background-color: {cat_color}18; border-left: 6px solid {cat_color}; padding: 16px 20px; border-radius: 6px;">
                                <span style="font-size: 0.9em; text-transform: uppercase; color: {cat_color}; font-weight: bold; letter-spacing: 1px;">Predicted Category</span>
                                <h2 style="margin: 4px 0; color: {cat_color};">{predicted_category.capitalize()}</h2>
                                <p style="margin: 0; font-size: 1.1em; color: #444;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
                                <span style="font-size: 0.85em; color: #666;">Pipeline: {selected_model} ({selected_rep} - {selected_agg})</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with res_col2:
                        st.write("**Category Probability Distribution:**")
                        df_probs = pd.DataFrame({
                            "Category": [c.capitalize() for c in probabilities.keys()],
                            "Probability": list(probabilities.values()),
                            "Percentage": [f"{p*100:.2f}%" for p in probabilities.values()],
                        }).sort_values("Probability", ascending=False)

                        for _, row in df_probs.iterrows():
                            c_name = row["Category"].lower()
                            bar_col = CATEGORY_COLORS.get(c_name, "#555")
                            st.write(f"**{row['Category']}** — {row['Percentage']}")
                            st.progress(min(1.0, max(0.0, float(row["Probability"]))))

                # -----------------------------------------------------
                # Section 3: N-Gram Language Model Analysis
                # -----------------------------------------------------
                st.divider()
                st.subheader(f"3. N-Gram Language Model Analysis ({selected_ngram_label})")

                ngram_models = get_ngram_models()
                lm = ngram_models.get(selected_n)

                if lm is None:
                    st.info("N-Gram language model is initializing...")
                else:
                    extracted_ngrams = lm.extract_ngrams(tokens, n=selected_n)
                    col_ng1, col_ng2 = st.columns(2)

                    with col_ng1:
                        st.write(f"**Extracted {selected_n}-Grams ({len(extracted_ngrams):,} found):**")
                        if not extracted_ngrams:
                            st.write(f"*Sentence length ({len(tokens)} tokens) is shorter than n={selected_n}.*")
                        else:
                            display_ngrams = [" ".join(ng) for ng in extracted_ngrams[:12]]
                            st.write(", ".join([f"`{ng}`" for ng in display_ngrams]))
                            if len(extracted_ngrams) > 12:
                                st.caption(f"*...and {len(extracted_ngrams) - 12} more.*")

                    with col_ng2:
                        last_tokens = tokens[- (selected_n - 1) :] if len(tokens) >= (selected_n - 1) else tokens
                        ctx_str = " ".join(last_tokens) if last_tokens else "<s>"
                        st.write(f"**Top-5 Next-Word Predictions given context: `'{ctx_str}'`**")

                        preds = lm.predict_next_words(last_tokens, top_k=5)
                        if not preds:
                            st.write("*No in-vocabulary context predictions available.*")
                        else:
                            pred_data = []
                            for word, prob in preds:
                                pred_data.append({
                                    "Next Word": f"'{word}'",
                                    "Conditional Probability P(w|context)": f"{prob:.5f}",
                                })
                            st.table(pd.DataFrame(pred_data))


# =====================================================================
# TAB 2: Word2Vec Semantic Explorer
# =====================================================================
with tab_semantic:
    st.subheader("Word2Vec Semantic Explorer (Custom Skip-Gram Architecture)")
    st.markdown(
        "Query the custom **100-dimensional Skip-Gram Word2Vec embeddings** trained directly on the BBC News corpus. "
        "Calculates real mathematical **Cosine Similarity**: $\\cos(a, b) = \\frac{a \\cdot b}{\\|a\\| \\|b\\|}$."
    )

    if w2v is None:
        st.error("❌ Custom Word2Vec model not found at `models/custom_word2vec.pt`. Run `python train_pipeline.py`.")
    else:
        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            query_word = st.text_input("Enter a query word:", value="football", placeholder="e.g. government, software, market, film")
        with q_col2:
            top_k_val = st.slider("Top K Neighbors", min_value=3, max_value=15, value=5)

        if st.button("🔎 Find Nearest Neighbors", key="btn_w2v_search"):
            q_clean = query_word.strip().lower()
            if not q_clean:
                st.warning("Please enter a query word.")
            elif q_clean not in w2v.word_to_idx:
                st.warning(f"⚠️ Word `'{q_clean}'` is out-of-vocabulary in the Custom Word2Vec model ({w2v.vocab_size:,} active words).")
            else:
                neighbors = w2v.find_nearest_neighbors(q_clean, top_k=top_k_val)
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
# TAB 3: Benchmarks & Clustering Visualizations
# =====================================================================
with tab_benchmarks:
    st.subheader("Evaluation Benchmarks & Unsupervised Clustering")
    metrics_data = load_metrics_report()

    if metrics_data is None:
        st.info("ℹ️ Evaluation metrics not found. Run `python src/clustering_eval.py` or `python train_pipeline.py`.")
    else:
        st.write("### 1. Model Comparison Table (Held-Out Test Split: 445 Articles)")
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

        st.divider()
        st.write("### 2. Multi-Pipeline Confusion Matrices")
        cm_image_path = os.path.join(project_root, "reports", "confusion_matrix.png")
        if os.path.exists(cm_image_path):
            st.image(cm_image_path, caption="Confusion Matrices across all 5 pipelines (Held-out 445-sample test split)", use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found at `reports/confusion_matrix.png`.")

        st.divider()
        st.write("### 3. Unsupervised K-Means Clustering Analysis (k=5)")
        clustering_info = metrics_data.get("clustering", {})
        s_score = clustering_info.get("silhouette_score", 0.0)

        st.info(
            f"**Clustering Metric — Silhouette Score:** `{s_score:.4f}`\n\n"
            f"**Academic Constraint:** {clustering_info.get('note', 'Unsupervised clustering analysis; not classification accuracy')}"
        )

        cluster_image_path = os.path.join(project_root, "reports", "clusters.png")
        if os.path.exists(cluster_image_path):
            st.image(cluster_image_path, caption="2D PCA Projection: Unsupervised K-Means Clusters vs Ground-Truth References", use_container_width=True)
        else:
            st.warning("Cluster visualization plot not found at `reports/clusters.png`.")

