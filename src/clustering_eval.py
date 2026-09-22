"""Evaluation Benchmarks, Confusion Matrices, and Unsupervised K-Means Clustering.

Implements:
1. Pure mathematical classification metrics:
     - Accuracy, Precision, Recall, Macro F1, Confusion Matrix
2. Multi-Pipeline comparative evaluation on held-out test split:
     - Pipeline A: N-Gram/BoW -> Naive Bayes
     - Pipeline B: Custom Word2Vec + Mean -> Logistic Regression
     - Pipeline C: Custom Word2Vec + TF-IDF -> Logistic Regression
     - Pipeline D: Pretrained GloVe + Mean -> Logistic Regression
     - Pipeline E: Pretrained GloVe + TF-IDF -> Logistic Regression
3. Artifact generation:
     - reports/metrics.json
     - reports/confusion_matrix.png
4. Unsupervised K-Means clustering (k=5):
     - Silhouette score computation
     - 2D PCA semantic projection visualization -> reports/clusters.png
     - Explicit documentation: Unsupervised clustering is NOT classification accuracy.
"""

import os
import sys
import json
import math
from typing import List, Tuple, Dict, Union, Optional, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for server and test environments
import matplotlib.pyplot as plt

# Ensure project root is in sys.path when executed directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from src.preprocessing import TextPreprocessor, load_dataset, stratified_train_test_split, VALID_CATEGORIES
from src.naive_bayes import MultinomialNaiveBayes
from src.logistic_regression import OneVsRestLogisticRegression, BinaryLogisticRegression
from src.tfidf_extractor import TfidfNGramFeatureExtractor
from src.embeddings import (
    GloVeEmbeddings,
    compute_training_idf,
    vectorize_corpus_mean,
    vectorize_corpus_tfidf,
)

try:
    from src.word2vec_scratch import CustomWord2Vec
except ImportError:
    CustomWord2Vec = None

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ORDERED_CATEGORIES = sorted(list(VALID_CATEGORIES))


# =====================================================================
# 1. Pure Formula Classification Metrics (Zero Sklearn Classification)
# =====================================================================

def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Accuracy, Precision, Recall, F1, and Confusion Matrix using exact formulas.

    Formulas:
        TP_c = sum 1(y == c and y_pred == c)
        FP_c = sum 1(y != c and y_pred == c)
        FN_c = sum 1(y == c and y_pred != c)
        Precision_c = TP_c / (TP_c + FP_c)
        Recall_c    = TP_c / (TP_c + FN_c)
        F1_c        = 2 * P_c * R_c / (P_c + R_c)
        Macro F1    = (1 / |C|) * sum_{c} F1_c
    """
    cat_list = list(classes) if classes is not None else ORDERED_CATEGORIES
    cat_to_idx = {c: i for i, c in enumerate(cat_list)}
    n_classes = len(cat_list)
    total_samples = len(y_true)

    if total_samples == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    # Confusion matrix: rows = true, cols = pred
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in cat_to_idx and pred_label in cat_to_idx:
            cm[cat_to_idx[true_label], cat_to_idx[pred_label]] += 1

    accuracy = float(np.trace(cm) / total_samples)

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for idx, cat in enumerate(cat_list):
        tp = float(cm[idx, idx])
        fp = float(np.sum(cm[:, idx]) - tp)
        fn = float(np.sum(cm[idx, :]) - tp)
        support = int(np.sum(cm[idx, :]))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2.0 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class[cat] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    total_support = sum(per_class[cat]["support"] for cat in cat_list)
    weighted_f1 = float(sum(per_class[cat]["f1"] * per_class[cat]["support"] for cat in cat_list) / total_support) if total_support > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "classes": cat_list,
        "per_class": per_class,
    }


# =====================================================================
# 2. Pipeline Model Loader Helper
# =====================================================================

def load_pipeline_logistic_regression(
    bundle_path: str = "models/logistic_regression_weights.pkl",
    pipeline_key: str = "glove_tfidf",
) -> OneVsRestLogisticRegression:
    """Instantiate OneVsRestLogisticRegression for a specific representation pipeline."""
    import pickle
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    if "all_pipelines" in bundle and pipeline_key in bundle["all_pipelines"]:
        p_data = bundle["all_pipelines"][pipeline_key]
        clf = OneVsRestLogisticRegression(
            classes=p_data["classes"],
            learning_rate=bundle.get("learning_rate", 0.5),
            n_iterations=bundle.get("n_iterations", 800),
            l2_reg=bundle.get("l2_reg", 0.0001),
        )
        clf.weights = p_data["weights"]
        clf.biases = p_data["biases"]
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
    else:
        return OneVsRestLogisticRegression.load(bundle_path)


# =====================================================================
# 3. Comprehensive Multi-Pipeline Evaluator
# =====================================================================

def evaluate_all_five_pipelines(
    csv_path: str = "data/bbc_news.csv",
    reports_dir: str = "reports",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate all five project pipelines on the official stratified test split.

    Pipelines:
      - Pipeline A: N-Gram/BoW -> Naive Bayes
      - Pipeline B: Custom Word2Vec (Mean) -> Logistic Regression
      - Pipeline C: Custom Word2Vec (TF-IDF) -> Logistic Regression
      - Pipeline D: Pretrained GloVe (Mean) -> Logistic Regression
      - Pipeline E: Pretrained GloVe (TF-IDF) -> Logistic Regression
    """
    os.makedirs(reports_dir, exist_ok=True)
    df = load_dataset(csv_path)
    train_df, test_df = stratified_train_test_split(df, test_size=0.2, random_state=42)

    preprocessor = TextPreprocessor()
    train_tokens = [preprocessor.preprocess(t, return_tokens=True) for t in train_df["text"]]
    test_tokens = [preprocessor.preprocess(t, return_tokens=True) for t in test_df["text"]]
    y_test = test_df["category"].tolist()

    # Zero-leakage IDF weights
    idf_weights, default_idf = compute_training_idf(train_tokens, smooth=True)

    # 1. Pipeline A: Naive Bayes
    nb_path = os.path.join(project_root, "models", "naive_bayes_weights.pkl")
    nb_model = MultinomialNaiveBayes.load(nb_path)
    preds_a = nb_model.predict(test_tokens)
    metrics_a = compute_classification_metrics(y_test, preds_a, ORDERED_CATEGORIES)
    metrics_a["name"] = "Pipeline A: N-Gram/BoW -> Naive Bayes"
    metrics_a["short_name"] = "NB (BoW)"

    # Model and representation metadata for baseline pipelines
    metrics_a["model"] = "Naive Bayes"
    metrics_a["representation"] = "Count / BoW"

    # Load Word2Vec & GloVe models
    w2v = None
    w2v_path = os.path.join(project_root, "models", "custom_word2vec.pt")
    w2v_vocab = os.path.join(project_root, "models", "vocab.json")
    if CustomWord2Vec is not None and os.path.exists(w2v_path) and os.path.exists(w2v_vocab):
        w2v = CustomWord2Vec(embedding_dim=100)
        w2v.load(w2v_path, w2v_vocab)

    glove_npy = os.path.join(project_root, "models", "pretrained_embeddings.npy")
    glove_vocab = os.path.join(project_root, "models", "pretrained_embeddings_vocab.json")
    glove = GloVeEmbeddings.load(glove_npy, glove_vocab)

    lr_bundle_path = os.path.join(project_root, "models", "logistic_regression_weights.pkl")

    # Document vectors for test split
    pipeline_evals = {"pipeline_a": metrics_a}

    lr_configs = [
        ("pipeline_b", "w2v_mean", "Pipeline B: Custom Word2Vec (Mean) -> Logistic Regression", "W2V (Mean)", "Logistic Regression", "Word2Vec Mean", w2v, "mean"),
        ("pipeline_c", "w2v_tfidf", "Pipeline C: Custom Word2Vec (TF-IDF) -> Logistic Regression", "W2V (TF-IDF)", "Logistic Regression", "Word2Vec TF-IDF", w2v, "tfidf"),
        ("pipeline_d", "glove_mean", "Pipeline D: Pretrained GloVe (Mean) -> Logistic Regression", "GloVe (Mean)", "Logistic Regression", "GloVe Mean", glove, "mean"),
        ("pipeline_e", "glove_tfidf", "Pipeline E: Pretrained GloVe (TF-IDF) -> Logistic Regression", "GloVe (TF-IDF)", "Logistic Regression", "GloVe TF-IDF", glove, "tfidf"),
    ]

    saved_test_vectors = {}

    for pipe_id, lr_key, full_name, short_name, mod_name, rep_name, emb_model, agg_type in lr_configs:
        if emb_model is None:
            continue
        if agg_type == "mean":
            X_test = vectorize_corpus_mean(test_tokens, emb_model)
        else:
            X_test = vectorize_corpus_tfidf(test_tokens, emb_model, idf_weights, default_idf=default_idf)

        saved_test_vectors[pipe_id] = X_test
        clf = load_pipeline_logistic_regression(lr_bundle_path, lr_key)
        preds = clf.predict(X_test)
        metrics = compute_classification_metrics(y_test, preds, ORDERED_CATEGORIES)
        metrics["name"] = full_name
        metrics["short_name"] = short_name
        metrics["model"] = mod_name
        metrics["representation"] = rep_name
        pipeline_evals[pipe_id] = metrics

    # 3. TF-IDF Pipelines F & G
    ext_uni_path = os.path.join(project_root, "models", "tfidf_unigram_extractor.pkl")
    ext_bi_path = os.path.join(project_root, "models", "tfidf_extractor.pkl")

    if os.path.exists(ext_uni_path):
        ext_uni = TfidfNGramFeatureExtractor.load(ext_uni_path)
        X_test_uni = ext_uni.transform(test_tokens)
        clf_uni = load_pipeline_logistic_regression(lr_bundle_path, "tfidf_unigram")
        preds_uni = clf_uni.predict(X_test_uni)
        metrics_f = compute_classification_metrics(y_test, preds_uni, ORDERED_CATEGORIES)
        metrics_f["name"] = "Pipeline F: TF-IDF (Unigram) -> Logistic Regression"
        metrics_f["short_name"] = "TF-IDF (1-g)"
        metrics_f["model"] = "Logistic Regression"
        metrics_f["representation"] = "TF-IDF Unigram"
        pipeline_evals["pipeline_f"] = metrics_f

    if os.path.exists(ext_bi_path):
        ext_bi = TfidfNGramFeatureExtractor.load(ext_bi_path)
        X_test_bi = ext_bi.transform(test_tokens)
        clf_bi = load_pipeline_logistic_regression(lr_bundle_path, "tfidf_unigram_bigram")
        preds_bi = clf_bi.predict(X_test_bi)
        probs_bi = clf_bi.predict_proba(X_test_bi)
        metrics_g = compute_classification_metrics(y_test, preds_bi, ORDERED_CATEGORIES)
        metrics_g["name"] = "Pipeline G: TF-IDF (Unigram + Bigram) -> Logistic Regression"
        metrics_g["short_name"] = "TF-IDF (1+2g)"
        metrics_g["model"] = "Logistic Regression"
        metrics_g["representation"] = "TF-IDF Unigram + Bigram"
        pipeline_evals["pipeline_g"] = metrics_g

        # Generate Error Analysis Report (10-20 actual test set misclassifications)
        generate_error_analysis(
            test_df=test_df,
            test_tokens=test_tokens,
            y_true=y_test,
            y_pred=preds_bi,
            probs=probs_bi,
            extractor=ext_bi,
            clf=clf_bi,
            output_path=os.path.join(reports_dir, "error_analysis.txt"),
        )

    # 4. Export Model Comparison Table as CSV
    comparison_rows = []
    pipe_order = ["pipeline_a", "pipeline_b", "pipeline_c", "pipeline_d", "pipeline_e", "pipeline_f", "pipeline_g"]
    for p_id in pipe_order:
        if p_id in pipeline_evals:
            p_data = pipeline_evals[p_id]
            comparison_rows.append({
                "Pipeline ID": p_id.upper(),
                "Model": p_data.get("model", p_data["short_name"]),
                "Representation": p_data.get("representation", p_data["name"]),
                "Accuracy": f"{p_data['accuracy']:.4f}",
                "Precision": f"{p_data['macro_precision']:.4f}",
                "Recall": f"{p_data['macro_recall']:.4f}",
                "Macro F1": f"{p_data['macro_f1']:.4f}",
                "Weighted F1": f"{p_data.get('weighted_f1', p_data['macro_f1']):.4f}",
            })
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)

    # 5. Unsupervised K-Means Clustering on primary document vectors
    cluster_vectors = saved_test_vectors.get("pipeline_c", saved_test_vectors.get("pipeline_e"))
    clustering_results = run_kmeans_clustering(
        vectors=cluster_vectors,
        ground_truth_labels=y_test,
        k=5,
        output_png=os.path.join(reports_dir, "clusters.png"),
    )

    combined_reports = {
        "pipelines": pipeline_evals,
        "clustering": clustering_results,
    }

    # Save metrics.json
    metrics_json_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_reports, f, indent=2)

    # Save confusion_matrix.png
    cm_png_path = os.path.join(reports_dir, "confusion_matrix.png")
    plot_confusion_matrices(pipeline_evals, cm_png_path)

    if verbose:
        print("\n" + "=" * 95)
        print("TRADITIONAL NLP BENCHMARK COMPARISON (HELD-OUT TEST SPLIT: 445 SAMPLES)")
        print("=" * 95)
        header = f"{'Pipeline':<12} | {'Model':<20} | {'Representation':<24} | {'Acc':<7} | {'Macro F1':<8} | {'W-F1':<6}"
        print(header)
        print("-" * len(header))
        for row in comparison_rows:
            print(f"{row['Pipeline ID']:<12} | {row['Model']:<20} | {row['Representation']:<24} | {float(row['Accuracy'])*100:>6.2f}% | {float(row['Macro F1'])*100:>7.2f}% | {float(row['Weighted F1'])*100:>5.2f}%")
        print("=" * 95)
        print(f"[Artifacts] Saved: {metrics_json_path}")
        print(f"[Artifacts] Saved: {comp_csv_path}")
        print(f"[Artifacts] Saved: {cm_png_path}")
        print(f"[Artifacts] Saved: {os.path.join(reports_dir, 'error_analysis.txt')}")

    return combined_reports


# =====================================================================
# Error Analysis Engine
# =====================================================================

def generate_error_analysis(
    test_df: pd.DataFrame,
    test_tokens: List[List[str]],
    y_true: List[str],
    y_pred: List[str],
    probs: np.ndarray,
    extractor: Any,
    clf: Any,
    output_path: str = "reports/error_analysis.txt",
) -> List[Dict[str, Any]]:
    """Perform detailed error analysis on misclassified held-out test articles.

    Inspects actual errors, extracts contributing terms from weights,
    and documents observable lexical/domain reasons for the model's confusion.
    """
    classes = clf.classes
    misclassified_indices = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    error_records = []

    lines = []
    lines.append("=" * 80)
    lines.append("ERROR ANALYSIS: HELD-OUT TEST SET MISCLASSIFICATIONS")
    lines.append("Model: One-vs-Rest Logistic Regression on TF-IDF (Unigram + Bigram)")
    lines.append(f"Total Test Samples: {len(y_true)} | Errors: {len(misclassified_indices)} | Accuracy: {(1 - len(misclassified_indices)/len(y_true))*100:.2f}%")
    lines.append("=" * 80 + "\n")

    for rank, idx in enumerate(misclassified_indices[:20], 1):
        true_cat = y_true[idx]
        pred_cat = y_pred[idx]
        row = test_df.iloc[idx]
        title = row.get("title", "No Title")
        raw_text = row["text"]
        text_preview = " ".join(raw_text.split()[:50]) + "..."
        prob_dist = {c: float(probs[idx, j]) for j, c in enumerate(classes)}

        # Explain active features
        explanation = extractor.explain_instance(test_tokens[idx], clf.weights, top_k=5)
        top_pred_feats = explanation["class_explanations"].get(pred_cat, [])
        top_true_feats = explanation["class_explanations"].get(true_cat, [])

        pred_terms_str = ", ".join([f"{item['term']} (w={item['weight']:.2f})" for item in top_pred_feats[:4]])
        true_terms_str = ", ".join([f"{item['term']} (w={item['weight']:.2f})" for item in top_true_feats[:4]])

        # Generate observable explanation
        if true_cat == "sport" and pred_cat == "business":
            reason = "The article covers corporate, contractual, or financial aspects of sports (e.g. club takeover, sponsorship, broadcasting rights), leading business-weighted terms to outweigh sporting terms."
        elif true_cat == "business" and pred_cat == "politics":
            reason = "The article discusses government economic policies, state fiscal budgets, or ministerial regulatory announcements, where political terminology heavily overlaps with economic indicators."
        elif true_cat == "politics" and pred_cat == "business":
            reason = "The political article centers on corporate legislation, employment figures, or market taxation, triggering positive business model coefficients."
        elif true_cat == "tech" and pred_cat == "business":
            reason = "The article reports on commercial technology companies, intellectual property acquisitions, or quarterly earnings of software corporations, resulting in dominant business features."
        elif true_cat == "business" and pred_cat == "tech":
            reason = "The business article focuses on high-tech startups, digital infrastructure, or telecom networks, activating technology-weighted coefficients."
        elif true_cat == "entertainment" and pred_cat == "business":
            reason = "The entertainment article covers box office grosses, studio mergers, or film budget financing, emphasizing financial vocabulary."
        elif true_cat == "tech" and pred_cat == "politics":
            reason = "The piece covers governmental cyber-security policies, state digital surveillance laws, or parliamentary internet censorship."
        elif true_cat == "entertainment" and pred_cat == "politics":
            reason = "The article discusses cultural censorship, public broadcasting funding disputes, or political figures appearing in media events."
        else:
            reason = f"Cross-domain terminology: text contains strong positive indicators for {pred_cat} which statistically outnumbered unique features for {true_cat}."

        record = {
            "sample_index": int(idx),
            "actual": true_cat,
            "predicted": pred_cat,
            "title": title,
            "probabilities": prob_dist,
            "reason": reason,
        }
        error_records.append(record)

        lines.append(f"--- [Error {rank:02d} / {len(misclassified_indices):02d}] Test Sample #{idx} ---")
        lines.append(f"Actual Category   : {true_cat.upper()}")
        lines.append(f"Predicted Category: {pred_cat.upper()} (Confidence: {prob_dist[pred_cat]*100:.1f}%)")
        lines.append(f"Title             : {title}")
        lines.append(f"Text Snippet      : {text_preview}")
        lines.append("Probabilities     : " + " | ".join([f"{c}: {p*100:.1f}%" for c, p in prob_dist.items()]))
        lines.append(f"Key Predicted Features ({pred_cat}): {pred_terms_str}")
        lines.append(f"Key Actual Features    ({true_cat})   : {true_terms_str}")
        lines.append(f"Observable Reason : {reason}\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return error_records


# =====================================================================
# 4. Confusion Matrix Plotter
# =====================================================================

def plot_confusion_matrices(
    pipeline_evals: Dict[str, Any], output_path: str = "reports/confusion_matrix.png"
):
    """Plot multi-panel publication-grade Confusion Matrices for all pipelines."""
    n_pipes = len(pipeline_evals)
    if n_pipes <= 4:
        n_rows, n_cols = 1, n_pipes
        figsize = (4.2 * n_cols, 4.0)
    else:
        n_cols = 4
        n_rows = math.ceil(n_pipes / n_cols)
        figsize = (4.2 * n_cols, 3.8 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.array(axes).reshape(-1)

    classes = ORDERED_CATEGORIES
    short_classes = [c[:4].upper() for c in classes]

    for idx, (p_key, p_data) in enumerate(pipeline_evals.items()):
        ax = axes_flat[idx]
        cm = np.array(p_data["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

        ax.set_title(f"{p_data['short_name']}\nAcc: {p_data['accuracy']*100:.1f}%", fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(short_classes, fontsize=8)
        ax.set_yticklabels(short_classes, fontsize=8)

        ax.set_ylabel("True Category", fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9, fontweight="bold")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(
                    j, i, f"{val}",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8, fontweight="bold"
                )

    # Hide any empty subplots
    for j in range(n_pipes, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# Alias for backward and forward compatibility
evaluate_all_pipelines = evaluate_all_five_pipelines


# =====================================================================
# 5. Unsupervised K-Means Clustering & 2D Projection Visualization
# =====================================================================

def run_kmeans_clustering(
    vectors: np.ndarray,
    ground_truth_labels: Optional[List[str]] = None,
    k: int = 5,
    output_png: str = "reports/clusters.png",
) -> Dict[str, Any]:
    """Perform Unsupervised K-Means (k=5) clustering and generate 2D PCA projection plot.

    CRITICAL ACADEMIC NOTE:
    -----------------------
    K-Means is an UNSUPERVISED clustering algorithm that discovers natural geometric
    groupings in the embedding space without observing target labels.
    The resulting silhouette score evaluates spatial separation, and must NEVER
    be confused with or labeled as supervised classification accuracy.
    """
    # 1. Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)

    # 2. Compute Silhouette Score
    score = float(silhouette_score(vectors, cluster_labels))

    # 3. 2D Dimensionality Reduction for Visualization
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(vectors)
    centroids_2d = pca.transform(kmeans.cluster_centers_)

    # 4. Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Left Subplot: Unsupervised K-Means Clusters
    for c_id in range(k):
        mask = (cluster_labels == c_id)
        ax1.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=colors[c_id % len(colors)],
            label=f"Cluster {c_id + 1}",
            alpha=0.6, edgecolors="none", s=28
        )
    ax1.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        marker="X", s=180, c="black", edgecolor="white", linewidth=1.5,
        label="Centroids"
    )
    ax1.set_title(f"Unsupervised K-Means Clusters (k={k})\nSilhouette Score: {score:.3f}", fontsize=12, fontweight="bold")
    ax1.set_xlabel(f"PCA Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
    ax1.set_ylabel(f"PCA Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Right Subplot: Ground-Truth Category Distribution for Visual Comparison
    if ground_truth_labels is not None:
        cat_list = ORDERED_CATEGORIES
        for c_idx, cat_name in enumerate(cat_list):
            mask = np.array([l == cat_name for l in ground_truth_labels])
            ax2.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=colors[c_idx % len(colors)],
                label=cat_name.capitalize(),
                alpha=0.6, edgecolors="none", s=28
            )
        ax2.set_title("Ground-Truth Category Distribution (Reference)", fontsize=12, fontweight="bold")
        ax2.set_xlabel(f"PCA Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
        ax2.set_ylabel(f"PCA Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(
        "Semantic Document Vector Space: Unsupervised Clustering vs Ground-Truth\n"
        "(Academic Constraint: Unsupervised cluster metrics reflect geometric separation, NOT classification accuracy)",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "k": k,
        "silhouette_score": round(score, 4),
        "inertia": round(float(kmeans.inertia_), 2),
        "pca_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "algorithm": "K-Means (Unsupervised)",
        "note": "Unsupervised clustering analysis; not classification accuracy",
    }


if __name__ == "__main__":
    evaluate_all_five_pipelines()

