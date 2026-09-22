"""One-vs-Rest (OvR) Multi-Class Logistic Regression from Scratch using Pure NumPy.

Implements formula-based discriminative classification:
- Sigmoid activation with numerical clipping:
    z = clip(z, -250, 250)
    sigma(z) = 1 / (1 + exp(-z))
- One-vs-Rest (OvR) multi-class strategy:
    5 independent binary classifiers (one per BBC category)
    Target y_c = 1 for class c, 0 for all other classes
- Batch Gradient Descent (for m examples):
    dw = (1 / m) * X^T * (prediction - y)
    db = (1 / m) * sum(prediction - y)
    W := W - alpha * dw
    b := b - alpha * db
- Multi-class scoring and probability normalization across categories:
    p_c = sigma(X W_c + b_c)
    P(c | D) = p_c / sum_{k} p_k
- Training across 4 document-vector representations (Word2Vec/GloVe x Mean/TF-IDF)
- Serialization to models/logistic_regression_weights.pkl
"""

import os
import sys
import pickle
import math
from typing import List, Tuple, Dict, Union, Optional, Any
import numpy as np

# Ensure project root is in sys.path when executed directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import TextPreprocessor, load_dataset, stratified_train_test_split
from src.embeddings import (
    GloVeEmbeddings,
    compute_training_idf,
    vectorize_corpus_mean,
    vectorize_corpus_tfidf,
    generate_offline_fallback_glove_cache,
)

try:
    from src.word2vec_scratch import CustomWord2Vec
except ImportError:
    CustomWord2Vec = None


VALID_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]


def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Numerically stable Sigmoid activation with clipping.

    Formula:
        z = clip(z, -250, 250)
        sigma(z) = 1 / (1 + exp(-z))
    """
    z_clipped = np.clip(z, -250.0, 250.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


class BinaryLogisticRegression:
    """Single binary logistic regression classifier trained via Batch Gradient Descent."""

    def __init__(
        self,
        learning_rate: float = 0.5,
        n_iterations: int = 1000,
        l2_reg: float = 0.0001,
        seed: int = 42,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_reg = l2_reg
        self.seed = seed

        self.w: Optional[np.ndarray] = None  # shape (d,)
        self.b: float = 0.0
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Train binary classifier using Batch Gradient Descent.

        Formulas:
            prediction = sigma(XW + b)
            dw = (1 / m) * X^T * (prediction - y) + (l2_reg * W)
            db = (1 / m) * sum(prediction - y)
            W := W - learning_rate * dw
            b := b - learning_rate * db

        Args:
            X: 2D array of shape (m, d).
            y: 1D binary target array of shape (m,), values in {0, 1}.

        Returns:
            List of binary cross-entropy losses per 100 iterations.
        """
        rng = np.random.RandomState(self.seed)
        m, d = X.shape

        # Initialize weights with small normal values and bias at 0
        self.w = (rng.randn(d) * 0.01).astype(np.float32)
        self.b = 0.0
        self.loss_history = []

        y = y.astype(np.float32)

        for i in range(self.n_iterations):
            # Forward pass: z = Xw + b
            z = np.dot(X, self.w) + self.b
            predictions = sigmoid(z)

            # Error gradient: prediction - y
            errors = predictions - y

            # Batch gradient: dw = (1/m) * X^T * error + l2_reg * w
            dw = (1.0 / m) * np.dot(X.T, errors) + (self.l2_reg * self.w)
            db = float((1.0 / m) * np.sum(errors))

            # Gradient descent update
            self.w -= (self.learning_rate * dw).astype(np.float32)
            self.b -= float(self.learning_rate * db)

            # Record binary cross-entropy loss periodically
            if (i + 1) % 100 == 0 or i == 0:
                eps = 1e-12
                p_clipped = np.clip(predictions, eps, 1.0 - eps)
                bce = - (1.0 / m) * np.sum(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
                self.loss_history.append(float(bce))

        return self.loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute sigmoid probability score: p = sigma(Xw + b)."""
        if self.w is None:
            raise ValueError("Classifier is not trained yet.")
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)


class OneVsRestLogisticRegression:
    """Multi-Class One-vs-Rest (OvR) Logistic Regression from scratch."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        learning_rate: float = 0.5,
        n_iterations: int = 1000,
        l2_reg: float = 0.0001,
        seed: int = 42,
    ):
        self.classes = list(classes) if classes is not None else list(VALID_CATEGORIES)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_reg = l2_reg
        self.seed = seed

        # Five independent binary classifiers
        self.classifiers: Dict[str, BinaryLogisticRegression] = {}
        self.weights: Dict[str, np.ndarray] = {}
        self.biases: Dict[str, float] = {}

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def fit(
        self, X: np.ndarray, y: Union[List[str], np.ndarray], verbose: bool = False
    ) -> Dict[str, List[float]]:
        """Fit 5 independent binary classifiers for One-vs-Rest classification.

        For each category c in universal classes:
            target = 1 if y == c else 0
            train binary classifier using batch gradient descent

        Args:
            X: Feature matrix of shape (m, d).
            y: Category labels of length m.
            verbose: Whether to log training progress.

        Returns:
            Dictionary of loss trajectories for each binary classifier.
        """
        y_arr = np.array(y)
        all_losses: Dict[str, List[float]] = {}

        if verbose:
            print(f"[OvR Logistic Regression] Training on {X.shape[0]} samples ({X.shape[1]} features) across {self.n_classes} classes...")

        for idx, category in enumerate(self.classes):
            # Binary target: 1 for current class, 0 for all others
            binary_target = (y_arr == category).astype(np.float32)

            classifier = BinaryLogisticRegression(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                l2_reg=self.l2_reg,
                seed=self.seed + idx,
            )
            losses = classifier.fit(X, binary_target)

            self.classifiers[category] = classifier
            self.weights[category] = classifier.w.copy()
            self.biases[category] = float(classifier.b)
            all_losses[category] = losses

            if verbose:
                final_loss = losses[-1] if losses else 0.0
                print(f"  Class [{category:15s}]: Initial Loss={losses[0]:.4f} -> Final Loss={final_loss:.4f}")

        return all_losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized probability distribution across categories.

        For each category c:
            z_c = X * w_c + b_c
            p_c = sigmoid(z_c)

        Normalize:
            P(c | X) = p_c / sum_{k} p_k

        Args:
            X: Feature matrix of shape (m, d) or (d,).

        Returns:
            Normalized 2D array of shape (m, n_classes) summing to 1.0 per row.
        """
        if not self.classifiers:
            raise ValueError("Model has not been trained yet.")

        X_mat = np.atleast_2d(X)
        m = X_mat.shape[0]

        # Gather raw sigmoid scores for all 5 categories
        raw_scores = np.zeros((m, self.n_classes), dtype=np.float64)
        for j, category in enumerate(self.classes):
            raw_scores[:, j] = self.classifiers[category].predict_proba(X_mat)

        # Normalize raw scores so each row represents a valid probability distribution
        row_sums = np.sum(raw_scores, axis=1, keepdims=True)
        # Avoid zero division
        row_sums[row_sums == 0.0] = 1.0

        normalized_probs = raw_scores / row_sums
        return normalized_probs.astype(np.float32)

    def predict(self, X: np.ndarray) -> List[str]:
        """Predict the category with the highest normalized probability.

        Args:
            X: Feature matrix of shape (m, d) or (d,).

        Returns:
            List of predicted category strings.
        """
        probs = self.predict_proba(X)
        top_indices = np.argmax(probs, axis=1)
        return [self.classes[idx] for idx in top_indices]

    def score(self, X: np.ndarray, y: Union[List[str], np.ndarray]) -> float:
        """Compute classification accuracy on provided samples."""
        predictions = self.predict(X)
        y_arr = np.array(y)
        return float(np.mean(predictions == y_arr))

    def save(self, filepath: str = "models/logistic_regression_weights.pkl"):
        """Serialize trained weights, biases, and hyperparameters to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "classes": self.classes,
            "weights": self.weights,
            "biases": self.biases,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "l2_reg": self.l2_reg,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)
        print(f"[OvR Logistic Regression] Model weights successfully saved to: {filepath}")

    @classmethod
    def load(
        cls,
        filepath: str = "models/logistic_regression_weights.pkl",
        pipeline_key: Optional[str] = None,
    ) -> "OneVsRestLogisticRegression":
        """Load trained weights, biases, and classifiers from local pickle file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        if "all_pipelines" in payload:
            target_key = pipeline_key if pipeline_key is not None else payload.get("primary_pipeline", "tfidf_unigram_bigram")
            if target_key in payload["all_pipelines"]:
                pipe_data = payload["all_pipelines"][target_key]
                model = cls(
                    classes=pipe_data["classes"],
                    learning_rate=payload.get("learning_rate", 0.5),
                    n_iterations=payload.get("n_iterations", 1000),
                    l2_reg=payload.get("l2_reg", 0.0001),
                )
                model.weights = pipe_data["weights"]
                model.biases = pipe_data["biases"]
                for cat in model.classes:
                    clf = BinaryLogisticRegression(
                        learning_rate=model.learning_rate,
                        n_iterations=model.n_iterations,
                        l2_reg=model.l2_reg,
                    )
                    clf.w = model.weights[cat].copy()
                    clf.b = float(model.biases[cat])
                    model.classifiers[cat] = clf
                return model

        # Fallback for single-model payload
        model = cls(
            classes=payload["classes"],
            learning_rate=payload.get("learning_rate", 0.5),
            n_iterations=payload.get("n_iterations", 1000),
            l2_reg=payload.get("l2_reg", 0.0001),
        )
        model.weights = payload["weights"]
        model.biases = payload["biases"]

        for cat in model.classes:
            clf = BinaryLogisticRegression(
                learning_rate=model.learning_rate,
                n_iterations=model.n_iterations,
                l2_reg=model.l2_reg,
            )
            clf.w = model.weights[cat].copy()
            clf.b = float(model.biases[cat])
            model.classifiers[cat] = clf

        return model


# =====================================================================
# Multi-Pipeline Training & Evaluation Engine
# =====================================================================

def train_and_evaluate_all_pipelines(
    csv_path: str = "data/bbc_news.csv",
    save_path: str = "models/logistic_regression_weights.pkl",
    learning_rate: float = 0.5,
    n_iterations: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate OvR Logistic Regression across all representation pipelines.

    Evaluates 6 Logistic Regression configurations:
      1. Pipeline B: Custom Word2Vec + Mean Aggregation
      2. Pipeline C: Custom Word2Vec + TF-IDF Weighted Aggregation
      3. Pipeline D: Pretrained GloVe + Mean Aggregation
      4. Pipeline E: Pretrained GloVe + TF-IDF Weighted Aggregation
      5. Pipeline F: TF-IDF (Unigram)
      6. Pipeline G: TF-IDF (Unigram + Bigram)
    """
    from src.tfidf_extractor import TfidfNGramFeatureExtractor
    import json

    if verbose:
        print("=" * 70)
        print("TRAINING MULTI-CLASS ONE-VS-REST LOGISTIC REGRESSION FROM SCRATCH")
        print("=" * 70)

    # 1. Load Dataset & Perform Zero-Leakage Stratified Split
    df = load_dataset(csv_path)
    train_df, test_df = stratified_train_test_split(df, test_size=0.2, random_state=42)

    preprocessor = TextPreprocessor()
    if verbose:
        print(f"Tokenizing {len(train_df):,} train articles and {len(test_df):,} test articles...")

    train_tokens = [preprocessor.preprocess(t, return_tokens=True) for t in train_df["text"]]
    test_tokens = [preprocessor.preprocess(t, return_tokens=True) for t in test_df["text"]]

    y_train = train_df["category"].tolist()
    y_test = test_df["category"].tolist()

    # 2. Strict Zero-Data-Leakage IDF Weights for Embeddings
    idf_weights, default_idf = compute_training_idf(train_tokens, smooth=True)

    # 3. Load Embedding Models (Word2Vec + GloVe)
    w2v_model: Optional[Any] = None
    if CustomWord2Vec is not None and os.path.exists("models/custom_word2vec.pt"):
        try:
            w2v = CustomWord2Vec()
            w2v.load("models/custom_word2vec.pt", "models/vocab.json")
            w2v_model = w2v
            if verbose:
                print(f"[Word2Vec] Loaded Custom Word2Vec model ({w2v.vocab_size:,} words, {w2v.embedding_dim}d)")
        except Exception as e:
            if verbose:
                print(f"[Word2Vec Warning] Could not load Custom Word2Vec: {e}")

    glove_npy = "models/pretrained_embeddings.npy"
    glove_vocab = "models/pretrained_embeddings_vocab.json"
    if not os.path.exists(glove_npy) or not os.path.exists(glove_vocab):
        glove_model = generate_offline_fallback_glove_cache(glove_npy, glove_vocab)
    else:
        glove_model = GloVeEmbeddings.load(glove_npy, glove_vocab)

    if verbose:
        print(f"[GloVe] Loaded GloVe model ({glove_model.vocab_size:,} words, {glove_model.embedding_dim}d)")

    # 4. Prepare TF-IDF Feature Extractors (Fitted strictly on training data)
    if verbose:
        print("\n[TF-IDF] Fitting Unigram Extractor (min_df=2, max_features=5000)...")
    ext_unigram = TfidfNGramFeatureExtractor(ngram_range=(1, 1), min_df=2, max_features=5000)
    X_train_uni = ext_unigram.fit_transform(train_tokens)
    X_test_uni = ext_unigram.transform(test_tokens)
    ext_unigram.save("models/tfidf_unigram_extractor.pkl")

    if verbose:
        print(f"[TF-IDF] Fitting Unigram+Bigram Extractor (min_df=2, max_features=8000)...")
    ext_bigram = TfidfNGramFeatureExtractor(ngram_range=(1, 2), min_df=2, max_features=8000)
    X_train_bi = ext_bigram.fit_transform(train_tokens)
    X_test_bi = ext_bigram.transform(test_tokens)
    ext_bigram.save("models/tfidf_extractor.pkl")

    # 5. Define representation pipelines to train
    pipeline_configs = []

    # Embedding pipelines
    if w2v_model is not None:
        pipeline_configs.append(("w2v_mean", "Custom Word2Vec (Mean)", "emb", w2v_model, "mean"))
        pipeline_configs.append(("w2v_tfidf", "Custom Word2Vec (TF-IDF)", "emb", w2v_model, "tfidf"))

    pipeline_configs.append(("glove_mean", "Pretrained GloVe (Mean)", "emb", glove_model, "mean"))
    pipeline_configs.append(("glove_tfidf", "Pretrained GloVe (TF-IDF)", "emb", glove_model, "tfidf"))

    # TF-IDF pipelines
    pipeline_configs.append(("tfidf_unigram", "TF-IDF (Unigram)", "direct", (X_train_uni, X_test_uni), None))
    pipeline_configs.append(("tfidf_unigram_bigram", "TF-IDF (Unigram + Bigram)", "direct", (X_train_bi, X_test_bi), None))

    results: Dict[str, Any] = {
        "pipeline_accuracies": {},
        "models": {},
        "primary_pipeline": "tfidf_unigram_bigram",
    }

    trained_models: Dict[str, OneVsRestLogisticRegression] = {}

    for pipe_key, pipe_name, p_type, model_or_data, agg_type in pipeline_configs:
        if verbose:
            print(f"\n--- Training Pipeline: {pipe_name} ---")

        if p_type == "emb":
            emb_model = model_or_data
            if agg_type == "mean":
                X_tr = vectorize_corpus_mean(train_tokens, emb_model)
                X_te = vectorize_corpus_mean(test_tokens, emb_model)
            else:
                X_tr = vectorize_corpus_tfidf(train_tokens, emb_model, idf_weights, default_idf=default_idf)
                X_te = vectorize_corpus_tfidf(test_tokens, emb_model, idf_weights, default_idf=default_idf)
        else:
            X_tr, X_te = model_or_data

        clf = OneVsRestLogisticRegression(
            classes=VALID_CATEGORIES,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            l2_reg=0.0001,
            seed=42,
        )
        clf.fit(X_tr, y_train, verbose=False)

        train_acc = clf.score(X_tr, y_train)
        test_acc = clf.score(X_te, y_test)

        trained_models[pipe_key] = clf
        results["pipeline_accuracies"][pipe_key] = {
            "name": pipe_name,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
        }

        if verbose:
            print(f"  Result: Train Acc = {train_acc * 100:.2f}% | Test Acc = {test_acc * 100:.2f}%")

    # Select primary model
    primary_key = results["primary_pipeline"]
    primary_clf = trained_models[primary_key]

    # 6. Extract learned feature weights per category for explainability
    top_features_per_category: Dict[str, List[Dict[str, Any]]] = {}
    feature_names = ext_bigram.feature_names_
    for cat in primary_clf.classes:
        w = primary_clf.weights[cat]  # shape (8000,)
        # Sort indices by positive weight descending
        top_indices = np.argsort(w)[::-1][:25]
        cat_top = []
        for idx in top_indices:
            cat_top.append({
                "term": feature_names[idx],
                "weight": round(float(w[idx]), 4),
            })
        top_features_per_category[cat] = cat_top

    os.makedirs("reports", exist_ok=True)
    with open("reports/top_features_per_category.json", "w", encoding="utf-8") as f:
        json.dump(top_features_per_category, f, indent=2)

    if verbose:
        print("\n[Explainability] Saved top learned positive feature weights to reports/top_features_per_category.json")
        for cat, feats in top_features_per_category.items():
            top_terms = [f"{item['term']} ({item['weight']})" for item in feats[:5]]
            print(f"  {cat.capitalize():15s}: {', '.join(top_terms)}")

    # 7. Save comprehensive bundle preserving weights for all representations
    bundle_payload = {
        "classes": primary_clf.classes,
        "weights": primary_clf.weights,
        "biases": primary_clf.biases,
        "learning_rate": primary_clf.learning_rate,
        "n_iterations": primary_clf.n_iterations,
        "l2_reg": primary_clf.l2_reg,
        "primary_pipeline": primary_key,
        "all_pipelines": {
            k: {
                "classes": m.classes,
                "weights": m.weights,
                "biases": m.biases,
            }
            for k, m in trained_models.items()
        },
        "accuracies": results["pipeline_accuracies"],
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(bundle_payload, f)

    if verbose:
        print(f"\n[Artifact] Saved serialized weights bundle to: {save_path}")

    return results


if __name__ == "__main__":
    train_and_evaluate_all_pipelines()

