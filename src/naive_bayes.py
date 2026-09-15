"""Multinomial Naive Bayes Classifier from Scratch using NumPy.

Implements formula-based Bayesian text classification with:
- Class prior estimation: P(c) = N_c / N
- Word likelihood with Laplace (Add-1) smoothing:
    P(w | c) = (count(w, c) + 1) / (total words in class c + |V|)
- Document scoring in log space:
    log P(c | D) = log P(c) + sum_{i} log P(w_i | c)
- Numerically stable normalized probabilities via Log-Sum-Exp:
    P(c | D) = exp(s_c - max(s)) / sum_{c'} exp(s_{c'} - max(s))
- Bag-of-Words count representation and model serialization.
"""

import os
import sys
import pickle
from typing import List, Tuple, Dict, Union, Optional
import numpy as np

# Ensure project root is in sys.path when executed directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import TextPreprocessor, load_dataset, stratified_train_test_split


def logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    """Compute the log of the sum of exponentials in a numerically stable manner.

    Formula:
        logsumexp(a) = m + log(sum(exp(a - m)))
    where m = max(a) along the specified axis.

    Args:
        a: Input numpy array of log values.
        axis: Axis along which the log-sum-exp is computed.
        keepdims: Whether to retain the reduced dimension.

    Returns:
        Numpy array of log-sum-exp values.
    """
    max_val = np.max(a, axis=axis, keepdims=True)
    # Prevent -inf - (-inf) = nan if an entire row is -inf
    max_val_safe = np.where(np.isneginf(max_val), 0.0, max_val)
    sum_exp = np.sum(np.exp(a - max_val_safe), axis=axis, keepdims=True)
    lse = max_val_safe + np.log(sum_exp)
    if not keepdims:
        lse = np.squeeze(lse, axis=axis)
    return lse


class BagOfWords:
    """Formula-based Bag-of-Words count vectorizer implemented in NumPy."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """Initialize BagOfWords vectorizer.

        Args:
            vocab: Optional pre-constructed dictionary mapping token to column index.
        """
        self.vocab: Dict[str, int] = vocab if vocab is not None else {}
        self.id_to_word: Dict[int, str] = {i: w for w, i in self.vocab.items()} if vocab else {}
        self.vocab_size: int = len(self.vocab)

    def fit(self, documents: List[List[str]]) -> "BagOfWords":
        """Build vocabulary mapping from a tokenized corpus.

        Tokens are sorted alphabetically for deterministic feature indices.

        Args:
            documents: List of tokenized documents (each document is a list of string tokens).

        Returns:
            self
        """
        unique_tokens = set()
        for doc in documents:
            for token in doc:
                unique_tokens.add(token)

        sorted_tokens = sorted(list(unique_tokens))
        self.vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.id_to_word = {idx: token for idx, token in enumerate(sorted_tokens)}
        self.vocab_size = len(self.vocab)
        return self

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Transform tokenized documents into a 2D NumPy count matrix.

        Args:
            documents: List of tokenized documents.

        Returns:
            np.ndarray of shape (N_documents, |V|) containing term frequencies.
        """
        n_docs = len(documents)
        X = np.zeros((n_docs, self.vocab_size), dtype=np.int32)

        for i, doc in enumerate(documents):
            for token in doc:
                idx = self.vocab.get(token)
                if idx is not None:
                    X[i, idx] += 1

        return X

    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        """Fit vocabulary and transform tokenized documents into a count matrix.

        Args:
            documents: List of tokenized documents.

        Returns:
            Count matrix of shape (N_documents, |V|).
        """
        self.fit(documents)
        return self.transform(documents)


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes classifier built from scratch using NumPy."""

    def __init__(self, alpha: float = 1.0):
        """Initialize the Multinomial Naive Bayes classifier.

        Args:
            alpha: Laplace (Add-1) smoothing parameter (alpha >= 0, default: 1.0).
        """
        if alpha < 0:
            raise ValueError(f"Smoothing parameter alpha must be non-negative, got {alpha}")
        self.alpha = float(alpha)

        # Trained attributes
        self.classes_: Optional[np.ndarray] = None
        self.class_to_idx_: Optional[Dict[str, int]] = None
        self.class_counts_: Optional[np.ndarray] = None
        self.total_docs_: int = 0
        self.class_priors_: Optional[np.ndarray] = None
        self.log_class_priors_: Optional[np.ndarray] = None
        self.feature_counts_: Optional[np.ndarray] = None
        self.class_word_totals_: Optional[np.ndarray] = None
        self.vocab_size_: int = 0
        self.feature_log_prob_: Optional[np.ndarray] = None
        self.vocab_: Optional[Dict[str, int]] = None
        self.vectorizer_: Optional[BagOfWords] = None

    def fit_counts(self, X: np.ndarray, y: Union[np.ndarray, List[str]]) -> "MultinomialNaiveBayes":
        """Train the Naive Bayes model on a Bag-of-Words count matrix.

        Formulas:
            Class Prior:
                P(c) = N_c / N
                log P(c) = ln(N_c / N)

            Word Likelihood with Laplace (Add-1) Smoothing:
                P(w | c) = (count(w, c) + alpha) / (total words in class c + alpha * |V|)
                log P(w | c) = ln(count(w, c) + alpha) - ln(total words in class c + alpha * |V|)

        Args:
            X: Count matrix of shape (N_samples, |V|) where X[i, j] = count of word j in doc i.
            y: Array-like of class labels of shape (N_samples,).

        Returns:
            self
        """
        y_arr = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D count matrix X, got shape {X.shape}")
        if len(X) != len(y_arr):
            raise ValueError(f"Mismatch between X samples ({len(X)}) and y labels ({len(y_arr)})")

        self.total_docs_ = len(y_arr)
        if self.total_docs_ == 0:
            raise ValueError("Cannot train on an empty dataset.")

        # 1. Classes and Class Counts
        self.classes_, self.class_counts_ = np.unique(y_arr, return_counts=True)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        n_classes = len(self.classes_)
        self.vocab_size_ = X.shape[1]

        # 2. Class Prior: P(c) = N_c / N
        self.class_priors_ = self.class_counts_ / float(self.total_docs_)
        self.log_class_priors_ = np.log(self.class_priors_)

        # 3. Word Occurrences per Class: count(w, c)
        self.feature_counts_ = np.zeros((n_classes, self.vocab_size_), dtype=np.float64)
        for class_name, class_idx in self.class_to_idx_.items():
            mask = (y_arr == class_name)
            self.feature_counts_[class_idx, :] = np.sum(X[mask], axis=0)

        # 4. Total Words in Each Class: sum_{w'} count(w', c)
        self.class_word_totals_ = np.sum(self.feature_counts_, axis=1)

        # 5. Likelihood with Add-alpha (Add-1) smoothing:
        # P(w | c) = (count(w, c) + alpha) / (total words in c + alpha * |V|)
        smoothed_numerator = self.feature_counts_ + self.alpha
        smoothed_denominator = self.class_word_totals_[:, np.newaxis] + (self.alpha * float(self.vocab_size_))

        # Log Likelihood: log P(w | c)
        self.feature_log_prob_ = np.log(smoothed_numerator) - np.log(smoothed_denominator)
        return self

    def fit(
        self,
        documents: List[Union[List[str], str]],
        y: Union[np.ndarray, List[str]],
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> "MultinomialNaiveBayes":
        """Fit model directly on documents (either raw text strings or token lists).

        Args:
            documents: List of documents (either strings or list of tokens).
            y: Target class labels.
            preprocessor: Optional TextPreprocessor instance for string cleaning.

        Returns:
            self
        """
        # Ensure documents are tokenized
        tokenized_docs = self._ensure_tokenized(documents, preprocessor)

        # Fit internal BagOfWords vectorizer
        self.vectorizer_ = BagOfWords()
        X = self.vectorizer_.fit_transform(tokenized_docs)
        self.vocab_ = self.vectorizer_.vocab

        return self.fit_counts(X, y)

    def _ensure_tokenized(
        self,
        documents: List[Union[List[str], str]],
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> List[List[str]]:
        """Standardize document input into token lists."""
        if not documents:
            return []

        if isinstance(documents[0], str):
            prep = preprocessor or TextPreprocessor()
            return [prep.preprocess(doc, return_tokens=True) for doc in documents]
        else:
            return [list(doc) for doc in documents]

    def _to_count_matrix(
        self,
        documents: Union[List[Union[List[str], str]], np.ndarray],
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> np.ndarray:
        """Convert input documents or count matrix into a valid count matrix."""
        if isinstance(documents, np.ndarray) and documents.ndim == 2:
            if documents.shape[1] != self.vocab_size_:
                raise ValueError(
                    f"Count matrix feature dimension {documents.shape[1]} does not match "
                    f"model vocabulary size {self.vocab_size_}."
                )
            return documents

        if self.vectorizer_ is None:
            raise ValueError(
                "Model was fitted on count matrix without internal BagOfWords. "
                "Please pass a 2D count matrix X or fit using raw documents."
            )

        tokenized = self._ensure_tokenized(documents, preprocessor)
        return self.vectorizer_.transform(tokenized)

    def predict_log_proba(
        self,
        documents: Union[List[Union[List[str], str]], np.ndarray],
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> np.ndarray:
        """Compute unnormalized log-posterior scores for documents across all classes.

        Formula:
            log P(c | D) = log P(c) + sum_{i=1}^k log P(w_i | c)
            Vectorized: log_scores = X @ (feature_log_prob_)^T + log_class_priors_

        Args:
            documents: List of raw documents / tokens, or 2D count matrix.
            preprocessor: Optional TextPreprocessor for raw text.

        Returns:
            np.ndarray of shape (N_samples, N_classes) containing log posterior scores.
        """
        if self.feature_log_prob_ is None or self.log_class_priors_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() or fit_counts() first.")

        X = self._to_count_matrix(documents, preprocessor)
        # Vectorized document scoring in log space:
        # X shape: (N, |V|), feature_log_prob_ shape: (C, |V|)
        # Matrix multiply: (N, |V|) @ (|V|, C) -> (N, C)
        log_scores = np.dot(X, self.feature_log_prob_.T) + self.log_class_priors_
        return log_scores

    def predict_proba(
        self,
        documents: Union[List[Union[List[str], str]], np.ndarray],
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> np.ndarray:
        """Compute normalized class posterior probabilities using numerically stable Log-Sum-Exp.

        Formula:
            P(c | D) = exp(log P(c | D) - logsumexp(log P(* | D)))
                     = exp(s_c - max(s)) / sum_{c'} exp(s_{c'} - max(s))

        Args:
            documents: List of raw documents / tokens, or 2D count matrix.
            preprocessor: Optional TextPreprocessor for raw text.

        Returns:
            np.ndarray of shape (N_samples, N_classes) where each row sums to 1.0.
        """
        log_scores = self.predict_log_proba(documents, preprocessor)
        # Numerically stable Log-Sum-Exp normalization
        max_scores = np.max(log_scores, axis=1, keepdims=True)
        shifted_scores = log_scores - max_scores
        exp_scores = np.exp(shifted_scores)
        sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
        probabilities = exp_scores / sum_exp
        return probabilities

    def predict(
        self,
        documents: Union[List[Union[List[str], str]], np.ndarray],
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> np.ndarray:
        """Predict the most probable class for each document.

        Formula:
            hat{c} = argmax_{c in C} log P(c | D)

        Args:
            documents: List of raw documents / tokens, or 2D count matrix.
            preprocessor: Optional TextPreprocessor for raw text.

        Returns:
            np.ndarray of shape (N_samples,) containing predicted class label strings.
        """
        log_scores = self.predict_log_proba(documents, preprocessor)
        best_indices = np.argmax(log_scores, axis=1)
        return self.classes_[best_indices]

    def score(
        self,
        documents: Union[List[Union[List[str], str]], np.ndarray],
        y: Union[np.ndarray, List[str]],
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> float:
        """Calculate mean classification accuracy.

        Args:
            documents: List of documents / tokens, or 2D count matrix.
            y: Ground-truth class labels.
            preprocessor: Optional TextPreprocessor for raw text.

        Returns:
            Accuracy float in [0.0, 1.0].
        """
        y_true = np.asarray(y)
        y_pred = self.predict(documents, preprocessor)
        return float(np.mean(y_pred == y_true))

    def save(self, filepath: str):
        """Serialize and save trained model state to disk.

        Args:
            filepath: Destination file path (e.g., 'models/naive_bayes_weights.pkl').
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = {
            "alpha": self.alpha,
            "classes_": self.classes_,
            "class_to_idx_": self.class_to_idx_,
            "class_counts_": self.class_counts_,
            "total_docs_": self.total_docs_,
            "class_priors_": self.class_priors_,
            "log_class_priors_": self.log_class_priors_,
            "feature_counts_": self.feature_counts_,
            "class_word_totals_": self.class_word_totals_,
            "vocab_size_": self.vocab_size_,
            "feature_log_prob_": self.feature_log_prob_,
            "vocab_": self.vocab_,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> "MultinomialNaiveBayes":
        """Deserialize and restore a trained model from disk.

        Args:
            filepath: Path to the saved model pickle file.

        Returns:
            Restored MultinomialNaiveBayes instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        model = cls(alpha=state.get("alpha", 1.0))
        model.classes_ = state.get("classes_")
        model.class_to_idx_ = state.get("class_to_idx_")
        model.class_counts_ = state.get("class_counts_")
        model.total_docs_ = state.get("total_docs_", 0)
        model.class_priors_ = state.get("class_priors_")
        model.log_class_priors_ = state.get("log_class_priors_")
        model.feature_counts_ = state.get("feature_counts_")
        model.class_word_totals_ = state.get("class_word_totals_")
        model.vocab_size_ = state.get("vocab_size_", 0)
        model.feature_log_prob_ = state.get("feature_log_prob_")
        model.vocab_ = state.get("vocab_")
        if model.vocab_ is not None:
            model.vectorizer_ = BagOfWords(vocab=model.vocab_)
        return model


def train_and_evaluate_bbc(
    data_path: str = "data/bbc_news.csv",
    model_output_path: str = "models/naive_bayes_weights.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[MultinomialNaiveBayes, Dict[str, float]]:
    """Train Multinomial Naive Bayes on the preprocessed BBC News training split.

    Executes:
    1. Loads dataset and validates schema and classes.
    2. Performs stratified train/test split (80/20).
    3. Preprocesses text using TextPreprocessor.
    4. Fits Bag-of-Words and MultinomialNaiveBayes with Add-1 Laplace smoothing.
    5. Evaluates training and test classification accuracies.
    6. Saves trained model to models/naive_bayes_weights.pkl.

    Args:
        data_path: Path to bbc_news.csv.
        model_output_path: Path to save trained weights pickle.
        test_size: Proportion of dataset for test partition (default: 0.2).
        random_state: Random seed for stratified splitting (default: 42).

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    print(f"[Naive Bayes] Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    train_df, test_df = stratified_train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"[Naive Bayes] Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

    preprocessor = TextPreprocessor()
    print("[Naive Bayes] Preprocessing training documents...")
    train_tokens = [
        preprocessor.preprocess(text, return_tokens=True)
        for text in train_df["text"]
    ]
    print("[Naive Bayes] Preprocessing test documents...")
    test_tokens = [
        preprocessor.preprocess(text, return_tokens=True)
        for text in test_df["text"]
    ]

    # Fit Bag-of-Words and Multinomial Naive Bayes
    print("[Naive Bayes] Fitting Bag-of-Words and Multinomial Naive Bayes...")
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(train_tokens, train_df["category"].values)

    train_acc = model.score(train_tokens, train_df["category"].values)
    test_acc = model.score(test_tokens, test_df["category"].values)

    print(f"[Naive Bayes] Vocabulary size: {model.vocab_size_}")
    print(f"[Naive Bayes] Training Accuracy: {train_acc * 100:.2f}%")
    print(f"[Naive Bayes] Test Accuracy:     {test_acc * 100:.2f}%")

    print(f"[Naive Bayes] Saving model weights to: {model_output_path}")
    model.save(model_output_path)
    print("[Naive Bayes] Model weights successfully saved.")

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "vocab_size": float(model.vocab_size_),
        "num_classes": float(len(model.classes_)),
    }
    return model, metrics


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, "data", "bbc_news.csv")
    model_file = os.path.join(project_root, "models", "naive_bayes_weights.pkl")

    trained_model, eval_metrics = train_and_evaluate_bbc(
        data_path=data_file,
        model_output_path=model_file,
    )
    print("\n--- Training Complete ---")
    print(f"Classes: {list(trained_model.classes_)}")
    print(f"Class Priors: {dict(zip(trained_model.classes_, np.round(trained_model.class_priors_, 4)))}")
    print(f"Test Accuracy: {eval_metrics['test_accuracy'] * 100:.2f}%")
