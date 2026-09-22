"""TF-IDF and N-Gram Feature Extraction Engine from Scratch using Pure NumPy.

Implements transparent, formula-first TF-IDF vectorization:
- Configurable N-Gram extraction: Unigram (1-gram) or Unigram + Bigram (1-gram + 2-gram).
- Strict Zero-Data-Leakage fit/transform architecture:
    Vocabulary and IDF weights are computed strictly on the training partition.
    The test partition is only transformed using training-derived statistics.
- Exact Mathematical Formulas:
    TF(t, d) = count(t, d)  [or 1 + ln(count(t, d)) if sublinear_tf=True]
    DF(t) = sum_{d in D_train} 1(t in d)
    IDF(t) = ln((N_train + 1) / (DF(t) + 1)) + 1
    TF-IDF(t, d) = TF(t, d) * IDF(t)
    L2 Normalization: x_d = v_d / ||v_d||_2
- Top feature extraction and instance-level explainability for linear models.
- Serialization and deserialization to pickle files.
"""

import os
import math
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np


class TfidfNGramFeatureExtractor:
    """Production-grade TF-IDF Vectorizer with N-Gram support and zero data leakage."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: Optional[int] = 5000,
        sublinear_tf: bool = False,
        norm: str = "l2",
    ):
        """Initialize the feature extractor.

        Args:
            ngram_range: (min_n, max_n) tuple, e.g. (1, 1) for unigrams, (1, 2) for unigrams + bigrams.
            min_df: Minimum document frequency for a term to be retained in vocabulary.
            max_features: Maximum number of features in vocabulary (keeps top terms by DF).
            sublinear_tf: If True, uses sublinear scaling 1 + ln(tf) for term frequency.
            norm: Normalization method ('l2' or None).
        """
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError(f"Invalid ngram_range: {ngram_range}")

        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.norm = norm

        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self.idf_diag_: Optional[np.ndarray] = None
        self.df_: Dict[str, int] = {}
        self.n_samples_trained_: int = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary_)

    def extract_ngrams(self, tokens: List[str]) -> List[str]:
        """Extract unigrams and/or bigrams from a list of tokens according to ngram_range.

        Unigrams: "word"
        Bigrams: "word1_word2"
        """
        min_n, max_n = self.ngram_range
        extracted: List[str] = []
        n_tokens = len(tokens)

        for n in range(min_n, max_n + 1):
            if n == 1:
                extracted.extend(tokens)
            else:
                for i in range(n_tokens - n + 1):
                    ngram = "_".join(tokens[i : i + n])
                    extracted.append(ngram)

        return extracted

    def fit(self, corpus_tokens: List[List[str]]) -> "TfidfNGramFeatureExtractor":
        """Learn vocabulary and IDF weights strictly from the training corpus.

        Zero Data Leakage:
            N_train = len(corpus_tokens)
            DF(t) = count of training documents containing term t
            IDF(t) = ln((N_train + 1) / (DF(t) + 1)) + 1

        Args:
            corpus_tokens: List of token lists, one per document.

        Returns:
            self: Fitted extractor.
        """
        self.n_samples_trained_ = len(corpus_tokens)
        if self.n_samples_trained_ == 0:
            raise ValueError("Cannot fit TF-IDF extractor on an empty corpus.")

        # 1. Compute Document Frequencies across training corpus
        df_counts: Dict[str, int] = {}
        for doc_tokens in corpus_tokens:
            ngrams = self.extract_ngrams(doc_tokens)
            unique_in_doc = set(ngrams)
            for term in unique_in_doc:
                df_counts[term] = df_counts.get(term, 0) + 1

        # 2. Filter terms by min_df
        filtered_terms = [
            term for term, df in df_counts.items() if df >= self.min_df
        ]

        # 3. Sort terms: highest DF first, then alphabetical for determinism
        filtered_terms.sort(key=lambda t: (-df_counts[t], t))

        # 4. Cap at max_features if requested
        if self.max_features is not None and len(filtered_terms) > self.max_features:
            filtered_terms = filtered_terms[: self.max_features]

        # Alphabetical sorting for clean indexing
        filtered_terms.sort()

        # 5. Build vocabulary mapping
        self.feature_names_ = filtered_terms
        self.vocabulary_ = {term: idx for idx, term in enumerate(filtered_terms)}
        self.df_ = {term: df_counts[term] for term in filtered_terms}

        # 6. Compute smoothed IDF vector strictly on training statistics:
        #    IDF(t) = ln((N + 1) / (DF(t) + 1)) + 1
        n = float(self.n_samples_trained_)
        idf_values = np.zeros(len(self.feature_names_), dtype=np.float32)
        for idx, term in enumerate(self.feature_names_):
            df = float(self.df_[term])
            idf_values[idx] = math.log((n + 1.0) / (df + 1.0)) + 1.0

        self.idf_diag_ = idf_values
        return self

    def transform(self, corpus_tokens: List[List[str]]) -> np.ndarray:
        """Transform documents into normalized TF-IDF feature vectors.

        Formulas:
            TF(t, d) = count(t, d)  [or 1 + ln(count(t, d))]
            TFIDF(t, d) = TF(t, d) * IDF(t)
            L2: x_d = v_d / ||v_d||_2

        Args:
            corpus_tokens: List of token lists, one per document.

        Returns:
            2D NumPy array of shape (m, vocab_size).
        """
        if not self.vocabulary_ or self.idf_diag_ is None:
            raise ValueError("TfidfNGramFeatureExtractor must be fitted before transform.")

        m = len(corpus_tokens)
        d = len(self.feature_names_)
        X = np.zeros((m, d), dtype=np.float32)

        for i, doc_tokens in enumerate(corpus_tokens):
            ngrams = self.extract_ngrams(doc_tokens)
            if not ngrams:
                continue

            # Count term frequencies in this document
            tf_counts: Dict[str, int] = {}
            for term in ngrams:
                if term in self.vocabulary_:
                    tf_counts[term] = tf_counts.get(term, 0) + 1

            # Populate TF vector
            for term, count in tf_counts.items():
                idx = self.vocabulary_[term]
                if self.sublinear_tf:
                    X[i, idx] = 1.0 + math.log(count)
                else:
                    X[i, idx] = float(count)

        # Multiply by IDF weights: TF-IDF = TF * IDF
        X *= self.idf_diag_

        # Apply L2 Normalization if requested
        if self.norm == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            # Avoid division by zero for empty or all-OOV documents
            norms[norms == 0.0] = 1.0
            X /= norms

        return X

    def fit_transform(self, corpus_tokens: List[List[str]]) -> np.ndarray:
        """Fit extractor to corpus and transform it in a single pass."""
        return self.fit(corpus_tokens).transform(corpus_tokens)

    def explain_instance(
        self,
        tokens: List[str],
        weights_by_class: Dict[str, np.ndarray],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Explain model prediction on a single document by inspecting active features.

        Args:
            tokens: Preprocessed tokens for the single document.
            weights_by_class: Mapping from category name to 1D weight array of shape (d,).
            top_k: Number of top features to report per category.

        Returns:
            Dictionary containing active features found and top positive contributors per category.
        """
        ngrams = self.extract_ngrams(tokens)
        active_terms = [t for t in ngrams if t in self.vocabulary_]
        unique_active = sorted(list(set(active_terms)))

        # Compute document TF-IDF vector
        doc_vec = self.transform([tokens])[0]  # shape (d,)

        class_explanations: Dict[str, List[Dict[str, Any]]] = {}
        for category, w in weights_by_class.items():
            # Linear contribution: x_j * w_j
            contributions = doc_vec * w
            # Rank active features by positive contribution
            active_indices = [self.vocabulary_[t] for t in unique_active]
            ranked_indices = sorted(
                active_indices, key=lambda idx: contributions[idx], reverse=True
            )

            top_features = []
            for idx in ranked_indices[:top_k]:
                term = self.feature_names_[idx]
                top_features.append({
                    "term": term,
                    "tfidf": float(doc_vec[idx]),
                    "weight": float(w[idx]),
                    "contribution": float(contributions[idx]),
                })
            class_explanations[category] = top_features

        return {
            "total_tokens": len(tokens),
            "total_ngrams": len(ngrams),
            "active_vocab_terms": unique_active,
            "class_explanations": class_explanations,
        }

    def save(self, filepath: str = "models/tfidf_extractor.pkl"):
        """Serialize extractor state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = {
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "norm": self.norm,
            "vocabulary_": self.vocabulary_,
            "feature_names_": self.feature_names_,
            "idf_diag_": self.idf_diag_,
            "df_": self.df_,
            "n_samples_trained_": self.n_samples_trained_,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"[TF-IDF Extractor] Serialized model to: {filepath} ({len(self.vocabulary_):,} features)")

    @classmethod
    def load(cls, filepath: str = "models/tfidf_extractor.pkl") -> "TfidfNGramFeatureExtractor":
        """Load serialized extractor state from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"TF-IDF model file not found at: {filepath}")

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        extractor = cls(
            ngram_range=state["ngram_range"],
            min_df=state["min_df"],
            max_features=state.get("max_features", None),
            sublinear_tf=state.get("sublinear_tf", False),
            norm=state.get("norm", "l2"),
        )
        extractor.vocabulary_ = state["vocabulary_"]
        extractor.feature_names_ = state["feature_names_"]
        extractor.idf_diag_ = state["idf_diag_"]
        extractor.df_ = state["df_"]
        extractor.n_samples_trained_ = state["n_samples_trained_"]
        return extractor
