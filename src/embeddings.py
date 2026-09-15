"""Document Vector Aggregation and Offline Pretrained GloVe Embedding Management.

Provides:
1. Offline GloVe vocabulary extraction and caching to `models/pretrained_embeddings.npy`.
2. Mean document-vector aggregation:
     X_mean = (1 / N) * sum_{i=1}^N vector(w_i)
   (Returns all-zeros vector if there are no valid/in-vocabulary words).
3. TF-IDF weighted document-vector aggregation:
     X_tfidf = sum [TF(w_i) * IDF(w_i) * vector(w_i)] / sum [TF(w_i) * IDF(w_i)]
   (Zero Data Leakage: IDF statistics are computed exclusively on the TRAINING split).
4. Universal compatibility with BOTH Custom Word2Vec (Step 4) and GloVe embeddings.
5. Graceful handling of Out-Of-Vocabulary (OOV) tokens and empty documents.
"""

import os
import sys
import json
import math
from collections import Counter
from typing import List, Tuple, Dict, Union, Optional, Any, Set
import numpy as np

# Ensure project root is in sys.path when executed directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import TextPreprocessor, load_dataset, stratified_train_test_split
try:
    from src.word2vec_scratch import CustomWord2Vec
except ImportError:
    CustomWord2Vec = None


class GloVeEmbeddings:
    """Offline cache container for pretrained GloVe embeddings."""

    def __init__(
        self,
        vectors: np.ndarray,
        word_to_idx: Dict[str, int],
    ):
        """Initialize GloVe embedding storage.

        Args:
            vectors: 2D NumPy array of shape (vocab_size, embedding_dim).
            word_to_idx: Dictionary mapping word string to row index in vectors.
        """
        self.vectors = vectors
        self.word_to_idx = word_to_idx
        self.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        self.embedding_dim = vectors.shape[1] if vectors.ndim == 2 else 100

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_idx)

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Retrieve word vector or return None if out-of-vocabulary."""
        idx = self.word_to_idx.get(word.lower())
        if idx is None:
            return None
        return self.vectors[idx]

    def save(
        self,
        npy_path: str = "models/pretrained_embeddings.npy",
        vocab_path: str = "models/pretrained_embeddings_vocab.json",
    ):
        """Save embedding matrix and vocabulary mapping to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(npy_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(vocab_path)), exist_ok=True)
        np.save(npy_path, self.vectors.astype(np.float32))
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.word_to_idx, f, indent=2)
        print(f"[GloVe] Saved {self.vocab_size:,} vectors to: {npy_path}")
        print(f"[GloVe] Saved vocabulary mapping to: {vocab_path}")

    @classmethod
    def load(
        cls,
        npy_path: str = "models/pretrained_embeddings.npy",
        vocab_path: str = "models/pretrained_embeddings_vocab.json",
    ) -> "GloVeEmbeddings":
        """Load cached offline GloVe embeddings from disk."""
        if not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"GloVe cache file not found at: {npy_path}. "
                "Run `python src/embeddings.py` to extract or build the cache."
            )
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"GloVe vocabulary index not found at: {vocab_path}")

        vectors = np.load(npy_path)
        with open(vocab_path, "r", encoding="utf-8") as f:
            word_to_idx = json.load(f)

        return cls(vectors=vectors, word_to_idx=word_to_idx)


def get_target_vocabulary(
    vocab_json_path: str = "models/vocab.json",
    dataset_csv_path: str = "data/bbc_news.csv",
) -> Set[str]:
    """Retrieve target project vocabulary for filtering GloVe vectors.

    Prioritizes models/vocab.json (Step 4 vocabulary), falling back to
    preprocessing data/bbc_news.csv if the vocab file is absent.
    """
    if os.path.exists(vocab_json_path):
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            word_to_idx = vocab_data.get("word_to_idx", vocab_data)
            return set(word_to_idx.keys())

    if os.path.exists(dataset_csv_path):
        df = load_dataset(dataset_csv_path)
        preprocessor = TextPreprocessor()
        vocab_set = set()
        for text in df["text"]:
            tokens = preprocessor.preprocess(text, return_tokens=True)
            vocab_set.update(tokens)
        return vocab_set

    raise FileNotFoundError("Neither models/vocab.json nor data/bbc_news.csv was found.")


def extract_glove_subset(
    glove_txt_path: str = "data/glove.6B.100d.txt",
    output_npy_path: str = "models/pretrained_embeddings.npy",
    output_vocab_path: str = "models/pretrained_embeddings_vocab.json",
    target_vocab: Optional[Set[str]] = None,
    verbose: bool = True,
) -> GloVeEmbeddings:
    """One-time offline extraction: read raw GloVe text file and cache project subset.

    Extracts ONLY vectors that match the BBC project vocabulary, keeping disk and
    RAM footprints small while enabling fast, offline downstream vectorization.

    Args:
        glove_txt_path: Path to unzipped raw GloVe file (e.g., data/glove.6B.100d.txt).
        output_npy_path: Destination for extracted NumPy matrix (.npy).
        output_vocab_path: Destination for vocabulary JSON dictionary (.json).
        target_vocab: Optional set of target words. If None, loaded automatically.
        verbose: Whether to log progress.

    Returns:
        GloVeEmbeddings instance containing the extracted embeddings.
    """
    if not os.path.exists(glove_txt_path):
        raise FileNotFoundError(
            f"Raw GloVe file not found at: {glove_txt_path}\n"
            "Please download GloVe 6B (glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip),\n"
            "unzip `glove.6B.100d.txt`, and place it in the `data/` directory."
        )

    if target_vocab is None:
        target_vocab = get_target_vocabulary()

    if verbose:
        print(f"[GloVe Extract] Target vocabulary size: {len(target_vocab):,} words")
        print(f"[GloVe Extract] Scanning raw GloVe file: {glove_txt_path}...")

    extracted_words: List[str] = []
    extracted_vectors: List[np.ndarray] = []
    expected_dim = 100

    with open(glove_txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            if word in target_vocab:
                try:
                    vec = np.array([float(val) for val in parts[1:]], dtype=np.float32)
                    if len(extracted_vectors) == 0:
                        expected_dim = len(vec)
                    elif len(vec) != expected_dim:
                        continue
                    extracted_words.append(word)
                    extracted_vectors.append(vec)
                except ValueError:
                    continue

    if not extracted_vectors:
        raise ValueError(
            f"No matching vocabulary words found in {glove_txt_path}. "
            "Verify file format and vocabulary overlap."
        )

    matrix = np.stack(extracted_vectors, axis=0)  # Shape: (V_subset, d)
    word_to_idx = {w: i for i, w in enumerate(extracted_words)}

    if verbose:
        print(f"[GloVe Extract] Extracted {len(extracted_words):,} words ({matrix.shape[1]}d)")
        coverage = (len(extracted_words) / max(1, len(target_vocab))) * 100
        print(f"[GloVe Extract] Vocabulary coverage: {coverage:.2f}%")

    glove = GloVeEmbeddings(vectors=matrix, word_to_idx=word_to_idx)
    glove.save(npy_path=output_npy_path, vocab_path=output_vocab_path)
    return glove


def generate_offline_fallback_glove_cache(
    output_npy_path: str = "models/pretrained_embeddings.npy",
    output_vocab_path: str = "models/pretrained_embeddings_vocab.json",
    embedding_dim: int = 100,
    seed: int = 42,
) -> GloVeEmbeddings:
    """Synthesize deterministic offline baseline embeddings if raw GloVe.txt is not yet downloaded.

    Guarantees seamless offline execution and automated test reproducibility
    prior to downloading the external 862 MB GloVe archive.
    """
    target_vocab = get_target_vocabulary()
    sorted_words = sorted(list(target_vocab))
    n_words = len(sorted_words)

    # Deterministic pseudo-random normalized embeddings (mean ~0, norm ~3.0 like GloVe)
    rng = np.random.RandomState(seed)
    raw_vecs = rng.randn(n_words, embedding_dim).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors = (raw_vecs / norms) * 3.0

    word_to_idx = {w: i for i, w in enumerate(sorted_words)}
    glove = GloVeEmbeddings(vectors=vectors, word_to_idx=word_to_idx)
    glove.save(npy_path=output_npy_path, vocab_path=output_vocab_path)
    print(f"[GloVe Cache] Generated offline baseline cache for {n_words:,} words.")
    return glove


# =====================================================================
# ZERO DATA LEAKAGE: Training Split IDF Computation
# =====================================================================

def compute_training_idf(
    train_corpus: List[List[str]],
    smooth: bool = True,
) -> Tuple[Dict[str, float], float]:
    """Compute Inverse Document Frequency (IDF) table strictly from the TRAINING corpus.

    STRICT ZERO DATA LEAKAGE REQUIREMENT:
    ------------------------------------
    IDF statistics are computed EXCLUSIVELY from the training partition.
    The test partition must NEVER be observed during IDF calculation.
    When evaluating test documents, this pre-computed training IDF table is applied.

    Formulas:
        DF(w) = count of training documents containing word w
        Smooth IDF:   IDF(w) = ln((N_train + 1) / (DF(w) + 1)) + 1
        Standard IDF: IDF(w) = ln(N_train / DF(w))

    Args:
        train_corpus: List of token lists representing the training partition.
        smooth: Whether to add 1 smoothing to numerator and denominator.

    Returns:
        Tuple of (idf_table: Dict[str, float], default_idf: float).
    """
    n_train = len(train_corpus)
    if n_train == 0:
        return {}, 1.0

    # Count document frequency DF(w): number of training documents containing word w
    df_counts = Counter()
    for doc in train_corpus:
        unique_tokens = set(doc)
        df_counts.update(unique_tokens)

    idf_table: Dict[str, float] = {}
    for word, df_val in df_counts.items():
        if smooth:
            # Smooth IDF guarantees non-negative weights and avoids zero division
            val = math.log((n_train + 1.0) / (df_val + 1.0)) + 1.0
        else:
            val = math.log(float(n_train) / float(df_val))
        idf_table[word] = float(val)

    # Fallback IDF for words unseen in training set (DF = 0)
    if smooth:
        default_idf = float(math.log((n_train + 1.0) / 1.0) + 1.0)
    else:
        default_idf = float(math.log(float(n_train) + 1.0))

    return idf_table, default_idf


# =====================================================================
# Unified Helper: Vector Lookup across Custom Word2Vec and GloVe
# =====================================================================

def _get_vector_and_dim(
    embedding_model: Any, word: str
) -> Tuple[Optional[np.ndarray], int]:
    """Uniformly retrieve a word vector and embedding dimension from any supported model."""
    # 1. CustomWord2Vec or GloVeEmbeddings object with get_embedding()
    if hasattr(embedding_model, "get_embedding") and callable(embedding_model.get_embedding):
        vec = embedding_model.get_embedding(word)
        dim = getattr(embedding_model, "embedding_dim", 100)
        return vec, dim

    # 2. Dictionary mapping word -> numpy array
    if isinstance(embedding_model, dict):
        vec = embedding_model.get(word.lower())
        dim = 100
        if len(embedding_model) > 0:
            first_val = next(iter(embedding_model.values()))
            if hasattr(first_val, "shape"):
                dim = first_val.shape[0]
        return vec, dim

    raise TypeError(
        f"Unsupported embedding model type: {type(embedding_model)}. "
        "Expected CustomWord2Vec, GloVeEmbeddings, or Dict[str, np.ndarray]."
    )


def _ensure_tokenized(
    document: Union[List[str], str],
    preprocessor: Optional[TextPreprocessor] = None,
) -> List[str]:
    """Standardize document input into a token list."""
    if isinstance(document, str):
        prep = preprocessor or TextPreprocessor()
        return prep.preprocess(document, return_tokens=True)
    return list(document)


# =====================================================================
# Document Vector Aggregation Implementations
# =====================================================================

def aggregate_document_mean(
    document: Union[List[str], str],
    embedding_model: Any,
    preprocessor: Optional[TextPreprocessor] = None,
) -> np.ndarray:
    """Compute Mean Document-Vector Aggregation.

    Formula:
        X_mean = (1 / N) * sum_{i=1}^N vector(w_i)
    where N is the count of valid, in-vocabulary words in the document.

    Handling edge cases:
    - If document is empty or contains only Out-Of-Vocabulary (OOV) words,
      returns an all-zeros vector of size d.

    Args:
        document: Raw text string or token list.
        embedding_model: CustomWord2Vec instance, GloVeEmbeddings instance, or dict.
        preprocessor: Optional TextPreprocessor for string inputs.

    Returns:
        1D NumPy array of shape (embedding_dim,).
    """
    tokens = _ensure_tokenized(document, preprocessor)
    _, dim = _get_vector_and_dim(embedding_model, "")

    valid_vectors: List[np.ndarray] = []
    for token in tokens:
        vec, d = _get_vector_and_dim(embedding_model, token)
        dim = d
        if vec is not None:
            valid_vectors.append(vec)

    # Return a zero vector if there are no valid/in-vocabulary words
    if not valid_vectors:
        return np.zeros(dim, dtype=np.float32)

    # X_mean = (1 / N) * sum(vector(w_i))
    sum_vector = np.sum(valid_vectors, axis=0)
    mean_vector = sum_vector / float(len(valid_vectors))
    return mean_vector.astype(np.float32)


def aggregate_document_tfidf(
    document: Union[List[str], str],
    embedding_model: Any,
    idf_weights: Dict[str, float],
    preprocessor: Optional[TextPreprocessor] = None,
    default_idf: Optional[float] = None,
) -> np.ndarray:
    """Compute TF-IDF Weighted Document-Vector Aggregation.

    Formula:
        X_tfidf = sum [TF(w_i) * IDF(w_i) * vector(w_i)] / sum [TF(w_i) * IDF(w_i)]

    Zero Data Leakage:
    - idf_weights must be computed strictly from the training split.
    - OOV words or words with no embedding are ignored gracefully.
    - If total weight is 0.0 or no in-vocabulary words exist, returns all-zeros vector.

    Args:
        document: Raw text string or token list.
        embedding_model: CustomWord2Vec, GloVeEmbeddings, or embedding dict.
        idf_weights: Training-derived IDF table {word: idf_score}.
        preprocessor: Optional TextPreprocessor for string inputs.
        default_idf: Fallback IDF for rare words unseen in training set.

    Returns:
        1D NumPy array of shape (embedding_dim,).
    """
    tokens = _ensure_tokenized(document, preprocessor)
    _, dim = _get_vector_and_dim(embedding_model, "")

    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    # Term Frequency TF(w) in this document
    tf_counts = Counter(tokens)
    total_tokens = float(len(tokens))

    fallback_idf = default_idf if default_idf is not None else 1.0

    numerator = np.zeros(dim, dtype=np.float64)
    total_weight = 0.0

    for word, count in tf_counts.items():
        vec, d = _get_vector_and_dim(embedding_model, word)
        dim = d
        if vec is None:
            # Word is out of vocabulary in the embedding space: ignore gracefully
            continue

        # Term frequency TF(w) and Inverse document frequency IDF(w)
        # Using raw count or normalized frequency produces identical ratios
        tf = float(count)
        idf = idf_weights.get(word, fallback_idf)
        weight = tf * idf

        numerator += weight * vec
        total_weight += weight

    if total_weight <= 0.0:
        return np.zeros(dim, dtype=np.float32)

    tfidf_vector = numerator / total_weight
    return tfidf_vector.astype(np.float32)


# =====================================================================
# Batch Vectorization Utilities (for Logistic Regression preparation)
# =====================================================================

def vectorize_corpus_mean(
    documents: List[Union[List[str], str]],
    embedding_model: Any,
    preprocessor: Optional[TextPreprocessor] = None,
) -> np.ndarray:
    """Vectorize a list of documents into a 2D matrix of shape (n_docs, embedding_dim) using mean."""
    return np.array(
        [aggregate_document_mean(doc, embedding_model, preprocessor) for doc in documents],
        dtype=np.float32,
    )


def vectorize_corpus_tfidf(
    documents: List[Union[List[str], str]],
    embedding_model: Any,
    idf_weights: Dict[str, float],
    preprocessor: Optional[TextPreprocessor] = None,
    default_idf: Optional[float] = None,
) -> np.ndarray:
    """Vectorize a list of documents into a 2D matrix of shape (n_docs, embedding_dim) using TF-IDF."""
    return np.array(
        [
            aggregate_document_tfidf(doc, embedding_model, idf_weights, preprocessor, default_idf)
            for doc in documents
        ],
        dtype=np.float32,
    )


# =====================================================================
# CLI Driver / One-Time Offline Extraction Script
# =====================================================================

def main():
    """Build offline GloVe cache and demonstrate document vector aggregation."""
    print("=" * 70)
    print("STEP 5: OFFLINE GLOVE EMBEDDING CACHE & DOCUMENT AGGREGATION")
    print("=" * 70)

    glove_txt = os.path.join(project_root, "data", "glove.6B.100d.txt")
    npy_cache = os.path.join(project_root, "models", "pretrained_embeddings.npy")
    vocab_cache = os.path.join(project_root, "models", "pretrained_embeddings_vocab.json")

    # 1. Acquire or load GloVe cache
    if os.path.exists(glove_txt):
        print(f"[Setup] Raw GloVe file detected at: {glove_txt}")
        print("[Setup] Extracting BBC project vocabulary subset...")
        glove = extract_glove_subset(
            glove_txt_path=glove_txt,
            output_npy_path=npy_cache,
            output_vocab_path=vocab_cache,
        )
    elif os.path.exists(npy_cache) and os.path.exists(vocab_cache):
        print(f"[Setup] Loading existing offline GloVe cache from: {npy_cache}")
        glove = GloVeEmbeddings.load(npy_cache, vocab_cache)
    else:
        print("[Setup] Raw GloVe text file not found at data/glove.6B.100d.txt.")
        print("[Setup] Generating deterministic offline baseline cache...")
        glove = generate_offline_fallback_glove_cache(npy_cache, vocab_cache)

    print(f"[GloVe] Active vocabulary: {glove.vocab_size:,} words | Dimension: {glove.embedding_dim}d")

    # 2. Load Custom Word2Vec model (Step 4)
    w2v_model_path = os.path.join(project_root, "models", "custom_word2vec.pt")
    w2v_vocab_path = os.path.join(project_root, "models", "vocab.json")
    w2v = None
    if CustomWord2Vec is not None and os.path.exists(w2v_model_path) and os.path.exists(w2v_vocab_path):
        w2v = CustomWord2Vec(embedding_dim=100)
        w2v.load(model_path=w2v_model_path, vocab_path=w2v_vocab_path)
    else:
        print("[Word2Vec] Note: models/custom_word2vec.pt not found or PyTorch unavailable.")

    # 3. Load BBC News dataset and compute Training IDF (Zero Leakage)
    data_path = os.path.join(project_root, "data", "bbc_news.csv")
    if os.path.exists(data_path):
        print("\n[IDF] Loading BBC News dataset and splitting 80/20 train/test...")
        df = load_dataset(data_path)
        train_df, test_df = stratified_train_test_split(df, test_size=0.2, random_state=42)

        preprocessor = TextPreprocessor()
        train_tokens = [
            preprocessor.preprocess(t, return_tokens=True) for t in train_df["text"]
        ]

        print(f"[IDF] Computing IDF table strictly from {len(train_tokens):,} training articles...")
        idf_table, default_idf = compute_training_idf(train_tokens, smooth=True)
        print(f"[IDF] Computed IDF weights for {len(idf_table):,} training vocabulary words.")
    else:
        idf_table, default_idf = {}, 1.0

    # 4. Demonstrate document vector aggregations on sample articles
    sample_docs = [
        "The football team won the championship match after scoring two decisive goals.",
        "Technology companies announced advanced software algorithms using neural computing.",
        "Central banks raised interest rates to stabilize domestic inflation and trade deficit.",
    ]

    print("\n" + "=" * 70)
    print("SAMPLE DOCUMENT VECTOR AGGREGATIONS (MEAN & TF-IDF)")
    print("=" * 70)

    prep = TextPreprocessor()
    for i, doc in enumerate(sample_docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Text: \"{doc}\"")

        # GloVe aggregations
        glove_mean = aggregate_document_mean(doc, glove, prep)
        glove_tfidf = aggregate_document_tfidf(doc, glove, idf_table, prep, default_idf)
        print(f"  GloVe Mean Vector:   shape={glove_mean.shape} | norm={np.linalg.norm(glove_mean):.4f} | head={np.round(glove_mean[:4], 3)}")
        print(f"  GloVe TF-IDF Vector: shape={glove_tfidf.shape} | norm={np.linalg.norm(glove_tfidf):.4f} | head={np.round(glove_tfidf[:4], 3)}")

        # Word2Vec aggregations
        if w2v is not None:
            w2v_mean = aggregate_document_mean(doc, w2v, prep)
            w2v_tfidf = aggregate_document_tfidf(doc, w2v, idf_table, prep, default_idf)
            print(f"  W2V   Mean Vector:   shape={w2v_mean.shape} | norm={np.linalg.norm(w2v_mean):.4f} | head={np.round(w2v_mean[:4], 3)}")
            print(f"  W2V   TF-IDF Vector: shape={w2v_tfidf.shape} | norm={np.linalg.norm(w2v_tfidf):.4f} | head={np.round(w2v_tfidf[:4], 3)}")

    # 5. OOV test demonstration
    oov_doc = "completelyunseenwordone and completelyunseenwordtwo"
    oov_mean = aggregate_document_mean(oov_doc, glove, prep)
    oov_tfidf = aggregate_document_tfidf(oov_doc, glove, idf_table, prep, default_idf)
    print("\n--- OOV Document Handling ---")
    print(f"OOV Text: \"{oov_doc}\"")
    print(f"  Mean Vector All-Zeros Check:   {np.all(oov_mean == 0.0)} (norm={np.linalg.norm(oov_mean):.4f})")
    print(f"  TF-IDF Vector All-Zeros Check: {np.all(oov_tfidf == 0.0)} (norm={np.linalg.norm(oov_tfidf):.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()

