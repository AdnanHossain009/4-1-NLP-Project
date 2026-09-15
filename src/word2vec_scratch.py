"""Custom Word2Vec Skip-Gram Architecture and Training from Scratch.

Implements foundational neural word representations using PyTorch:
- Sliding context window training pair generation: (w_c, w_o)
- Neural architecture: nn.Embedding(V, d) and nn.Linear(d, V, bias=False)
- Skip-Gram conditional probability:
    P(w_o | w_c) = exp(v'_wo^T v_wc) / sum_j exp(v'_wj^T v_wc)
- Training via CrossEntropyLoss minimization with Adam optimizer and mini-batches
- Cosine similarity:
    cosine(a, b) = (a . b) / (||a|| ||b||)
- Nearest neighbor semantic search and model serialization.
"""

import os
import sys
import json
import math
from collections import Counter
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path when executed directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import TextPreprocessor, load_dataset, stratified_train_test_split


class SkipGramDataset(Dataset):
    """PyTorch Dataset for center-context word index pairs."""

    def __init__(self, pairs: List[Tuple[int, int]]):
        self.centers = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        self.contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.centers[idx], self.contexts[idx]


class SkipGramModel(nn.Module):
    """PyTorch Skip-Gram neural network architecture.

    Layers:
      - nn.Embedding(V, d): Center word lookup representation (V_wc).
      - nn.Linear(d, V, bias=False): Context projection weights (V'_wo).
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Input embedding matrix: shape (V, d)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Output projection matrix: shape (d, V) via nn.Linear(d, V, bias=False)
        self.output_layer = nn.Linear(embedding_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with uniform Xavier initialization."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, center_words: torch.Tensor) -> torch.Tensor:
        """Forward pass computing output logits for all vocabulary words.

        Args:
            center_words: Tensor of center word indices, shape (batch_size,).

        Returns:
            Logits tensor of shape (batch_size, vocab_size).
        """
        # embed: (batch_size, embedding_dim)
        embed = self.embedding(center_words)
        # logits: (batch_size, vocab_size)
        logits = self.output_layer(embed)
        return logits


class CustomWord2Vec:
    """Orchestrator for Skip-Gram vocabulary, training, similarity, and persistence."""

    def __init__(
        self,
        embedding_dim: int = 100,
        window_size: int = 2,
        min_count: int = 5,
        lr: float = 0.002,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.lr = lr
        self.seed = seed

        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab: List[str] = []
        self.model: Optional[SkipGramModel] = None
        self.embeddings: Optional[np.ndarray] = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def build_vocab(self, tokenized_corpus: List[List[str]]) -> Dict[str, int]:
        """Build vocabulary from tokenized corpus using min_count threshold.

        Args:
            tokenized_corpus: List of preprocessed token lists.

        Returns:
            word_to_idx dictionary.
        """
        counter = Counter()
        for doc in tokenized_corpus:
            counter.update(doc)

        # Filter by minimum frequency
        sorted_words = [
            word for word, count in counter.most_common()
            if count >= self.min_count
        ]

        # If min_count was too aggressive on tiny corpus, retain all words
        if not sorted_words and counter:
            sorted_words = [word for word, _ in counter.most_common()]

        self.word_to_idx = {word: idx for idx, word in enumerate(sorted_words)}
        self.idx_to_word = {idx: word for idx, word in enumerate(sorted_words)}
        self.vocab = sorted_words
        return self.word_to_idx

    def generate_training_pairs(
        self, tokenized_corpus: List[List[str]]
    ) -> List[Tuple[int, int]]:
        """Generate (center, context) index pairs within sliding window.

        Args:
            tokenized_corpus: List of preprocessed token lists.

        Returns:
            List of (center_idx, context_idx) pairs.
        """
        pairs = []
        w = self.window_size

        for doc in tokenized_corpus:
            # Map tokens to indices, ignoring out-of-vocabulary tokens
            indices = [self.word_to_idx[t] for t in doc if t in self.word_to_idx]
            doc_len = len(indices)

            for i in range(doc_len):
                center = indices[i]
                start = max(0, i - w)
                end = min(doc_len, i + w + 1)

                for j in range(start, end):
                    if i != j:
                        context = indices[j]
                        pairs.append((center, context))

        return pairs

    def train(
        self,
        tokenized_corpus: List[List[str]],
        epochs: int = 5,
        batch_size: int = 1024,
        device: str = "cpu",
        verbose: bool = True,
    ) -> List[float]:
        """Train the Skip-Gram model on the tokenized corpus.

        Args:
            tokenized_corpus: Preprocessed token lists.
            epochs: Number of complete passes over the pairs.
            batch_size: Mini-batch size.
            device: 'cpu' or 'cuda'.
            verbose: Whether to print progress logs.

        Returns:
            List of average loss values per epoch.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if not self.word_to_idx:
            self.build_vocab(tokenized_corpus)

        if self.vocab_size == 0:
            raise ValueError("Vocabulary is empty. Check input corpus or min_count.")

        pairs = self.generate_training_pairs(tokenized_corpus)
        if not pairs:
            raise ValueError("No training pairs generated. Check window size and corpus.")

        if verbose:
            print(f"[Word2Vec] Vocab size: {self.vocab_size:,} | Training pairs: {len(pairs):,}")
            print(f"[Word2Vec] Dimension: {self.embedding_dim} | Window: {self.window_size} | Epochs: {epochs}")

        dataset = SkipGramDataset(pairs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = SkipGramModel(
            vocab_size=self.vocab_size, embedding_dim=self.embedding_dim
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        epoch_losses = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            total_batches = 0

            for centers, contexts in dataloader:
                centers = centers.to(device)
                contexts = contexts.to(device)

                optimizer.zero_grad()
                logits = self.model(centers)
                loss = criterion(logits, contexts)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_batches += 1

            avg_loss = running_loss / max(1, total_batches)
            epoch_losses.append(avg_loss)
            if verbose:
                print(f"  Epoch {epoch:2d}/{epochs} - Loss: {avg_loss:.4f}")

        # Cache learned embedding weights as numpy array: shape (V, d)
        self.embeddings = self.model.embedding.weight.detach().cpu().numpy()
        return epoch_losses

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Return embedding vector for a given word, or None if OOV."""
        if self.embeddings is None or word not in self.word_to_idx:
            return None
        idx = self.word_to_idx[word]
        return self.embeddings[idx]

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity: (a . b) / (||a|| ||b||)."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def find_nearest_neighbors(
        self, word: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find the top-k most semantically similar words by cosine similarity.

        Args:
            word: Query word string.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (neighbor_word, cosine_similarity_score) sorted descending.
            Returns empty list if word is out-of-vocabulary.
        """
        if self.embeddings is None or word not in self.word_to_idx:
            return []

        query_idx = self.word_to_idx[word]
        query_vec = self.embeddings[query_idx]  # shape (d,)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0.0:
            return []

        # Vectorized cosine similarity over all vocabulary vectors
        # matrix: (V, d), query: (d,) -> dot products: (V,)
        dot_products = np.dot(self.embeddings, query_vec)
        norms = np.linalg.norm(self.embeddings, axis=1) * query_norm
        # Handle zero-norm guards
        norms[norms == 0.0] = 1e-12

        similarities = dot_products / norms

        # Exclude query word itself from candidate recommendations
        similarities[query_idx] = -float("inf")

        # Extract top-k highest similarity indices
        top_indices = np.argsort(-similarities)[:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score == -float("inf"):
                continue
            results.append((self.idx_to_word[idx], score))

        return results

    def save(
        self,
        model_path: str = "models/custom_word2vec.pt",
        vocab_path: str = "models/vocab.json",
    ):
        """Serialize trained PyTorch model and vocabulary to local disk."""
        if self.model is None or not self.word_to_idx:
            raise ValueError("Cannot save an untrained model.")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

        # 1. Save PyTorch model state and hyper-parameters
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "window_size": self.window_size,
            "min_count": self.min_count,
        }
        torch.save(checkpoint, model_path)

        # 2. Save vocabulary JSON
        vocab_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": {str(k): v for k, v in self.idx_to_word.items()},
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

        print(f"[Word2Vec] Successfully saved model to: {model_path}")
        print(f"[Word2Vec] Successfully saved vocabulary to: {vocab_path}")

    def load(
        self,
        model_path: str = "models/custom_word2vec.pt",
        vocab_path: str = "models/vocab.json",
        device: str = "cpu",
    ):
        """Load trained weights and vocabulary mapping from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found at: {vocab_path}")

        # 1. Load vocabulary JSON
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.word_to_idx = vocab_data["word_to_idx"]
        self.idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}
        self.vocab = [self.idx_to_word[i] for i in range(len(self.idx_to_word))]
        self.embedding_dim = vocab_data["embedding_dim"]

        # 2. Load PyTorch model
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self.model = SkipGramModel(
            vocab_size=len(self.vocab), embedding_dim=self.embedding_dim
        ).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.embeddings = self.model.embedding.weight.detach().cpu().numpy()
        print(f"[Word2Vec] Loaded model ({self.vocab_size:,} words, {self.embedding_dim}d) from: {model_path}")


def train_and_save_bbc_word2vec(
    csv_path: str = "data/bbc_news.csv",
    model_path: str = "models/custom_word2vec.pt",
    vocab_path: str = "models/vocab.json",
    embedding_dim: int = 100,
    window_size: int = 2,
    min_count: int = 5,
    epochs: int = 6,
    batch_size: int = 1024,
) -> CustomWord2Vec:
    """High-level training pipeline executed on BBC News training split."""
    print("=" * 60)
    print("TRAINING CUSTOM WORD2VEC SKIP-GRAM (BBC NEWS CORPUS)")
    print("=" * 60)

    df = load_dataset(csv_path)
    train_df, _ = stratified_train_test_split(df, test_size=0.2, random_state=42)

    preprocessor = TextPreprocessor()
    print(f"Preprocessing {len(train_df):,} training articles...")
    tokenized_corpus = [
        preprocessor.preprocess(text, return_tokens=True)
        for text in train_df["text"]
    ]

    w2v = CustomWord2Vec(
        embedding_dim=embedding_dim,
        window_size=window_size,
        min_count=min_count,
        lr=0.003,
        seed=42,
    )

    w2v.train(
        tokenized_corpus=tokenized_corpus,
        epochs=epochs,
        batch_size=batch_size,
        device="cpu",
        verbose=True,
    )

    w2v.save(model_path=model_path, vocab_path=vocab_path)

    print("\n--- Semantic Nearest Neighbors (Top 5) ---")
    demo_words = ["football", "market", "technology", "government", "music"]
    for word in demo_words:
        neighbors = w2v.find_nearest_neighbors(word, top_k=5)
        neighbor_str = ", ".join([f"{w} ({s:.3f})" for w, s in neighbors])
        print(f"  {word:12s} -> {neighbor_str}")

    return w2v


if __name__ == "__main__":
    train_and_save_bbc_word2vec()

