"""Generalized N-Gram Language Model Module.

Implements Maximum Likelihood Estimation (MLE), Laplace (Add-1) smoothing,
sequence log-likelihood, perplexity computation, and top-k next-word prediction
for arbitrary n-gram orders (bigram, trigram, 4-gram, 5-gram).
"""

import math
from collections import defaultdict
from typing import List, Tuple, Union, Optional, Dict


class NGramLanguageModel:
    """Generalized N-Gram Language Model supporting arbitrary n >= 1."""

    def __init__(self, n: int = 2, laplace_smoothing: bool = True):
        """Initialize the generalized N-Gram Language Model.

        Args:
            n: Order of the n-gram model (e.g., 2=Bigram, 3=Trigram, 4=4-gram, 5=5-gram).
            laplace_smoothing: Whether to apply Add-1 Laplace smoothing by default.
        """
        if n < 1:
            raise ValueError(f"N-Gram order n must be >= 1, got {n}")
        self.n = n
        self.laplace_smoothing = laplace_smoothing

        # Storage for counts
        # Key for ngram_counts: ((w_{i-n+1}, ..., w_{i-1}), w_i)
        self.ngram_counts: Dict[Tuple[Tuple[str, ...], str], int] = defaultdict(int)
        # Key for context_counts: (w_{i-n+1}, ..., w_{i-1})
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)

        # Vocabulary and tokens
        self.vocab = set()
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.total_tokens = 0

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary |V| including boundary tokens."""
        return len(self.vocab)

    def _pad_tokens(self, tokens: List[str]) -> List[str]:
        """Prepend (n - 1) start tokens and append 1 end token."""
        start_padding = [self.start_token] * (self.n - 1)
        end_padding = [self.end_token]
        return start_padding + list(tokens) + end_padding

    def _normalize_context(self, context: Union[Tuple[str, ...], List[str], str]) -> Tuple[str, ...]:
        """Ensure context is a tuple of length (n - 1)."""
        if isinstance(context, str):
            ctx_tokens = context.strip().split()
        else:
            ctx_tokens = list(context)

        needed = self.n - 1
        if len(ctx_tokens) < needed:
            # Prepend start tokens if shorter than context window
            ctx_tokens = [self.start_token] * (needed - len(ctx_tokens)) + ctx_tokens
        elif len(ctx_tokens) > needed:
            # Take the most recent (n - 1) tokens
            ctx_tokens = ctx_tokens[-needed:]

        return tuple(ctx_tokens)

    def extract_ngrams(self, tokens: List[str], n: Optional[int] = None) -> List[Tuple[str, ...]]:
        """Extract all contiguous n-grams from a token sequence.

        Args:
            tokens: List of string tokens.
            n: Order of n-grams to extract (defaults to self.n).

        Returns:
            List of n-gram tuples.
        """
        order = self.n if n is None else n
        if len(tokens) < order:
            return []
        return [tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1)]

    def train(self, corpus: Union[List[List[str]], List[str]]):
        """Train the N-Gram language model on a tokenized corpus.

        Args:
            corpus: Either a list of sentences (each being a list of tokens),
                    or a flat list of tokens.
        """
        if not corpus:
            return

        # Determine if corpus is list of sentences or flat tokens
        if isinstance(corpus[0], str):
            sentences = [corpus]
        else:
            sentences = corpus

        for sentence in sentences:
            if not sentence:
                continue

            padded = self._pad_tokens(sentence)

            # Register tokens in vocabulary
            for token in padded:
                self.vocab.add(token)

            # Count n-grams and contexts
            needed_context = self.n - 1
            for i in range(needed_context, len(padded)):
                context = tuple(padded[i - needed_context : i])
                word = padded[i]

                self.ngram_counts[(context, word)] += 1
                self.context_counts[context] += 1
                self.total_tokens += 1

    def get_probability(
        self,
        word: str,
        context: Union[Tuple[str, ...], List[str], str],
        smoothing: Optional[bool] = None,
    ) -> float:
        """Calculate conditional probability P(word | context).

        Formula (Laplace):
            P(w | context) = (C(context, w) + 1) / (C(context) + |V|)

        Formula (MLE):
            P(w | context) = C(context, w) / C(context)

        Args:
            word: Target word w_i.
            context: Context history (w_{i-n+1}, ..., w_{i-1}).
            smoothing: True for Laplace smoothing, False for pure MLE.

        Returns:
            Conditional probability float.
        """
        use_smoothing = self.laplace_smoothing if smoothing is None else smoothing
        ctx_tuple = self._normalize_context(context)

        count_ngram = self.ngram_counts.get((ctx_tuple, word), 0)
        count_ctx = self.context_counts.get(ctx_tuple, 0)
        v = self.vocab_size

        if use_smoothing:
            # Add-1 Laplace Smoothing
            if v == 0:
                return 0.0
            return (count_ngram + 1.0) / (count_ctx + float(v))
        else:
            # Pure Maximum Likelihood Estimation (MLE)
            if count_ctx == 0:
                return 0.0
            return float(count_ngram) / float(count_ctx)

    def calculate_log_likelihood(
        self, tokens: List[str], smoothing: Optional[bool] = None
    ) -> float:
        """Compute the total log-likelihood of a sequence: sum(log P(w_i | context)).

        Args:
            tokens: Input token sequence.
            smoothing: True for Laplace smoothing, False for MLE.

        Returns:
            Log-likelihood (natural log) float.
        """
        if not tokens:
            return 0.0

        padded = self._pad_tokens(tokens)
        needed_context = self.n - 1
        log_likelihood = 0.0

        for i in range(needed_context, len(padded)):
            ctx = tuple(padded[i - needed_context : i])
            word = padded[i]
            prob = self.get_probability(word, ctx, smoothing=smoothing)

            if prob <= 0.0:
                return float("-inf")
            log_likelihood += math.log(prob)

        return log_likelihood

    def calculate_sequence_probability(
        self, tokens: List[str], smoothing: Optional[bool] = None
    ) -> float:
        """Calculate the overall joint probability of a sequence.

        Args:
            tokens: Input token sequence.
            smoothing: True for Laplace smoothing, False for MLE.

        Returns:
            Sequence probability float.
        """
        ll = self.calculate_log_likelihood(tokens, smoothing=smoothing)
        if ll == float("-inf"):
            return 0.0
        try:
            return math.exp(ll)
        except OverflowError:
            return 0.0

    def calculate_perplexity(
        self, tokens: List[str], smoothing: Optional[bool] = None
    ) -> float:
        """Calculate sequence perplexity: PP(W) = exp(- (1/N) * sum(ln P(w_i | context))).

        Args:
            tokens: Input token sequence.
            smoothing: True for Laplace smoothing, False for MLE.

        Returns:
            Perplexity float (lower indicates higher language model predictability).
        """
        if not tokens:
            return float("inf")

        padded = self._pad_tokens(tokens)
        needed_context = self.n - 1
        n_predictions = len(padded) - needed_context

        if n_predictions <= 0:
            return float("inf")

        ll = self.calculate_log_likelihood(tokens, smoothing=smoothing)
        if ll == float("-inf"):
            return float("inf")

        # Perplexity = exp(- (1 / N) * log_likelihood)
        return math.exp(-ll / n_predictions)

    def predict_next_words(
        self,
        context: Union[Tuple[str, ...], List[str], str],
        top_k: int = 5,
        smoothing: Optional[bool] = None,
    ) -> List[Tuple[str, float]]:
        """Predict the most probable next words given context, sorted descending.

        Args:
            context: Context preceding the predicted word.
            top_k: Maximum number of predictions to return.
            smoothing: True for Laplace smoothing, False for MLE.

        Returns:
            List of (word, probability) tuples sorted highest to lowest.
        """
        ctx_tuple = self._normalize_context(context)
        use_smoothing = self.laplace_smoothing if smoothing is None else smoothing

        if not self.vocab:
            return []

        # If pure MLE and context unseen, no predictions possible
        if not use_smoothing and self.context_counts.get(ctx_tuple, 0) == 0:
            return []

        # For pure MLE, we only need to score words actually observed with ctx_tuple
        if not use_smoothing:
            candidates = [w for (c, w) in self.ngram_counts.keys() if c == ctx_tuple]
        else:
            candidates = list(self.vocab)

        scored = []
        for word in candidates:
            prob = self.get_probability(word, ctx_tuple, smoothing=use_smoothing)
            scored.append((word, prob))

        # Sort descending by probability, break ties alphabetically
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

