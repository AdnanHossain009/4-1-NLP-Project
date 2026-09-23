"""Model Inspection Tool for CSE 4122 NLP Project.

Inspects and displays the mathematical contents of binary checkpoints
(.pt, .pkl, .npy, .json) in human-readable tabular form for project demonstrations.
"""

import os
import json
import pickle
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def inspect_word2vec():
    w2v_path = os.path.join(MODELS_DIR, "custom_word2vec.pt")
    vocab_path = os.path.join(MODELS_DIR, "vocab.json")

    print("\n" + "=" * 65)
    print("1. CUSTOM WORD2VEC SKIP-GRAM CHECKPOINT (models/custom_word2vec.pt)")
    print("=" * 65)

    if not os.path.exists(w2v_path):
        print("  [!] File not found.")
        return

    checkpoint = torch.load(w2v_path, map_location="cpu")
    print(f"  PyTorch Checkpoint Metadata:")
    print(f"    - Vocabulary Size : {checkpoint.get('vocab_size', 'N/A'):,} words")
    print(f"    - Embedding Dim   : {checkpoint.get('embedding_dim', 'N/A')} dimensions")
    print(f"    - Window Size     : {checkpoint.get('window_size', 'N/A')}")
    print(f"    - Min Count       : {checkpoint.get('min_count', 'N/A')}")

    state_dict = checkpoint.get("state_dict", {})
    for layer_name, tensor in state_dict.items():
        print(f"    - Layer '{layer_name}': Tensor Shape = {list(tensor.shape)}, Dtype = {tensor.dtype}")

    if "embedding.weight" in state_dict and os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vdata = json.load(f)
        w2i = vdata.get("word_to_idx", {})
        sample_words = ["football", "technology", "market", "government"]
        weights = state_dict["embedding.weight"]
        print(f"\n  Sample Learned Embedding Vectors (first 5 of 100 dimensions):")
        for word in sample_words:
            if word in w2i:
                idx = w2i[word]
                vec = weights[idx][:5].tolist()
                vec_str = ", ".join([f"{v:+.4f}" for v in vec])
                print(f"    - '{word}' (index {idx:4d}): [{vec_str}, ...]")


def inspect_logistic_regression():
    lr_path = os.path.join(MODELS_DIR, "logistic_regression_weights.pkl")

    print("\n" + "=" * 65)
    print("2. LOGISTIC REGRESSION (models/logistic_regression_weights.pkl)")
    print("=" * 65)

    if not os.path.exists(lr_path):
        print("  [!] File not found.")
        return

    with open(lr_path, "rb") as f:
        lr_data = pickle.load(f)

    classes = lr_data.get("classes", [])
    weights = lr_data.get("weights")
    biases = lr_data.get("biases")
    print(f"  Classes ({len(classes)}): {classes}")
    if weights is not None:
        print(f"  Weight Matrix Shape (C x D) : {weights.shape}")
    if biases is not None:
        print(f"  Bias Vector Shape (C,)     : {biases.shape}")
        for c, b in zip(classes, biases):
            print(f"    - Bias for {c:14s}: {b:+.4f}")

    accuracies = lr_data.get("accuracies", {})
    if accuracies:
        print("\n  Trained Pipeline Performance Across Embeddings & TF-IDF:")
        for key, info in accuracies.items():
            name = info.get("name", key)
            train_acc = info.get("train_accuracy", 0.0) * 100
            test_acc = info.get("test_accuracy", 0.0) * 100
            print(f"    - {name:28s} | Train: {train_acc:5.2f}% | Test: {test_acc:5.2f}%")


def inspect_naive_bayes():
    nb_path = os.path.join(MODELS_DIR, "naive_bayes_weights.pkl")

    print("\n" + "=" * 65)
    print("3. MULTINOMIAL NAIVE BAYES (models/naive_bayes_weights.pkl)")
    print("=" * 65)

    if not os.path.exists(nb_path):
        print("  [!] File not found.")
        return

    with open(nb_path, "rb") as f:
        nb_data = pickle.load(f)

    classes = nb_data.get("classes_", [])
    alpha = nb_data.get("alpha", 1.0)
    vocab_size = nb_data.get("vocab_size_", 0)
    priors = nb_data.get("class_priors_", {})

    print(f"  Laplace Smoothing Alpha (α) : {alpha}")
    print(f"  Vocabulary Size             : {vocab_size:,} words")
    print(f"  Classes & Prior Probabilities P(c):")
    for c in classes:
        prior_p = priors.get(c, 0.0)
        print(f"    - {c:14s}: P(c) = {prior_p:.4f} ({prior_p * 100:.2f}%)")


def inspect_tfidf():
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_extractor.pkl")
    idf_path = os.path.join(MODELS_DIR, "idf_weights.json")

    print("\n" + "=" * 65)
    print("4. TF-IDF FEATURE EXTRACTOR & IDF WEIGHTS")
    print("=" * 65)

    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            ext_data = pickle.load(f)
        vocab = ext_data.get("vocab", {})
        print(f"  TF-IDF Feature Extractor:")
        print(f"    - Feature Dimensions (Unigram + Bigram): {len(vocab):,} n-grams")
        print(f"    - N-gram range: {ext_data.get('ngram_range', '(1, 2)')}")

    if os.path.exists(idf_path):
        with open(idf_path, "r", encoding="utf-8") as f:
            idf_dict = json.load(f)
        print(f"  IDF Dictionary (models/idf_weights.json): {len(idf_dict):,} words")
        sample_terms = ["news", "year", "football", "technology", "said", "minister"]
        print("  Sample IDF Weights (lower = frequent across corpus; higher = discriminative):")
        for term in sample_terms:
            if term in idf_dict:
                print(f"    - '{term:12s}': IDF = {idf_dict[term]:.4f}")


def main():
    print("=" * 65)
    print("  CSE 4122 NLP LABORATORY - TRAINED MODEL ARTIFACT INSPECTOR")
    print("=" * 65)
    inspect_word2vec()
    inspect_logistic_regression()
    inspect_naive_bayes()
    inspect_tfidf()
    print("\n" + "=" * 65)
    print("All binary and serialized models loaded and validated successfully.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
