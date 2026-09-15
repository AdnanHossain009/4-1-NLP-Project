"""End-to-End NLP Training and Orchestration Pipeline.

Executes the entire project training workflow sequentially:
1. Load dataset (data/bbc_news.csv) and validate schema.
2. Preprocess text (HTML stripping, tokenization, lemmatization).
3. Stratified Train/Test split (80% train / 20% test, zero data leakage).
4. Train N-Gram Language Models (orders n = 2, 3, 4, 5).
5. Train Multinomial Naive Bayes from scratch & save weights.
6. Train/Load Custom Skip-Gram Word2Vec (PyTorch) & save weights.
7. Prepare/Load offline GloVe embedding cache & save subset.
8. Generate document vectors (Mean & TF-IDF, training-split IDF).
9. Train One-vs-Rest Multi-Class Logistic Regression from scratch & save weights.
10. Evaluate all 5 classification pipelines on held-out test split.
11. Generate confusion matrix visualization (reports/confusion_matrix.png).
12. Run Unsupervised K-Means clustering (k=5), compute Silhouette Score,
    and save 2D projection (reports/clusters.png).
13. Save benchmark metrics to reports/metrics.json.
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch

# Ensure project root is in sys.path when executed directly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import (
    load_dataset,
    stratified_train_test_split,
    TextPreprocessor,
    VALID_CATEGORIES,
)
from src.ngram_model import NGramLanguageModel
from src.naive_bayes import MultinomialNaiveBayes
from src.word2vec_scratch import CustomWord2Vec
from src.embeddings import (
    GloVeEmbeddings,
    extract_glove_subset,
    generate_offline_fallback_glove_cache,
    compute_training_idf,
    vectorize_corpus_mean,
    vectorize_corpus_tfidf,
)
from src.logistic_regression import (
    OneVsRestLogisticRegression,
    train_and_evaluate_all_pipelines,
)
from src.clustering_eval import evaluate_all_five_pipelines


def set_reproducible_seeds(seed: int = 42):
    """Set global random seeds for full reproducibility across NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_full_training_pipeline(
    csv_path: str = "data/bbc_news.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    retrain_w2v_if_exists: bool = False,
    w2v_epochs: int = 5,
    seed: int = 42,
) -> dict:
    """Execute the complete training and evaluation pipeline."""
    set_reproducible_seeds(seed)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    start_time = time.time()
    print("\n" + "=" * 80)
    print("STARTING COMPLETE END-TO-END NLP TRAINING PIPELINE")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Step 1: Dataset Acquisition & Validation
    # -----------------------------------------------------------------
    print("\n[Step 1/12] Loading BBC News Corpus & Validating Schema...")
    if not os.path.exists(csv_path):
        from data.setup_data import generate_fallback_dataset
        df = generate_fallback_dataset()
        df.to_csv(csv_path, index=False)
    else:
        df = load_dataset(csv_path)

    print(f"  Loaded {len(df):,} articles across categories: {sorted(list(df['category'].unique()))}")
    for cat in sorted(list(VALID_CATEGORIES)):
        count = int(np.sum(df["category"] == cat))
        print(f"    - {cat:15s}: {count:,} records")

    # -----------------------------------------------------------------
    # Step 2: Stratified Train / Test Split (Zero Data Leakage)
    # -----------------------------------------------------------------
    print("\n[Step 2/12] Performing Stratified Train/Test Split (80% Train, 20% Test)...")
    train_df, test_df = stratified_train_test_split(df, test_size=0.2, random_state=seed)
    print(f"  Training Split: {len(train_df):,} articles")
    print(f"  Testing  Split: {len(test_df):,} articles (Held-out evaluation)")

    # -----------------------------------------------------------------
    # Step 3: Text Preprocessing Pipeline
    # -----------------------------------------------------------------
    print("\n[Step 3/12] Preprocessing Articles (HTML stripping, tokenization, lemmatization)...")
    preprocessor = TextPreprocessor()
    train_tokens = [preprocessor.preprocess(t, return_tokens=True) for t in train_df["text"]]
    test_tokens = [preprocessor.preprocess(t, return_tokens=True) for t in test_df["text"]]
    y_train = train_df["category"].tolist()
    y_test = test_df["category"].tolist()
    print(f"  Preprocessed {len(train_tokens):,} training token sequences.")
    print(f"  Preprocessed {len(test_tokens):,} test token sequences.")

    # -----------------------------------------------------------------
    # Step 4: Generalized N-Gram Language Models (Orders 2, 3, 4, 5)
    # -----------------------------------------------------------------
    print("\n[Step 4/12] Training Generalized N-Gram Models (Bigram, Trigram, 4-Gram, 5-Gram)...")
    ngram_models = {}
    for n in [2, 3, 4, 5]:
        lm = NGramLanguageModel(n=n, laplace_smoothing=True)
        lm.train(train_tokens)
        ngram_models[n] = lm
        sample_preds = lm.predict_next_words(("government",), top_k=3)
        sample_str = ", ".join([f"{w} ({p:.3f})" for w, p in sample_preds])
        print(f"  Order n={n}: Vocab={lm.vocab_size:,} | Next after 'government': {sample_str}")

    # -----------------------------------------------------------------
    # Step 5: Multinomial Naive Bayes Classifier from Scratch
    # -----------------------------------------------------------------
    print("\n[Step 5/12] Training Multinomial Naive Bayes (Laplace Add-1 Smoothing)...")
    nb_weights_path = os.path.join(models_dir, "naive_bayes_weights.pkl")
    nb_model = MultinomialNaiveBayes(alpha=1.0)
    nb_model.fit(train_tokens, y_train)
    nb_train_acc = nb_model.score(train_tokens, y_train)
    nb_test_acc = nb_model.score(test_tokens, y_test)
    nb_model.save(nb_weights_path)
    print(f"  Naive Bayes: Vocab={nb_model.vocab_size_:,} | Train Acc={nb_train_acc*100:.2f}% | Test Acc={nb_test_acc*100:.2f}%")
    print(f"  Saved Naive Bayes weights -> {nb_weights_path}")

    # -----------------------------------------------------------------
    # Step 6: Custom Skip-Gram Word2Vec (PyTorch from Scratch)
    # -----------------------------------------------------------------
    print("\n[Step 6/12] Training / Loading Custom Skip-Gram Word2Vec...")
    w2v_model_path = os.path.join(models_dir, "custom_word2vec.pt")
    w2v_vocab_path = os.path.join(models_dir, "vocab.json")

    w2v = CustomWord2Vec(embedding_dim=100, window_size=2, min_count=5, lr=0.003, seed=seed)
    if os.path.exists(w2v_model_path) and os.path.exists(w2v_vocab_path) and not retrain_w2v_if_exists:
        print("  Existing Word2Vec model found on disk. Loading weights...")
        w2v.load(w2v_model_path, w2v_vocab_path)
    else:
        print(f"  Training Skip-Gram architecture for {w2v_epochs} epochs...")
        w2v.train(train_tokens, epochs=w2v_epochs, batch_size=2048, device="cpu", verbose=True)
        w2v.save(w2v_model_path, w2v_vocab_path)

    print(f"  Word2Vec Active Vocabulary: {w2v.vocab_size:,} words ({w2v.embedding_dim}d)")
    neighbors = w2v.find_nearest_neighbors("football", top_k=3)
    print(f"  Top neighbors for 'football': {', '.join([f'{w} ({s:.3f})' for w, s in neighbors])}")

    # -----------------------------------------------------------------
    # Step 7: Offline Pretrained GloVe Cache Management
    # -----------------------------------------------------------------
    print("\n[Step 7/12] Preparing Offline Pretrained GloVe Embedding Cache...")
    glove_txt_path = os.path.join(project_root, "data", "glove.6B.100d.txt")
    glove_npy_path = os.path.join(models_dir, "pretrained_embeddings.npy")
    glove_vocab_path = os.path.join(models_dir, "pretrained_embeddings_vocab.json")

    if os.path.exists(glove_txt_path):
        print(f"  Extracting BBC project vocabulary subset from {glove_txt_path}...")
        glove = extract_glove_subset(glove_txt_path, glove_npy_path, glove_vocab_path)
    elif os.path.exists(glove_npy_path) and os.path.exists(glove_vocab_path):
        print(f"  Loading existing GloVe cache from {glove_npy_path}...")
        glove = GloVeEmbeddings.load(glove_npy_path, glove_vocab_path)
    else:
        print("  Raw GloVe file not present at data/glove.6B.100d.txt.")
        print("  Generating deterministic offline baseline cache for testing...")
        glove = generate_offline_fallback_glove_cache(glove_npy_path, glove_vocab_path)

    print(f"  GloVe Cache: {glove.vocab_size:,} words ({glove.embedding_dim}d)")

    # -----------------------------------------------------------------
    # Step 8: Document Vector Aggregation (Zero Data Leakage IDF)
    # -----------------------------------------------------------------
    print("\n[Step 8/12] Generating Fixed-Length Document Vectors (Mean & TF-IDF)...")
    idf_weights, default_idf = compute_training_idf(train_tokens, smooth=True)
    print(f"  Computed frozen IDF table for {len(idf_weights):,} training words.")

    # -----------------------------------------------------------------
    # Step 9: One-vs-Rest Logistic Regression from Scratch
    # -----------------------------------------------------------------
    print("\n[Step 9/12] Training One-vs-Rest Logistic Regression on All Representations...")
    lr_weights_path = os.path.join(models_dir, "logistic_regression_weights.pkl")
    lr_results = train_and_evaluate_all_pipelines(
        csv_path=csv_path,
        save_path=lr_weights_path,
        learning_rate=0.5,
        n_iterations=800,
        verbose=True,
    )

    # -----------------------------------------------------------------
    # Step 10: Multi-Pipeline Comparative Benchmark Evaluation
    # -----------------------------------------------------------------
    print("\n[Step 10/12] Evaluating All 5 Pipelines on Held-Out Test Split...")
    eval_reports = evaluate_all_five_pipelines(
        csv_path=csv_path,
        reports_dir=reports_dir,
        verbose=True,
    )

    # -----------------------------------------------------------------
    # Step 11: Summary Verification Checklist
    # -----------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE — SUMMARY & VERIFICATION CHECKLIST")
    print("=" * 80)
    print(f"Elapsed Time: {elapsed:.2f} seconds")
    print("\nGenerated Model Artifacts:")
    for fn in os.listdir(models_dir):
        fpath = os.path.join(models_dir, fn)
        size_kb = os.path.getsize(fpath) / 1024.0
        print(f"  - models/{fn:<32} ({size_kb:>8.1f} KB)")

    print("\nGenerated Evaluation Reports:")
    for fn in os.listdir(reports_dir):
        fpath = os.path.join(reports_dir, fn)
        size_kb = os.path.getsize(fpath) / 1024.0
        print(f"  - reports/{fn:<31} ({size_kb:>8.1f} KB)")

    print("\nFinal Pipeline Test Accuracies:")
    for p_id, p_info in eval_reports["pipelines"].items():
        print(f"  * {p_info['name']:<42}: {p_info['accuracy']*100:.2f}% (Macro F1: {p_info['macro_f1']*100:.2f}%)")

    print("\n" + "=" * 80)
    print("Next step: Run `streamlit run app.py` to launch the interactive UI.")
    print("=" * 80 + "\n")

    return eval_reports


if __name__ == "__main__":
    run_full_training_pipeline()

