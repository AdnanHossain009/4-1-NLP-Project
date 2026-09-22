# News Category Classification and Semantic Analysis Using N-Gram Models, Word2Vec, Naive Bayes, and Logistic Regression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-68%20passed-brightgreen.svg)](https://pytest.org)
[![Offline First](https://img.shields.io/badge/architecture-offline--first-success.svg)](README.md)
[![Corpus](<https://img.shields.io/badge/corpus-BBC%20News%20(2%2C225%20docs)-orange.svg>)](data/setup_data.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, production-grade Natural Language Processing curriculum repository implementing foundational statistical and mathematical NLP algorithms from scratch for news categorization and semantic modeling.

---

## Table of Contents

1. [Project Overview & Objective](#1-project-overview--objective)
2. [Academic Constraints & Strict Exclusion of RAG/LLMs](#2-academic-constraints--strict-exclusion-of-ragllms)
3. [The "Gorur Rochona" Familiar Real-World Corpus](#3-the-gorur-rochona-familiar-real-world-corpus)
4. [Implemented Architecture](#4-implemented-architecture)
5. [Mathematical Formulations](#5-mathematical-formulations)
   - [5.1 Text Preprocessing Pipeline](#51-text-preprocessing-pipeline)
   - [5.2 Generalized N-Gram Language Model](#52-generalized-n-gram-language-model)
   - [5.3 Multinomial Naive Bayes from Scratch](#53-multinomial-naive-bayes-from-scratch)
   - [5.4 Custom Skip-Gram Word2Vec (PyTorch from Scratch)](#54-custom-skip-gram-word2vec-pytorch-from-scratch)
   - [5.5 Cosine Similarity & Nearest Neighbors](#55-cosine-similarity--nearest-neighbors)
   - [5.6 Offline GloVe Baseline & Caching](#56-offline-glove-baseline--caching)
   - [5.7 Document Vector Aggregation (Mean & TF-IDF)](#57-document-vector-aggregation-mean--tf-idf)
   - [5.8 One-vs-Rest (OvR) Logistic Regression from Scratch](#58-one-vs-rest-ovr-logistic-regression-from-scratch)
6. [Training Pipeline Orchestration (`train_pipeline.py`)](#6-training-pipeline-orchestration-train_pipelinepy)
7. [Evaluation Benchmarks & Comparative Analysis](#7-evaluation-benchmarks--comparative-analysis)
   - [7.1 Benchmark Comparison Table](#71-benchmark-comparison-table)
   - [7.2 Multi-Pipeline Confusion Matrices](#72-multi-pipeline-confusion-matrices)
   - [7.3 Unsupervised K-Means Clustering & Silhouette Score](#73-unsupervised-k-means-clustering--silhouette-score)
8. [Interactive Streamlit Web Dashboard (`app.py`)](#8-interactive-streamlit-web-dashboard-apppy)
9. [Two-Member Development Roadmap & Git Workflow](#9-two-member-development-roadmap--git-workflow)
10. [Installation & Setup](#10-installation--setup)
11. [Verification Checklist & Run Commands](#11-verification-checklist--run-commands)
12. [Example Test Inputs & Edge Cases](#12-example-test-inputs--edge-cases)
13. [Limitations & Future Work](#13-limitations--future-work)
14. [Conclusion](#14-conclusion)

---

## 1. Project Overview & Objective

Modern NLP education frequently degenerates into calling closed-source API endpoints (`chat.completions.create`) or chaining high-level black-box wrappers (`LangChain`, `LlamaIndex`, `HuggingFace pipelines`), leaving students without deep foundational grounding in probability theory, matrix operations, loss backpropagation, and geometric word vector spaces.

The objective of this curriculum project is to construct an entire, production-grade text classification and semantic analysis system **from first principles using pure mathematics, NumPy, and basic PyTorch neural building blocks**.

Every statistical and neural algorithm in this repository is derived from mathematical equations:

- **No external black-box classifier wrappers** (Zero `sklearn.naive_bayes.MultinomialNB`, zero `sklearn.linear_model.LogisticRegression`).
- **No RAG, no Transformer models, no BERT, no GPT, and zero runtime network queries.**
- **Every number shown in reports and the UI originates from real, local, trained weights.**

---

## 2. Academic Constraints & Strict Exclusion of RAG/LLMs

### Why RAG, External LLMs, and Transformers are Explicitly Excluded:

1. **Pedagogical Integrity**: Using a pre-trained Transformer (e.g., BERT or RoBERTa) or an LLM API renders the mathematical study of n-gram statistics, maximum likelihood estimation, Laplace smoothing, Skip-Gram negative sampling, term-frequency weighting, and batch gradient descent redundant.
2. **Deterministic Reproducibility & Offline First**: Generative APIs introduce non-deterministic hallucinations, network latency, token rate limits, and API deprecation. Our system runs 100% offline with full determinism.
3. **Transparent Complexity & Zero Black Boxes**: By hand-coding the sigmoid function with numerical clipping, cross-entropy loss, log-sum-exp normalization, and One-vs-Rest classification, students understand the exact internal gradients driving weight convergence.

---

## 3. The "Gorur Rochona" Familiar Real-World Corpus

Rather than arbitrary synthetic text, this project employs the classic, highly cited **BBC News Dataset** (_Greene & Cunningham, 2006_). This corpus serves as our familiar, intuitive benchmark—analogous to the classic _"Gorur Rochona"_ (the essay on the cow):

- **Total Documents**: 2,225 articles
- **Universal Classes**: 5 balanced categories
  - **Business**: 510 articles
  - **Entertainment**: 386 articles
  - **Politics**: 417 articles
  - **Sport**: 511 articles
  - **Tech**: 401 articles
- **Acquisition**: Automatically downloaded via `data/setup_data.py` directly from University College Dublin (`http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip`) with an offline deterministic fallback generator.
- **Partitioning**: 80% Training ($1,780$ articles) / 20% Testing ($445$ articles) via stratified splitting with fixed seed $42$.

---

## 4. Implemented Architecture

```text
       ┌─────────────────────────────────────────────────────────┐
       │             BBC News Corpus (data/bbc_news.csv)         │
       │   5 Classes: Business, Entertainment, Politics, Sport,  │
       │              Tech (2,225 records)                       │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │             Text Preprocessor Pipeline (Step 1)         │
       │  - HTML Stripping (<[^>]+?>) & URL Removal              │
       │  - Lowercasing & Non-Alphabetic Character Filtering     │
       │  - NLTK Word Tokenization & Stop-word Elimination       │
       │  - WordNet Morphological Lemmatization                  │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │         Stratified Train / Test Split (Zero Leakage)    │
       │  - 80% Training (1,780 docs) / 20% Test (445 docs)      │
       └──────┬─────────────────────┬─────────────────────┬──────┘
              │                     │                     │
              ▼                     ▼                     ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  N-Gram Model (Step 2)│ │  Naive Bayes (Step 3)│ │ Custom W2V (Step 4)  │
│  - Orders n=2,3,4,5  │ │  - Prior P(c)=N_c/N  │ │  - Skip-Gram PyTorch │
│  - Laplace Smoothing │ │  - Add-1 Likelihood  │ │  - 100d Embeddings   │
│  - Next-word Predict │ │  - Log-Sum-Exp Probs │ │  - Cosine Similarity │
└──────────────────────┘ └──────────┬───────────┘ └──────────┬───────────┘
                                    │                        │
                                    │   ┌────────────────────┴───────────┐
                                    │   │ Pretrained GloVe Cache (Step 5)│
                                    │   │ - 100d Offline Filtered Subset │
                                    │   └────────────┬───────────────────┘
                                    │                │
                                    ▼                ▼
       ┌─────────────────────────────────────────────────────────┐
       │      Document Vector Aggregations (Mean & TF-IDF)       │
       │  - Strict Training-Split IDF Weights (Zero Data Leakage)│
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │     One-vs-Rest Multi-Class Logistic Regression (Step 6)│
       │  - 5 Binary Classifiers (NumPy Batch Gradient Descent)  │
       │  - Sigmoid Activation with Numerical Overflow Clipping   │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │   Orchestration, Benchmarks & Streamlit Dashboard       │
       │  - train_pipeline.py: End-to-end reproducible runner    │
       │  - src/clustering_eval.py: Metrics, CM, K-Means (k=5)   │
       │  - app.py: Full interactive web UI (Streamlit)          │
       └─────────────────────────────────────────────────────────┘
```

---

## 5. Mathematical Formulations

### 5.1 Text Preprocessing Pipeline

Text normalization removes spurious noise while retaining semantic tokens:

1. **HTML Stripping**: Regex substitution $\text{re.sub}(r"<[^>]+?>", " ", \text{text})$.
2. **Lowercasing & Non-Alphabetic Filtering**: Retains exclusively `[a-z\s]`.
3. **Tokenization & Stop-word Filtering**: Removes closed-class function words using NLTK English stopwords.
4. **WordNet Lemmatization**: Reduces inflected forms to base lemmas ($\text{lemmatize}(w, \text{pos}='v')$ then noun fallback).

### 5.2 Generalized N-Gram Language Model

For arbitrary order $n \ge 1$, each sentence is padded with $n-1$ start tokens (`<s>`) and one end token (`</s>`).

- **Maximum Likelihood Estimation (MLE)**:
  $$P_{\text{MLE}}(w_i \mid w_{i-n+1}, \dots, w_{i-1}) = \frac{C(w_{i-n+1}, \dots, w_{i-1}, w_i)}{C(w_{i-n+1}, \dots, w_{i-1})}$$
- **Add-1 Laplace Smoothing**:
  $$P_{\text{Laplace}}(w_i \mid \text{context}) = \frac{C(\text{context}, w_i) + 1}{C(\text{context}) + |V|}$$
- **Perplexity Computation**:
  $$\text{Perplexity}(W) = \exp\left( - \frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid \text{context}_i) \right)$$

### 5.3 Multinomial Naive Bayes from Scratch

- **Class Prior**:
  $$P(c) = \frac{N_c}{N}, \quad \log P(c) = \ln(N_c / N)$$
- **Word Likelihood with Add-1 Smoothing**:
  $$P(w \mid c) = \frac{\text{count}(w, c) + 1}{\sum_{w'} \text{count}(w', c) + |V|}$$
- **Log-Space Scoring & Log-Sum-Exp Normalization**:
  $$\log P(c \mid D) \propto \log P(c) + \sum_{w \in D} f_{w, D} \cdot \log P(w \mid c)$$
  $$P(c \mid D) = \frac{\exp(s_c - \max_{k} s_k)}{\sum_{c'} \exp(s_{c'} - \max_k s_k)}$$

### 5.4 Custom Skip-Gram Word2Vec (PyTorch from Scratch)

Trains $100$-dimensional distributed dense representations by maximizing context prediction:

- **Sliding Context Window ($w=2$)**: Generates target-context pairs $(w_c, w_o)$.
- **Architecture**: Center lookup layer $E \in \mathbb{R}^{V \times d}$ and projection layer $W_{\text{proj}} \in \mathbb{R}^{d \times V}$.
- **Conditional Probability**:
  $$P(w_o \mid w_c) = \frac{\exp({v'_{w_o}}^T v_{w_c})}{\sum_{j=1}^V \exp({v'_{w_j}}^T v_{w_c})}$$
- **Optimization**: Mini-batch Cross-Entropy minimization via Adam optimizer ($\text{lr}=0.003$).

### 5.5 Cosine Similarity & Nearest Neighbors

Evaluates geometric orientation invariant to vector magnitude:
$$\text{cosine}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|} = \frac{\sum_{i=1}^d a_i b_i}{\sqrt{\sum_{i=1}^d a_i^2} \sqrt{\sum_{i=1}^d b_i^2}}$$

### 5.6 Offline GloVe Baseline & Caching

- GloVe 6B ($100$d) provides a competitive pre-trained baseline.
- **One-Time Offline Extraction**: Scans raw `data/glove.6B.100d.txt` and caches only the $7,116$ BBC vocabulary tokens to `models/pretrained_embeddings.npy` ($2.8$ MB) and `models/pretrained_embeddings_vocab.json`.
- Zero network querying at runtime.

### 5.7 Document Vector Aggregation (Mean & TF-IDF)

- **Mean Pooling**:
  $$X_{\text{mean}} = \frac{1}{|V_D|} \sum_{w_i \in V_D} \vec{v}(w_i) \quad (\text{zero vector if } |V_D| = 0)$$
- **TF-IDF Weighted Pooling**:
  $$X_{\text{tfidf}} = \frac{\sum_{w_i \in V_D} \text{TF}(w_i, D) \cdot \text{IDF}(w_i) \cdot \vec{v}(w_i)}{\sum_{w_i \in V_D} \text{TF}(w_i, D) \cdot \text{IDF}(w_i)}$$
- **Strict Zero Data Leakage Proof**:
  Inverse Document Frequencies are computed **strictly from the training partition**:
  $$\text{IDF}(w) = \ln\left( \frac{N_{\text{train}} + 1}{\text{DF}_{\text{train}}(w) + 1} \right) + 1$$
  The test partition is never touched during IDF fitting.

### 5.8 One-vs-Rest (OvR) Logistic Regression from Scratch

Decomposes 5-class classification into five independent binary classifiers.

- **Sigmoid with Numerical Guard**:
  $$z = \text{clip}(XW + b, -250, 250), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Batch Gradient Descent Updates**:
  $$\frac{\partial J}{\partial W} = \frac{1}{m} X^T (\hat{y} - y) + \lambda W, \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)$$
  $$W \leftarrow W - \alpha \frac{\partial J}{\partial W}, \quad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
- **Multi-Class Probability Normalization**:
  $$P(c \mid D) = \frac{\sigma(X W_c + b_c)}{\sum_{k=1}^5 \sigma(X W_k + b_k)}$$

### 5.9 Custom TF-IDF & N-Gram Feature Engineering (`src/tfidf_extractor.py`)

Averaged word embeddings (Word2Vec / GloVe) inherently destroy word order and phrase boundaries. In contrast, **TF-IDF with Unigrams and Consecutive Bigrams** directly captures discriminative multi-word phrases (e.g., *"world_cup"*, *"stock_market"*, *"prime_minister"*):

- **N-Gram Generation**:
  $$\text{Unigrams}: w_i \quad \mid \quad \text{Bigrams}: w_i\_w_{i+1}$$
- **Term Frequency**:
  $$\text{TF}(t, d) = \text{count}(t, d)$$
- **Strict Training-Only Smoothed IDF (Zero Leakage)**:
  $$\text{IDF}(t) = \ln\left( \frac{N_{\text{train}} + 1}{\text{DF}_{\text{train}}(t) + 1} \right) + 1$$
- **Document Vector & L2 Normalization**:
  $$v_d(t) = \text{TF}(t, d) \times \text{IDF}(t), \quad x_d = \frac{v_d}{\|v_d\|_2} = \frac{v_d}{\sqrt{\sum_j v_{d, j}^2}}$$
- **Direct Mathematical Explainability**:
  Because each feature dimension $j$ maps to an explicit vocabulary n-gram, the linear score contribution for class $c$ is:
  $$\text{Contribution}_c(j) = x_d[j] \times W_c[j]$$
  Positive weights $W_c[j] > 0$ directly explain the observable terms driving classification.

---

## 6. Training Pipeline Orchestration (`train_pipeline.py`)

The orchestration script runs the entire sequence automatically:

```bash
python -m src.logistic_regression
python -m src.clustering_eval
```

**Execution Sequence**:

1. Load dataset & validate schemas (`data/bbc_news.csv`)
2. Stratified 80/20 train/test split (seed 42)
3. Text cleaning, tokenization, lemmatization (supporting Standard and Numeric-preserving configurations)
4. Train N-Gram models ($n=2, 3, 4, 5$)
5. Train & save Naive Bayes (`models/naive_bayes_weights.pkl`)
6. Train & save Custom Word2Vec (`models/custom_word2vec.pt`, `models/vocab.json`)
7. Prepare offline GloVe cache (`models/pretrained_embeddings.npy`)
8. Fit Unigram & Bigram TF-IDF Extractors (`models/tfidf_extractor.pkl`, `models/tfidf_unigram_extractor.pkl`)
9. Train One-vs-Rest Logistic Regression across all representation configurations
10. Extract top learned positive feature weights per category (`reports/top_features_per_category.json`)
11. Evaluate all 7 configurations on the 445-article held-out test split (`reports/model_comparison.csv`)
12. Conduct comprehensive error analysis on misclassified test articles (`reports/error_analysis.txt`)
13. Generate multi-panel confusion matrix plot (`reports/confusion_matrix.png`)
14. Save JSON benchmark report (`reports/metrics.json`)

---

## 7. Evaluation Benchmarks & Comparative Analysis

### 7.1 Benchmark Comparison Table (All 7 Traditional Configurations)

Evaluated on the held-out test split ($445$ articles, $20\%$):

| Pipeline ID | Model | Representation / Aggregation | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A** | Multinomial Naive Bayes | Count / Bag-of-Words (BoW) | **97.53%** | 97.47% | 97.51% | 97.48% | 97.53% |
| **Pipeline B** | Logistic Regression (OvR) | Custom Word2Vec (Mean Pooling) | **96.40%** | 96.36% | 96.26% | 96.30% | 96.41% |
| **Pipeline C** | Logistic Regression (OvR) | Custom Word2Vec (TF-IDF Weighted) | **96.18%** | 96.15% | 96.00% | 96.05% | 96.18% |
| **Pipeline D** | Logistic Regression (OvR) | Pretrained GloVe (Mean Pooling) | **76.63%** | 82.20% | 74.96% | 75.73% | 76.06% |
| **Pipeline E** | Logistic Regression (OvR) | Pretrained GloVe (TF-IDF Weighted) | **68.31%** | 78.43% | 66.27% | 67.47% | 67.74% |
| **Pipeline F** | Logistic Regression (OvR) | TF-IDF (Unigram Features) | **97.08%** | 97.09% | 97.13% | 97.10% | 97.09% |
| **Pipeline G** | Logistic Regression (OvR) | TF-IDF (Unigram + Bigram Features) | **97.30%** | 97.30% | 97.33% | 97.31% | 97.32% |

_Key Empirical Insights_:
1. **Unigram + Bigram Superiority**: Adding bigram phrase features elevates Logistic Regression accuracy to **97.30%** (Macro F1: **97.31%**), outperforming all dense word-embedding pooling models (Word2Vec at 96.40% and GloVe at 76.63%).
2. **Domain-Specific vs Generic Embeddings**: Custom Word2Vec Skip-Gram trained directly on the BBC News corpus beats out-of-domain GloVe by over **+20% accuracy**, demonstrating the vital importance of corpus-specific vocabulary alignment.
3. **Naive Bayes Strength**: Multinomial Naive Bayes with add-1 Laplace smoothing remains an extraordinarily strong traditional baseline (**97.53%** accuracy).

### 7.2 Actual Test Set Error Analysis (`reports/error_analysis.txt`)

Out of 445 held-out test articles, Pipeline G makes only **12 misclassifications** (Accuracy: 97.30%). Analysis reveals genuine cross-domain lexical overlaps:
- **Sample #112** (*"News Corp eyes video games market"*): Rupert Murdoch expanding into Activision video games. Actual: **Business**, Predicted: **Tech** (Confidence: 41.2%) due to dominant tech terms (`game`, `video`, `microsoft`).
- **Sample #222** (*"Games maker fights for survival"*): Argonaut Games entering administration with 100 layoffs. Actual: **Tech**, Predicted: **Business** (Confidence: 37.9%) due to corporate crisis terms (`company`, `sale`, `stock`, `share`).
- **Sample #285** (*"Report attacks defence spending"*): National Audit Office criticizing £1.7bn MoD equipment overruns. Actual: **Politics**, Predicted: **Business** (Confidence: 40.6%) due to budgetary terms (`bn`, `cost`, `rise`, `report`).
- **Sample #435** (*"UK pioneers digital film network"*): UK Film Council £11.5m grant for 250 digital cinema screens. Actual: **Tech**, Predicted: **Entertainment** (Confidence: 47.9%) due to media tokens (`film`, `cinema`, `screen`).

### 7.3 Diagnostic Case Study: Short Text vs Full Match Report

We evaluated the diagnostic prompt without hardcoding:
- **Short Input** (*"Brazil lost 7-1 against Germany"*):
  - Tokens: `['brazil', 'lose', 'germany']`
  - Predicted: **Sport** (Confidence: **31.90%** | Business: 26.59%, Entertainment: 16.28%, Politics: 12.70%, Tech: 12.54%).
  - *Observation*: Without explicit athletic terminology (*"cup"*, *"football"*, *"goal"*), the country names *"Germany"* (which appears 35 times in Business vs 18 in Sport in the 2004–2005 BBC corpus) introduce expected statistical ambiguity.
- **Informative Input** (*"Germany defeated Brazil 7-1 in the World Cup semifinal, scoring five goals in the first half and reaching the final after a remarkable football performance."*):
  - Tokens: `['germany', 'defeat', 'brazil', 'world', 'cup', 'semifinal', 'score', 'five', 'goal', 'first', 'half', 'reach', 'final', 'remarkable', 'football', 'performance']`
  - Predicted: **Sport** (Confidence: **61.02%** | Business: 13.29%, Entertainment: 11.01%, Tech: 7.65%, Politics: 7.02%).
  - *Active Features*: `world_cup` (bigram), `first_half` (bigram), `cup`, `goal`, `final`, `score` strongly reinforced the Sport prediction.

_Key Takeaway_: Domain-specific Custom Word2Vec embeddings trained directly on the BBC corpus out-perform generic out-of-domain GloVe vectors by over $+20\%$ accuracy. Naive Bayes remains an extraordinarily strong baseline on bag-of-words text distributions ($97.53\%$).

### 7.2 Multi-Pipeline Confusion Matrices

Saved to `reports/confusion_matrix.png`:

- Across all 5 categories (`business`, `entertainment`, `politics`, `sport`, `tech`), diagonal true-positive counts dominate.
- Sport classification achieves the highest individual category F1 ($> 98\%$), driven by distinct athletic lexicons.
- Minor cross-category confusion occurs between `business` and `politics` due to overlapping economic governance vocabulary.

### 7.3 Unsupervised K-Means Clustering & Silhouette Score

Saved to `reports/clusters.png`:

- **Algorithm**: Unsupervised K-Means with $k=5$ clusters.
- **Silhouette Score**: `0.1760` (reflecting continuous semantic document density).
- **2D PCA Projection**: Highlights natural spatial clustering across articles.
- **Academic Rule Enforced**: _Unsupervised clustering measures geometric spatial compactness, NOT classification accuracy._

---

## 8. Interactive Streamlit Web Dashboard (`app.py`)

Launch the web application:

```bash
streamlit run app.py
```

### Features:

1. **Live Classification & Confidence Meter**:
   - Select between Naive Bayes and Logistic Regression.
   - Choose representation (BoW, Custom Word2Vec, GloVe) and aggregation (Mean, TF-IDF).
   - Instant live inference computing exact class probability distributions.
2. **N-Gram Language Model Analysis**:
   - Real-time n-gram extraction (Bigram through 5-gram).
   - Top-5 next-word conditional probability predictions $P(w \mid \text{context})$.
3. **Word2Vec Semantic Explorer**:
   - Interactive cosine similarity query for any vocabulary term.
   - Dynamic nearest neighbor ranking with visual progress bars.
4. **Benchmark & Evaluation Center**:
   - Interactive benchmark data tables.
   - Visual confusion matrix and 2D K-Means cluster maps.
5. **Zero Page-Load Retraining**:
   - Utilizes `@st.cache_resource` for sub-second page rendering and model reuse.

---

## 9. Two-Member Development Roadmap & Git Workflow

| Phase      | Milestone                                                    | Assignee |  Status   |
| :--------- | :----------------------------------------------------------- | :------: | :-------: |
| **Part 0** | Git Architecture, Directory Layout & Package Initialization  |  Adnan   | Completed |
| **Part 1** | BBC Corpus Acquisition, Text Preprocessor & Stratified Split |  Adnan   | Completed |
| **Part 2** | Generalized N-Gram Language Model (MLE, Laplace, Perplexity) |  Akash   | Completed |
| **Part 3** | Multinomial Naive Bayes from Scratch (Pure NumPy)            |  Adnan   | Completed |
| **Part 4** | Custom Skip-Gram Word2Vec (PyTorch from Scratch)             |  Akash   | Completed |
| **Part 5** | Offline GloVe Cache & Document Vector Aggregation            |  Adnan   | Completed |
| **Part 6** | One-vs-Rest Multi-Class Logistic Regression (NumPy)          |  Akash   | Completed |
| **Part 7** | Evaluation Benchmarks, Confusion Matrix & K-Means Clustering |  Adnan   | Completed |
| **Part 8** | Interactive Streamlit Graphical Dashboard (`app.py`)         |  Akash   | Completed |
| **Part 9** | End-to-End Orchestration & Integration Verification          |   Both   | Completed |

---

## 10. Installation & Setup

### 1. Clone Repository & Initialize Environment

```bash
git clone https://github.com/AdnanHossain009/4-1-NLP-Project.git
cd 4-1-NLP-Project

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest
```

### 2. BBC News Dataset

Acquire the 2,225 BBC News articles:

```bash
python data/setup_data.py
```

### 3. Optional: Pretrained GloVe Embeddings

To use raw Stanford GloVe vectors instead of the automatic offline cache:

1. Download `https://nlp.stanford.edu/data/glove.6B.zip`
2. Extract `glove.6B.100d.txt` and place it at `data/glove.6B.100d.txt`
3. Run extraction: `python src/embeddings.py`

---

## 11. Verification Checklist & Run Commands

Execute the following commands in exact sequential order:

```bash
# 1. Run Complete Automated Pytest Suite (60 tests)
python -m pytest tests/ -v

# 2. Run Complete End-to-End Training & Evaluation Pipeline
python train_pipeline.py

# 3. Launch Interactive Streamlit Web Interface
streamlit run app.py
```

---

## 12. Example Test Inputs & Edge Cases

Try these inputs in the Streamlit interface to verify real-time inference across all categories:

| Category          | Sample Text Input                                                                                           |    Expected Prediction     |
| :---------------- | :---------------------------------------------------------------------------------------------------------- | :------------------------: |
| **Sport**         | _"The striker scored two decisive goals in the football championship final to secure the trophy."_          |     `sport` ($> 95\%$)     |
| **Tech**          | _"Software engineers developed an advanced neural network algorithm for mobile computer microprocessors."_  |     `tech` ($> 95\%$)      |
| **Business**      | _"Corporate profits climbed as central banks cut interest rates to stimulate investment and employment."_   |   `business` ($> 90\%$)    |
| **Politics**      | _"The prime minister addressed parliament to defend the new taxation budget and welfare spending bill."_    |   `politics` ($> 90\%$)    |
| **Entertainment** | _"The film festival honored independent cinematic directors and actors during the annual red carpet gala."_ | `entertainment` ($> 95\%$) |

### Edge Case Verification:

- **Empty Input (`""`)**: Displays informative warning banner without throwing an exception.
- **Pure Out-of-Vocabulary (`"alienx yzqqq foobar123"`)**: Safely falls back to prior distribution with an all-zeros pooled embedding vector.
- **Very Short Text (`"goal match"` / `"budget tax"`)**: Rapidly classified correctly using available tokens.
- **Very Long Text ($> 1,000$ words)**: Processed in milliseconds via vectorized matrix multiplication.

---

## 13. Limitations & Future Work

1. **Bag-of-Words Order Invariance**: Naive Bayes and mean vector pooling ignore word sequence syntax. Future work could incorporate bi-directional sequence models (e.g. custom recurrent architectures).
2. **Polysemy in Static Embeddings**: Word2Vec and GloVe assign a single vector per word, collapsing distinct meanings (e.g. _bank_ as financial institution vs _bank_ of a river).
3. **Subword Information**: Out-of-vocabulary words are currently ignored; introducing FastText-style character n-gram embeddings would capture morphology.

---

## 14. Conclusion

This project successfully proves that robust, high-accuracy ($> 97\%$) NLP applications can be constructed entirely from fundamental mathematical and statistical algorithms without relying on billions of external parameters, RAG pipelines, or proprietary LLM APIs. By maintaining complete offline reproducibility and formula-first engineering, this codebase stands as a comprehensive educational and practical benchmark in classical and neural Natural Language Processing.
