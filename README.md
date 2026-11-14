# Introduction to NLP: Project Repository

## Project Overview
- Topic: Fake vs. Real News Detection
- Dataset: McIntire Fake/Real News Corpus (6,335 articles, balanced binary target)
- Course: Introduction to NLP

## Team Members
- Muhammad Faizan (1071)
- Muhammad Asim (0840)
- Daud Arif (0833)

---

## Phases Overview

### Phase 1: Data Acquisition & Preprocessing (`assignment-1`)
- Data downloading and directory setup
- Exploratory data analysis, Zipf's law distribution, messy text identification
- Tokenizer benchmarking (whitespace vs. NLTK vs. spaCy)
- 10-step cleaning pipeline: Unicode NFC, HTML unescape, dateline stripping, URL/handle removal, bracket stripping, lowercasing, NLTK tokenization, negation-safe stopword removal, memoized spaCy lemmatization
- Cleaned corpus exported to `data/processed/cleaned_news_corpus.csv`

### Phase 2: Word Representations (`assignment-2`)
- Part A: Count-based representations
  - Bag of Words document-term count matrix (shape, non-zero entries, sparsity)
  - TF-IDF matrix with analysis of high topical weights vs. ubiquitous words driven toward zero by low IDF
  - N-gram feature explosion (unigrams, bigrams, trigrams) and sparsity trade-offs
  - Hand-worked numerical calculations on 3-doc toy corpus verifying manual TF-IDF and bigram formulas against scikit-learn
- Part B: Learned embeddings
  - Word2Vec CBOW (`sg=0`) and Skip-gram (`sg=1`) trained with identical hyperparameters
  - Comparative analysis of context-to-centre vs. centre-to-context objectives
  - fastText training with sub-word character n-grams and evaluation on misspelled/OOV queries (`presidnt`, `disinformaton`, `politiician`)
- Part C: Probing & semantic evaluation
  - Top-5 nearest neighbors for 5 political news terms
  - Cosine similarity evaluation across related vs. unrelated word pairs
  - Word vector analogy testing (`king - man + woman` and political relationships) with honest failure analysis
  - Comparison table: count-based vs. learned embeddings (dimensionality, sparsity, OOV, semantics)
- Part D: Phase 3 packaging
  - Trained vectorizers and embedding models saved to `models/`
  - Unified interface `get_document_representation(raw_doc, rep)` using exact Phase 1 cleaning pipeline

---

## Setup & Reproduction

### 1. Environment Setup
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

### 2. Running the Pipelines
- Phase 1: Open and run `NLP_Assignment_1_Pipeline.ipynb`
- Phase 2: Open and run `NLP_Assignment_2.ipynb`

### 3. Model Regeneration (Optional)
To retrain and regenerate all models and vectorizers from scratch:
```bash
python train_models.py
```
