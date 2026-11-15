"""
Script to train and save all Phase 2 models and vectorizers.
Usage: python train_models.py
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText

os.makedirs("models", exist_ok=True)

data_path = os.path.join("data", "processed", "cleaned_news_corpus.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cleaned corpus not found at {data_path}. Run Phase 1 notebook first.")

print("Loading cleaned dataset...")
df = pd.read_csv(data_path)
df = df.dropna(subset=['clean_text']).reset_index(drop=True)
corpus = df['clean_text'].astype(str).tolist()
tokenized_corpus = [text.split() for text in corpus]

print(f"Loaded {len(corpus):,} clean articles.")

# 1. CountVectorizer (BoW)
print("Training CountVectorizer (BoW)...")
bow_vec = CountVectorizer(min_df=2)
bow_vec.fit(corpus)
joblib.dump(bow_vec, "models/bow_vectorizer.joblib")

# 2. TfidfVectorizer
print("Training TfidfVectorizer...")
tfidf_vec = TfidfVectorizer(min_df=2)
tfidf_vec.fit(corpus)
joblib.dump(tfidf_vec, "models/tfidf_vectorizer.joblib")

# 3. Word2Vec CBOW
print("Training Word2Vec CBOW...")
cbow = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=2, sg=0, epochs=5, workers=4, seed=42)
cbow.save("models/cbow_model.model")

# 4. Word2Vec Skip-gram
print("Training Word2Vec Skip-gram...")
skipgram = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=2, sg=1, epochs=5, workers=4, seed=42)
skipgram.save("models/skipgram_model.model")

# 5. fastText
print("Training fastText...")
fasttext = FastText(tokenized_corpus, vector_size=100, window=5, min_count=2, epochs=5, workers=4, seed=42)
fasttext.save("models/fasttext_model.model")

print("All Phase 2 models and vectorizers trained and saved in models/ directory.")
