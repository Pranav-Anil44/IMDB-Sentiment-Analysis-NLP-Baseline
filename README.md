# 🎬 IMDB Sentiment Analysis — Classic NLP Baseline Models

This project implements **classic Natural Language Processing techniques** on the IMDB Movie Review dataset to classify reviews as *positive* or *negative*.  
It forms **Week 1** of my structured NLP → LLM → GenAI learning journey.

## 📌 Overview

The IMDB dataset contains **50,000 labeled movie reviews**.  
This project explores two strong baseline approaches:

- **TF-IDF + Logistic Regression**
- **Average Word2Vec Embeddings + Logistic Regression**

Both models are evaluated and compared to build intuition before moving to **Week 2: Deep Learning + Transformers + BERT**.

---

# 📂 Project Structure

```
├── imdb_sentiment.ipynb       # Full project notebook
├── Untitled.ipynb             # Temporary notebook (ignored)
└── .ipynb_checkpoints/        # Jupyter checkpoints
```

> 🔹 Dataset is **not included** due to Kaggle Terms of Service.  
> You can download it manually from the link below.

---

# 📥 Dataset

IMDB Movie Review Dataset (50,000 labeled reviews):  
📎 https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

# 🧹 Text Preprocessing Pipeline

The following steps were applied:

1. Remove HTML tags  
2. Convert to lowercase  
3. Remove punctuation and numbers  
4. Tokenization  
5. Stopword removal  
6. Lemmatization  
7. Join tokens back into cleaned text

This clean text is used for all models.

---

# 🧪 Models Implemented

## 1️⃣ **TF-IDF + Logistic Regression**
- Max features: 10,000  
- Captures important words using sparse representations  

### ✅ Results:
- **Accuracy:** 89.45%  
- Balanced precision, recall, and F1-score  

TF-IDF performs strongly because it captures **key sentiment words** effectively.

---

## 2️⃣ **Word2Vec (Trained from Scratch) + Logistic Regression**

### Word2Vec Configuration:
- vector_size = 100  
- window = 5  
- min_count = 2  
- sg = 1 (Skip-Gram)  

Each review is represented as an **average of its word embeddings**.

### ✅ Results:
- **Accuracy:** 87.35%

Word2Vec captures semantic relationships but loses some sentence structure when averaged.

---

# 📊 Results Summary

| Model                               | Accuracy |
|-------------------------------------|----------|
| **TF-IDF + Logistic Regression**     | **89.45%** |
| **Avg Word2Vec + Logistic Regression** | **87.35%** |

TF-IDF remains the strongest classic baseline, while Word2Vec introduces semantic understanding.

---

# 🚀 Next Steps (Week 2)

In Week 2, I will implement:

- RNN, LSTM, GRU Models  
- Attention Mechanism  
- Transformer Intuition (Q/K/V, Multi-Head Attention, Positional Encoding)  
- **BERT Fine-Tuning for Text Classification**  

This project will evolve into a complete NLP → LLM learning repository.

---

# 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Gensim  
- Jupyter Notebook  

---

# 🙌 Acknowledgements

Dataset: Kaggle IMDB 50K Movie Reviews  
Tools: Scikit-learn, NLTK, Gensim

---

If you like this project or find it helpful, feel free to ⭐ star the repo!
