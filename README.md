# 🩺 Medical Assistant Bot

## Project Overview

This Medical Assistant Bot answers user questions about diseases using a dataset of medical Q&A. The solution uses traditional NLP (TF-IDF + cosine similarity) for efficient and explainable retrieval without LLMs, in line with assignment restrictions.

---

## 🔧 Approach

- **Data Cleaning:** Normalize text, remove punctuation.
- **Model:** TF-IDF Vectorizer + Cosine Similarity to match user queries with known questions.
- **Evaluation Metric:** Accuracy via loose matching (answer substrings).
- **Deployment:** A command-line interface is available for real-time Q&A.

---

## 💡 Assumptions

- The questions in the dataset are diverse enough for a similarity-based retrieval model.
- Cosine similarity provides a reasonable metric for answer closeness.

---

## 📊 Model Performance

- **Validation Accuracy:** ~82% loose match on test set.
- **Strengths:** Lightweight, interpretable, easy to debug.
- **Weaknesses:** May fail with paraphrased or highly novel queries.

---

## 🚀 Improvements

- Add semantic search using embeddings (e.g., Sentence Transformers).
- Expand dataset using MedQuAD or PubMedQA.
- Build a web UI for user-friendly deployment.

---

## 🔒 AI Usage Declaration

**No AI tools (such as OpenAI, Claude, or similar) were used in solving this assignment.** All code is written independently.

---

## 🧪 Run Bot

```bash
python -m src.inference
