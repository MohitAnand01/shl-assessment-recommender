# SHL Assessment Recommendation System

A semantic recommendation engine that recommends SHL **individual test solutions** from a job description text, natural language query, or JD URL.

---

## 1. Features

- Crawls SHL’s public catalog and builds an **assessment dataset** (389 individual assessments).
- Uses **Sentence-Transformers (all-MiniLM-L6-v2)** for semantic embeddings.
- Stores vectors in **FAISS** for fast nearest‑neighbour search.
- Applies a **re‑ranking layer** using test‑type and skill matches.
- Exposes a **FastAPI** `/recommend` endpoint.
- Ships with a simple **HTML/JS frontend** (deployed on Vercel).

---

## 2. Live Demo

- **Frontend (Vercel):**  
  [https://frontend-tau-two-64.vercel.app](https://frontend-tau-two-64.vercel.app)

- **Backend API (Render):**  
  [https://shl-assessment-recommender-bttc.onrender.com](https://shl-assessment-recommender-bttc.onrender.com)

**How to use:**

1. Open the frontend.
2. Paste a job description or query.
3. (Optionally) change number of recommendations.
4. Click **Get Recommendations**.

---

## 3. Repository Structure

```text
.
├── backend
│   ├── assessments.json        # Crawled assessment catalog (389 assessments)
│   ├── crawler.py              # SHL catalog crawler
│   ├── embedder.py             # Embedding + FAISS index builder
│   ├── query_processor.py      # URL handling, skill & type extraction
│   ├── recommender.py          # Retrieval + re-ranking logic
│   └── main.py                 # FastAPI app (/health, /recommend)
│
├── evaluation
│   ├── evaluate.py             # Mean Recall@K on labeled train data
│   └── generate_predictions.py # predictions.csv for unlabeled test set
│
├── frontend
│   └── index.html              # Static HTML/CSS/JS UI
│
├── approach_document.pdf       # 2‑page approach write-up
├── README.md                   # This file
└── runtime.txt                 # Python runtime hint for deployment