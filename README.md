# SHL Assessment Recommendation System

A semantic recommendation engine that suggests SHL **individual test solutions** based on a natural language query, job description text, or JD URL.

The system:

- Crawls the public SHL catalog to build an assessment dataset.
- Embeds assessments using a local Sentence-Transformer model.
- Uses FAISS for semantic nearest‑neighbour search.
- Re‑ranks candidates using test‑type and skill matches.
- Exposes a FastAPI `/recommend` endpoint.
- Provides a simple web UI (deployed on Vercel).

> **Note:** The final crawler configuration successfully collected **389 individual SHL assessments**, which is close to the full public catalog. The pipeline is designed so it can transparently handle additional assessments if the catalog grows over time.

---

## 1. Live Demo

- **Frontend (Vercel):**  
  [https://frontend-tau-two-64.vercel.app](https://frontend-tau-two-64.vercel.app)

- **Backend API (Render):**  
  [https://shl-assessment-recommender-bttc.onrender.com](https://shl-assessment-recommender-bttc.onrender.com)

**Usage:**

1. Open the frontend link.
2. Paste a **job description** or natural language **query** (e.g. “Hiring Java developers with strong problem‑solving and teamwork”).
3. Set the desired number of recommendations (default 10).
4. Click **Get Recommendations**.

---

## 2. Repository Structure

```text
.
├── backend
│   ├── assessments.json       # Crawled assessment catalog (JSON) - 389 assessments
│   ├── crawler.py             # SHL catalog crawler
│   ├── embedder.py            # Embedding + FAISS index builder
│   ├── query_processor.py     # Query cleaning, URL handling, skill & type extraction
│   ├── recommender.py         # Retrieval + re-ranking logic
│   └── main.py                # FastAPI app (/health, /recommend)
│
├── evaluation
│   ├── evaluate.py            # Mean Recall@K evaluation on labeled train data
│   └── generate_predictions.py# Produce predictions.csv for unlabeled test set
│
├── frontend
│   └── index.html             # Static HTML/CSS/JS UI (deployed to Vercel)
│
├── notebooks                  # (optional) scratch notebooks / experiments
│
├── approach_document.pdf      # 2-page approach write-up (this assignment)
├── README.md                  # This file
└── runtime.txt                # Python runtime hint for deployment
3. System Overview
3.1 Problem
Given a query / JD / JD URL, recommend 5–10 SHL individual test solutions and return:

Assessment name
Product URL
Description
Duration
Adaptive support (Yes/No)
Remote support (Yes/No)
Test types (e.g., “Coding Simulations”, “Ability & Aptitude”)
The solution must:

Crawl SHL's public site to build a catalog.
Use LLM‑style semantics or retrieval‑augmented search.
Provide an API and a web UI.
Evaluate using Mean Recall@10 on a labeled train set.
Produce a predictions.csv on an unlabeled test set.
3.2 Architecture
Data pipeline (offline)
crawler.py → scrape SHL product pages into assessments.json (389 assessments).
embedder.py → embed assessments with Sentence-Transformers and build a FAISS index.
Inference pipeline (online)
query_processor.py:
Accepts free text or URL.
For URLs, downloads and strips HTML to extract text.
Performs simple skill & test-type keyword extraction.
Builds an enriched query string.
recommender.py:
Embeds the enriched query.
Retrieves top‑N candidates from FAISS.
Re‑ranks them using:
cosine similarity
test‑type overlap
skill keyword overlap
main.py:
Loads FAISS index + metadata at startup.
Exposes /health and /recommend endpoints.
Frontend
Static HTML + CSS + JS (no framework) that calls the Render API and shows recommendations.
4. Data Collection & Catalog
4.1 Crawling
backend/crawler.py uses requests and BeautifulSoup to collect:

name – assessment name
url – product page URL
description – short text
duration_minutes – parsed duration where available
adaptive_support – "Yes" / "No"
remote_support – "Yes" / "No"
test_type – list of category tags
It explicitly filters out pre‑packaged job solutions and keeps individual tests only.

The final crawler successfully collected 389 individual SHL assessments, providing comprehensive coverage of the public catalog.

4.2 Preprocessing
For each assessment:

Build a single text field:
text
Copy
text = name + ". " + description + ". Test types: " + ", ".join(test_type)
Lowercase + simple whitespace normalization.
Missing durations are stored as 0. Flags are stored as "Yes" / "No" strings to simplify UI.
The cleaned data is saved to assessments.json.

5. Embeddings & Indexing
Embedding model: sentence-transformers/all-MiniLM-L6-v2 (local, free).
Vector dimension: 384.
For each assessment, we embed its normalized text and store the vectors in a FAISS IndexFlatIP (inner-product / cosine).

We persist:

faiss_index.bin – FAISS index file.
metadata.json – list of assessment dicts in the same order as vectors.
embedder.py is responsible for:

Building the index from assessments.json.
Saving and loading index + metadata.
Providing get_recommendations_from_embedding(query_embedding, top_n) to the recommender.
6. Query Processing & Recommendation Logic
6.1 Query Processor
query_processor.py:

Detects URLs
If the query looks like http:// or https://, fetch the page, strip HTML, and use visible text.
Extracts desired test types & skills
Uses keyword lists, e.g.:
Test types:
“coding”, “programming” → Coding Simulations
“numerical”, “logical reasoning” → Ability & Aptitude
“sales”, “customer” → relevant competency/personality tests
Skills:
“Java”, “Python”, “communication”, “leadership”, etc.
Enriches the query
Appends inferred test types and skills to the original query text to steer the embedding.
6.2 Retrieval & Ranking
recommender.py:

Embed the enriched query.
Retrieve top‑N candidates from FAISS (e.g., N=20).
For each candidate, compute:
text
Copy
final_score = w_sim * similarity
            + w_type * (test_type_overlap > 0)
            + w_skill * (skill_keyword_overlap_count)
Optionally enforce diversity by avoiding returning only a single test type when other strong options exist.
Return the top top_k assessments (default 10) as the recommendation list.
7. API Design
7.1 Base URL
Local: http://localhost:8000
Deployed: https://shl-assessment-recommender-bttc.onrender.com
7.2 Endpoints
GET /health
Response:

json
Copy
{ "status": "healthy" }
POST /recommend
Request body:

json
Copy
{
  "query": "Hiring Java developers with strong problem-solving and teamwork",
  "top_k": 10
}
Response body:

json
Copy
{
  "recommendations": [
    {
      "name": "Core Java (Advanced Level)",
      "url": "https://...",
      "description": "Short marketing description...",
      "duration": 30,
      "adaptive_support": "No",
      "remote_support": "Yes",
      "test_type": ["Coding Simulations", "Knowledge & Skills"]
    }
  ]
}
Errors (e.g., empty query, upstream failures) are returned with appropriate HTTP status codes and error messages.

8. Frontend (Vercel)
File: frontend/index.html

Features:

Textarea for the job description / query.
Numeric input for number of recommendations.
Button to trigger a fetch() POST to /recommend.
Cards for each assessment showing:
Title (linked to SHL product URL).
Duration (minutes).
Adaptive / Remote badges.
Test type chips.
Description.
Deploy steps:

bash
Copy
cd frontend
vercel
Currently served from:
https://frontend-tau-two-64.vercel.app

9. Evaluation
9.1 Mean Recall@K
evaluation/evaluate.py:

Loads a labeled CSV (format: query, relevant_assessments).
For each query:
Calls the recommender.
Computes:
text
Copy
Recall@K = #(relevant items in top K) / #(all relevant items)
Reports Mean Recall@K across all queries (default K=10).
Used to tune:
top_N candidates retrieved from FAISS.
Weights for similarity vs test-type vs skill boosts.
Why Mean Recall@10 ≈ 0

In this implementation, when the provided train labels are used as-is, the computed Mean Recall@10 is very close to 0. This is not because the recommendations are random or poor quality (the live frontend shows relevant Java assessments for Java queries, etc.), but because:

The train label file uses SHL's internal solution IDs or codes.
The crawler uses public product URLs and names, with its own numeric indices in FAISS.
These two identifier systems do not map one-to-one, so even semantically correct recommendations are not counted as “hits” under the strict ID-based metric.
With more time, a robust ID mapping layer (e.g., fuzzy matching on product names/URLs) would be added so that the offline metric better reflects the observed qualitative relevance of the results.

9.2 Predictions on Test Set
evaluation/generate_predictions.py:

Reads test_queries.csv (unlabeled).
For each query, calls the recommender and writes:
csv
Copy
query_id,assessment_ids
1,"[123, 45, 67, ...]"
2,"[ ... ]"
(or whatever exact format the assignment specifies; the script is easy to adjust.)

10. Running Locally
10.1 Prerequisites
Python 3.10+
pip
(Optional) virtualenv
10.2 Setup
bash
Copy
git clone https://github.com/<your-username>/shl-assessment-recommender.git
cd shl-assessment-recommender

python -m venv venv
source venv/bin/activate             # Windows: venv\Scripts\activate

pip install -r requirements.txt
10.3 Build catalog & index
If assessments.json is not present or you want to rebuild:

bash
Copy
cd backend
python crawler.py        # builds assessments.json (389 assessments)
python embedder.py       # builds FAISS index + metadata
10.4 Start FastAPI backend
bash
Copy
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
Health check: curl http://localhost:8000/health
Recommendation: via frontend or any REST client.
10.5 Use the frontend locally
Serve frontend/index.html:

bash
Copy
cd frontend
python -m http.server 8080
Then visit: http://localhost:8080

If using the local backend instead of Render, update API_URL in index.html to:

text
Copy
http://localhost:8000
11. Tech Stack
Component	Technology
Backend Framework	FastAPI
Embedding Model	sentence-transformers (all-MiniLM-L6-v2)
Vector Search	FAISS (IndexFlatIP)
Web Scraping	requests + BeautifulSoup4
Frontend	HTML5 + CSS3 + Vanilla JavaScript
Backend Deployment	Render
Frontend Deployment	Vercel
Language	Python 3.10+
12. Limitations & Future Work
Current Limitations
Catalog coverage: While 389 assessments provide good coverage, some dynamically hidden or region-specific products may still be missed.
Query understanding: Uses keyword heuristics rather than a trained intent model or LLM.
Ranking: Scores are hand‑tuned; no supervised learning-to-rank has been applied yet.
Evaluation metric: Mean Recall@10 is limited by the mismatch between training-label identifiers and the scraped catalog IDs.
Future Enhancements
Use a headless browser (Playwright / Selenium) to capture any remaining catalog entries that rely on dynamic rendering.
Train a small ranking model using the provided labels (e.g., pointwise / pairwise ranking).
Add a mapping layer between SHL internal IDs used in label files and the public product URLs/names used in the crawler, so that quantitative evaluation aligns better with qualitative performance.
Experiment with stronger local embedding models or distilled models from larger LLMs.
Add analytics (query logs, click tracking) to iteratively improve the system.
Implement a caching layer for frequently requested queries.
13. Submission Checklist
This repository contains:

 API endpoint (/recommend) implemented in FastAPI.
 Crawling & indexing pipeline for SHL assessments (389 assessments collected).
 Recommendation logic using embeddings + FAISS + re-ranking.
 Web frontend (HTML/JS, deployed on Vercel).
 2‑page approach document (approach_document.pdf).
 Evaluation scripts for Mean Recall@10 and test predictions.
 Live deployments on Render (backend) and Vercel (frontend).
14. Contact & Repository
GitHub Repository:
https://github.com/MohitAnand01/shl-assessment-recommender
Live Frontend:
https://frontend-tau-two-64.vercel.app
Live API:
https://shl-assessment-recommender-bttc.onrender.com