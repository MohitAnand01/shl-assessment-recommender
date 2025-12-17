from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List

from embedder import Embedder
from query_processor import QueryProcessor
from recommender import Recommender


app = FastAPI(title="SHL Assessment Recommendation API")

# Global singletons
embedder: Embedder
query_processor: QueryProcessor
recommender: Recommender


class RecommendRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup_event() -> None:
    global embedder, query_processor, recommender

    # Load index and metadata
    embedder = Embedder()
    embedder.load_index()

    query_processor = QueryProcessor()
    recommender = Recommender(embedder, query_processor)

    print("Application startup complete.")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/recommend")
def recommend(request: RecommendRequest) -> Dict[str, List[Dict[str, Any]]]:
    """
    Required API contract:
    {
      "recommended_assessments": [
        {
          "url": "...",
          "name": "...",
          "adaptive_support": "Yes/No",
          "description": "...",
          "duration": 0,
          "remote_support": "Yes/No",
          "test_type": [...]
        }
      ]
    }
    """
    results = recommender.recommend(request.query, top_k=10)
    return {"recommended_assessments": results}


if __name__ == "__main__":
    import uvicorn

    # Run the app when we do: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)