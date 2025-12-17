import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embedder import Embedder
from query_processor import QueryProcessor
from recommender import Recommender

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends SHL assessments based on job descriptions or queries",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
embedder: Optional[Embedder] = None
query_processor: Optional[QueryProcessor] = None
recommender: Optional[Recommender] = None


@app.on_event("startup")
def startup_event():
    """
    Load models and FAISS index on startup.
    Embedder now loads the index automatically in __init__.
    """
    global embedder, query_processor, recommender
    
    print("Starting up SHL Assessment Recommendation API...")
    
    # Embedder loads FAISS index & metadata in __init__
    embedder = Embedder()
    
    query_processor = QueryProcessor()
    recommender = Recommender(embedder, query_processor)
    
    print("Startup complete. API is ready.")


@app.on_event("shutdown")
def shutdown_event():
    """
    Cleanup on shutdown if needed.
    """
    print("Shutting down SHL Assessment Recommendation API...")


# Request/Response models
class RecommendationRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I need a cognitive ability test for software engineers",
                "top_k": 10
            }
        }


class Assessment(BaseModel):
    name: str
    url: str
    description: str
    test_type: List[str]
    duration: int
    adaptive_support: str
    remote_support: str
    score: Optional[float] = None


class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[Assessment]
    count: int


# API Endpoints
@app.get("/")
def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    if embedder is None or recommender is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "embedder_loaded": embedder.index is not None,
        "num_assessments": len(embedder.metadata) if embedder.metadata else 0
    }


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    """
    Get assessment recommendations based on a query.
    
    The query can be:
    - A natural language description of requirements
    - A job description text
    - A URL containing a job description
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        # Get recommendations
        recommendations = recommender.recommend(
            query=request.query,
            top_k=request.top_k
        )
        
        # Convert to response format
        assessments = [
            Assessment(
                name=rec.get("name", ""),
                url=rec.get("url", ""),
                description=rec.get("description", ""),
                test_type=rec.get("test_type", []),
                duration=rec.get("duration", 0),
                adaptive_support=rec.get("adaptive_support", "No"),
                remote_support=rec.get("remote_support", "No"),
                score=rec.get("score")
            )
            for rec in recommendations
        ]
        
        return RecommendationResponse(
            query=request.query,
            recommendations=assessments,
            count=len(assessments)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)