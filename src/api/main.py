
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.user_to_item_retriever import UserToItemRetriever


app = FastAPI(
    title="Context-Aware Neural Recommendation Engine",
    description="H&M article recommendations using user embeddings and FAISS retrieval",
    version="1.1.0",
)

# Load the encoder and item index once when the API process starts.
recommender = UserToItemRetriever()


class RecommendationItem(BaseModel):
    article_id: int
    item_index: int
    similarity: float


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: list[RecommendationItem]


@app.get("/")
def root():
    return {
        "message": "Recommendation API is running",
        "status": "ready",
    }


@app.get(
    "/recommendations/{customer_id}",
    response_model=RecommendationResponse,
)
def recommendations(
    customer_id: str,
    top_k: int = Query(default=10, ge=1, le=100),
):
    customer_id = customer_id.strip()

    if not customer_id:
        raise HTTPException(
            status_code=400,
            detail="Customer ID cannot be empty",
        )

    try:
        results = recommender.recommend(
            customer_id=customer_id,
            top_k=top_k,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Customer ID was not found in the user-context lookup",
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No recommendations could be generated for this customer",
        )

    return RecommendationResponse(
        customer_id=customer_id,
        recommendations=[
            RecommendationItem(**item)
            for item in results
        ],
    )
