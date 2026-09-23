from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

API_DIR = Path(__file__).resolve().parent
sys.path.append(str(API_DIR))

from redis_store import (
    create_redis_client,
    get_recommendations
)


app = FastAPI(
    title="Context-Aware Neural Recommendation Engine",
    description="Recommendation API using Redis",
    version="1.0.0"
)


redis_client = create_redis_client()


class RecommendationItem(BaseModel):
    item_id: int
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]


@app.get("/")
def root():
    return {
        "message": "Recommendation API is running"
    }


@app.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse
)
def recommendations(user_id: int):

    result = get_recommendations(
        redis_client,
        user_id
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found for this user"
        )

    recommendation_items = [
        RecommendationItem(
            item_id=item_id,
            score=1.0
        )
        for item_id in result["items"]
    ]

    return RecommendationResponse(
        user_id=result["user_id"],
        recommendations=recommendation_items
    )