import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException


# ---------------------------------------------------------
# Add API directory to Python import path
# ---------------------------------------------------------

API_DIR = Path(__file__).resolve().parent

sys.path.append(str(API_DIR))


from redis_store import (
    create_redis_client,
    get_recommendations
)


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="Context-Aware Neural Recommendation Engine",
    description="Recommendation API using Redis",
    version="1.0.0"
)


# ---------------------------------------------------------
# Redis Connection
# ---------------------------------------------------------

redis_client = create_redis_client()


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Recommendation API is running"
    }


# ---------------------------------------------------------
# Recommendation Endpoint
# ---------------------------------------------------------

@app.get("/recommendations/{user_id}")
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

    return result