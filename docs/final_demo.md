# Context-Aware Neural Recommendation Engine

## 1. Project Overview

The Context-Aware Neural Recommendation Engine is a recommendation system built using the H&M Personalized Fashion Recommendations dataset.

The project processes customer and article information, creates recommendation features, generates item embeddings, performs approximate nearest-neighbor retrieval, stores recommendations in Redis, exposes them through FastAPI, and orchestrates pipeline tasks using Airflow.

---

## 2. System Architecture

```text
H&M Dataset
     |
     v
PySpark Data Loading
     |
     v
Feature Engineering
     |
     v
Training Interaction Preparation
     |
     v
User / Item ID Mapping
     |
     v
Two-Tower Model
     |
     v
Item Embeddings
     |
     v
FAISS ANN Retrieval
     |
     v
Redis Recommendation Store
     |
     v
FastAPI Recommendation API
     ^
     |
Airflow Pipeline Orchestration