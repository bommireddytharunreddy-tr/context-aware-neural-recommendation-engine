from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def pipeline_start():
    print("=== RECOMMENDATION PIPELINE STARTED ===")


def update_embeddings():
    print("=== ITEM EMBEDDING UPDATE STEP ===")


def update_redis():
    print("=== REDIS RECOMMENDATION UPDATE STEP ===")


with DAG(
    dag_id="recommendation_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["recommendation", "hm"],
) as dag:

    start_pipeline = PythonOperator(
        task_id="start_pipeline",
        python_callable=pipeline_start,
    )

    embedding_update = PythonOperator(
        task_id="update_embeddings",
        python_callable=update_embeddings,
    )

    redis_update = PythonOperator(
        task_id="update_redis",
        python_callable=update_redis,
    )

    start_pipeline >> embedding_update >> redis_update