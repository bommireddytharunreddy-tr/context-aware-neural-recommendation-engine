import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_DIR = PROJECT_ROOT / "src" / "api"

sys.path.append(str(API_DIR))

from redis_store import (
    create_redis_client,
    get_recommendations,
)


def check_project_files():
    required_files = [
        "src/data/spark_data_loader.py",
        "src/features/feature_engineering.py",
        "src/model/user_tower.py",
        "src/model/item_tower.py",
        "src/model/train_two_tower.py",
        "src/model/evaluate_model.py",
        "src/model/generate_item_embeddings.py",
        "src/retrieval/ann_retrieval.py",
        "src/api/redis_store.py",
        "src/api/main.py",
        "src/integration/end_to_end_test.py",
        "airflow/dags/recommendation_pipeline.py",
    ]

    print("\n=== PROJECT FILE CHECK ===")

    all_present = True

    for file_path in required_files:
        path = PROJECT_ROOT / file_path

        if path.exists():
            print(f"[PASS] {file_path}")
        else:
            print(f"[FAIL] {file_path}")
            all_present = False

    return all_present


def check_redis():
    print("\n=== REDIS CHECK ===")

    try:
        redis_client = create_redis_client()

        if not redis_client.ping():
            print("[FAIL] Redis ping failed")
            return False

        print("[PASS] Redis connection")

        result = get_recommendations(
            redis_client,
            5
        )

        if result is None:
            print("[FAIL] No recommendations found for user 5")
            return False

        print("[PASS] Recommendation data found")

        item_count = len(result["items"])

        if item_count != 10:
            print(
                f"[FAIL] Expected 10 recommendations, found {item_count}"
            )
            return False

        print("[PASS] Recommendation count: 10")

        return True

    except Exception as error:
        print(f"[FAIL] Redis check failed: {error}")
        return False


def main():
    print("\n===================================")
    print(" SYSTEM HEALTH CHECK")
    print(" Context-Aware Recommendation Engine")
    print("===================================")

    files_ok = check_project_files()
    redis_ok = check_redis()

    print("\n=== FINAL RESULT ===")

    if files_ok and redis_ok:
        print("SYSTEM HEALTH CHECK PASSED")
        return 0

    print("SYSTEM HEALTH CHECK FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())