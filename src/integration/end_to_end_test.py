import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_DIR = PROJECT_ROOT / "src" / "api"
RETRIEVAL_DIR = PROJECT_ROOT / "src" / "retrieval"

sys.path.append(str(API_DIR))
sys.path.append(str(RETRIEVAL_DIR))

from redis_store import create_redis_client, store_recommendations, get_recommendations


def run_end_to_end_test():
    print("\n=== END-TO-END RECOMMENDATION TEST ===")

    user_id = 5

    recommended_items = [
        43146,
        66187,
        99298,
        28575,
        65496,
        51177,
        7099,
        54865,
        70320,
        73957,
    ]

    print("\n1. Creating Redis connection...")
    redis_client = create_redis_client()
    print("Redis connection created.")

    print("\n2. Storing recommendations...")
    key = store_recommendations(
        redis_client,
        user_id,
        recommended_items
    )
    print("Redis key:", key)

    print("\n3. Retrieving recommendations...")
    result = get_recommendations(
        redis_client,
        user_id
    )

    if result is None:
        print("ERROR: No recommendations found.")
        return False

    print("Retrieved recommendations:")
    print(result)

    print("\n4. Validating recommendation count...")

    if len(result["items"]) != 10:
        print("ERROR: Expected 10 recommendations.")
        return False

    print("Recommendation count: 10")

    print("\n=== END-TO-END TEST PASSED ===")

    return True


if __name__ == "__main__":
    success = run_end_to_end_test()

    if not success:
        raise SystemExit(1)