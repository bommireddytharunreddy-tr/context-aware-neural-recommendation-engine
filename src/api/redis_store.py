import json

import redis


REDIS_HOST = "localhost"
REDIS_PORT = 6379

DEFAULT_TTL = 3600


def create_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True
    )


def store_recommendations(
    redis_client,
    user_id,
    item_indices,
    ttl=DEFAULT_TTL
):
    key = f"recommendations:user_{user_id}"

    recommendations = {
        "user_id": user_id,
        "items": item_indices
    }

    redis_client.set(
        key,
        json.dumps(recommendations),
        ex=ttl
    )

    return key


def get_recommendations(redis_client, user_id):
    key = f"recommendations:user_{user_id}"

    data = redis_client.get(key)

    if data is None:
        return None

    return json.loads(data)


def main():

    print("\n=== REDIS RECOMMENDATION STORE ===")

    redis_client = create_redis_client()

    # Test user and recommendations
    user_id = 5

    item_indices = [
        43146,
        66187,
        99298,
        28575,
        65496,
        51177,
        7099,
        54865,
        70320,
        73957
    ]

    print("\nStoring recommendations...")

    key = store_recommendations(
        redis_client,
        user_id,
        item_indices
    )

    print("Redis key:", key)

    print("\nRetrieving recommendations...")

    recommendations = get_recommendations(
        redis_client,
        user_id
    )

    print("Retrieved data:")
    print(recommendations)

    if recommendations is not None:
        print("\n=== REDIS TEST PASSED ===")
    else:
        print("\n=== REDIS TEST FAILED ===")


if __name__ == "__main__":
    main()