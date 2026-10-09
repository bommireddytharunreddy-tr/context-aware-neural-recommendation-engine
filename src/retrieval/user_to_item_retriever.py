
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.trained_user_encoder import TrainedUserEncoder
from src.retrieval.trained_ann_retriever import TrainedANNRetriever


class UserToItemRetriever:
    """Generate a user embedding and retrieve matching H&M articles."""

    def __init__(self):
        self.user_encoder = TrainedUserEncoder()
        self.item_retriever = TrainedANNRetriever()

    def recommend(self, customer_id, top_k=10):
        """Return the top-K retrieved articles for an H&M customer."""

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        user_embedding = self.user_encoder.encode(str(customer_id))
        user_embedding = np.asarray(user_embedding, dtype=np.float32).reshape(-1)

        if user_embedding.shape != (64,):
            raise ValueError(
                f"Expected a 64-dimensional user embedding, "
                f"received shape {user_embedding.shape}"
            )

        if not np.isfinite(user_embedding).all():
            raise ValueError("User embedding contains non-finite values")

        norm = float(np.linalg.norm(user_embedding))
        if not np.isclose(norm, 1.0, atol=1e-4):
            raise ValueError(
                f"Expected a normalized user embedding; received L2 norm {norm}"
            )

        results = self.item_retriever.retrieve(
            query_embedding=user_embedding,
            top_k=top_k,
        )

        return [
            {
                "article_id": int(result["article_id"]),
                "item_index": int(result["item_index"]),
                "similarity": float(result["similarity"]),
            }
            for result in results
        ]


def main():
    recommender = UserToItemRetriever()

    # Use a known customer ID from the saved user-context lookup.
    sample = np.load(
        PROJECT_ROOT / "trained_models" / "user_context.npz",
        allow_pickle=False,
    )
    customer_id = str(sample["customer_ids"][5])
    sample.close()

    recommendations = recommender.recommend(customer_id, top_k=10)

    print("=== USER-TO-ITEM RETRIEVAL TEST ===")
    print(f"Customer ID: {customer_id}")
    print(f"Recommendations returned: {len(recommendations)}")

    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. article_id={item['article_id']}, "
            f"item_index={item['item_index']}, "
            f"similarity={item['similarity']:.6f}"
        )

    if not recommendations:
        raise RuntimeError("The retriever returned no recommendations")

    if len({item["article_id"] for item in recommendations}) != len(recommendations):
        raise RuntimeError("Duplicate article IDs found in recommendations")

    if not all(np.isfinite(item["similarity"]) for item in recommendations):
        raise RuntimeError("Non-finite similarity score found")

    print("=== USER-TO-ITEM RETRIEVAL PASSED ===")


if __name__ == "__main__":
    main()