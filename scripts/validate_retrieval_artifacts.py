from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "trained_models"


def main():
    user_data = np.load(MODEL_DIR / "user_context.npz")
    item_data = np.load(MODEL_DIR / "item_embeddings.npz")

    try:
        customer_ids = user_data["customer_ids"]
        user_indices = user_data["user_indices"]

        article_ids = item_data["article_ids"]
        item_indices = item_data["item_indices"]
        embeddings = item_data["embeddings"]

        ann_article_ids = np.load(
            MODEL_DIR / "ann_article_ids.npy"
        )
        index = faiss.read_index(
            str(MODEL_DIR / "item_embeddings.faiss")
        )

        checks = {
            "User indices are sequential": np.array_equal(
                user_indices, np.arange(len(user_indices))
            ),
            "Customer IDs are sorted": np.array_equal(
                customer_ids, np.sort(customer_ids)
            ),
            "Item indices are sequential": np.array_equal(
                item_indices, np.arange(len(item_indices))
            ),
            "Article IDs are sorted": np.array_equal(
                article_ids, np.sort(article_ids)
            ),
            "ANN article IDs match saved IDs": np.array_equal(
                article_ids, ann_article_ids
            ),
            "Embedding count matches article count": (
                len(embeddings) == len(article_ids)
            ),
            "FAISS vector count matches": (
                index.ntotal == len(article_ids)
            ),
            "FAISS dimension matches": (
                index.d == embeddings.shape[1]
            ),
            "FAISS vectors match saved embeddings": np.allclose(
                index.reconstruct_n(0, index.ntotal),
                embeddings,
                atol=1e-6,
            ),
            "Embeddings are finite": np.isfinite(embeddings).all(),
        }

        print("=== RETRIEVAL ARTIFACT VALIDATION ===")

        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")

        failed = [name for name, passed in checks.items() if not passed]

        if failed:
            raise SystemExit(
                f"\nValidation failed: {len(failed)} check(s)."
            )

        print("\nAll retrieval artifact checks passed.")
        print(f"Customers: {len(customer_ids):,}")
        print(f"Articles: {len(article_ids):,}")
        print(f"Embedding shape: {embeddings.shape}")

    finally:
        user_data.close()
        item_data.close()


if __name__ == "__main__":
    main()
