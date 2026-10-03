import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.train_real_two_tower import (
    NUM_ITEMS,
    EMBEDDING_DIM,
    normalize_item_context,
    ContextAwareTwoTowerModel
)


DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "trained_models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "context_aware_two_tower.weights.h5"
)

OUTPUT_PATH = os.path.join(
    MODEL_DIR,
    "item_embeddings.npz"
)

TRANSACTIONS_PATH = os.path.join(
    DATA_DIR,
    "transactions_train.csv"
)

BATCH_SIZE = 2048
CSV_CHUNK_SIZE = 500_000


# ============================================================
# ITEM CONTEXT
# ============================================================

def load_item_context():

    print("\n=== STREAMING TRANSACTION DATA ===")
    print("Source:", TRANSACTIONS_PATH)
    print("Chunk size:", CSV_CHUNK_SIZE)

    if not os.path.exists(TRANSACTIONS_PATH):
        raise FileNotFoundError(
            f"Transactions file not found: {TRANSACTIONS_PATH}"
        )

    item_stats = {}

    total_rows = 0
    chunk_number = 0

    for chunk in pd.read_csv(
        TRANSACTIONS_PATH,
        usecols=["article_id", "price"],
        chunksize=CSV_CHUNK_SIZE
    ):

        chunk_number += 1
        total_rows += len(chunk)

        chunk["article_id"] = pd.to_numeric(
            chunk["article_id"],
            errors="coerce"
        )

        chunk["price"] = pd.to_numeric(
            chunk["price"],
            errors="coerce"
        )

        chunk = chunk.dropna(
            subset=["article_id", "price"]
        )

        grouped = (
            chunk
            .groupby("article_id")["price"]
            .agg(["count", "sum"])
        )

        for article_id, row in grouped.iterrows():

            article_id = int(article_id)

            if article_id not in item_stats:
                item_stats[article_id] = [
                    int(row["count"]),
                    float(row["sum"])
                ]
            else:
                item_stats[article_id][0] += int(
                    row["count"]
                )
                item_stats[article_id][1] += float(
                    row["sum"]
                )

        print(
            f"Processed chunk {chunk_number} | "
            f"rows: {total_rows:,} | "
            f"unique items: {len(item_stats):,}"
        )

    if len(item_stats) != NUM_ITEMS:
        raise ValueError(
            f"Expected {NUM_ITEMS} items, "
            f"but found {len(item_stats)}"
        )

    article_ids = np.array(
        sorted(item_stats.keys()),
        dtype=np.int64
    )

    raw_context = np.array(
        [
            [
                item_stats[article_id][0],
                (
                    item_stats[article_id][1]
                    / item_stats[article_id][0]
                )
            ]
            for article_id in article_ids
        ],
        dtype=np.float32
    )

    item_indices = np.arange(
        len(article_ids),
        dtype=np.int32
    )

    print("\n=== ITEM CONTEXT READY ===")
    print("Transaction rows processed:", f"{total_rows:,}")
    print("Unique items:", len(article_ids))
    print("Context shape:", raw_context.shape)

    return (
        article_ids,
        item_indices,
        raw_context
    )


# ============================================================
# MODEL
# ============================================================

def build_and_load_model():

    model = ContextAwareTwoTowerModel()

    dummy_inputs = {
        "user_index": tf.constant(
            [0],
            dtype=tf.int32
        ),
        "item_index": tf.constant(
            [0],
            dtype=tf.int32
        ),
        "user_context": tf.constant(
            [[1.0, 1.0]],
            dtype=tf.float32
        ),
        "item_context": tf.constant(
            [[1.0, 1.0]],
            dtype=tf.float32
        )
    }

    model(
        dummy_inputs,
        training=False
    )

    model.load_weights(
        MODEL_PATH
    )

    return model


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_item_embeddings(
    model,
    article_ids,
    item_indices,
    raw_context
):

    normalized_context = normalize_item_context(
        tf.constant(
            raw_context,
            dtype=tf.float32
        )
    )

    all_embeddings = []

    total_items = len(item_indices)

    print("\n=== GENERATING ITEM EMBEDDINGS ===")
    print("Total items:", total_items)
    print("Embedding dimension:", EMBEDDING_DIM)
    print("Batch size:", BATCH_SIZE)

    for start in range(
        0,
        total_items,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total_items
        )

        batch_indices = tf.constant(
            item_indices[start:end],
            dtype=tf.int32
        )

        batch_context = normalized_context[
            start:end
        ]

        batch_embeddings = model.item_tower(
            batch_indices,
            batch_context
        )

        all_embeddings.append(
            batch_embeddings.numpy()
        )

        print(
            f"Processed {end:,}/{total_items:,} items"
        )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    return embeddings


# ============================================================
# SAVE
# ============================================================

def save_embeddings(
    article_ids,
    item_indices,
    embeddings
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        article_ids=article_ids,
        item_indices=item_indices,
        embeddings=embeddings
    )

    print("\n=== ITEM EMBEDDINGS SAVED ===")
    print("Path:", OUTPUT_PATH)
    print("Shape:", embeddings.shape)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("COMMIT 32 - ITEM EMBEDDING GENERATION")
    print("==========================================")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model weights not found: {MODEL_PATH}"
        )

    print("\n=== LOADING TRAINED MODEL ===")

    model = build_and_load_model()

    print("Trained model loaded successfully.")

    article_ids, item_indices, raw_context = (
        load_item_context()
    )

    embeddings = generate_item_embeddings(
        model,
        article_ids,
        item_indices,
        raw_context
    )

    print("\n=== VALIDATING EMBEDDINGS ===")

    print(
        "Article IDs shape:",
        article_ids.shape
    )

    print(
        "Item indices shape:",
        item_indices.shape
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Embedding dtype:",
        embeddings.dtype
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    print(
        "Minimum embedding norm:",
        norms.min()
    )

    print(
        "Maximum embedding norm:",
        norms.max()
    )

    if embeddings.shape != (
        NUM_ITEMS,
        EMBEDDING_DIM
    ):
        raise ValueError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5
    ):
        raise ValueError(
            "Embeddings are not L2-normalized."
        )

    save_embeddings(
        article_ids,
        item_indices,
        embeddings
    )

    print(
        "\n=== COMMIT 32 ITEM EMBEDDING "
        "GENERATION PASSED ==="
    )


if __name__ == "__main__":
    main()
