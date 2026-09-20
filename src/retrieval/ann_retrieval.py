import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import faiss


# ---------------------------------------------------------
# Add the model directory to Python's import path
# ---------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.append(str(MODEL_DIR))


from train_two_tower import TwoTowerModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_ITEMS = 104_547
EMBEDDING_DIM = 64
TOP_K = 10


# ---------------------------------------------------------
# Generate Item Embeddings
# ---------------------------------------------------------

def generate_item_embeddings(model):

    item_indices = tf.range(
        NUM_ITEMS,
        dtype=tf.int32
    )

    embeddings = model.item_tower(item_indices)

    return embeddings.numpy().astype("float32")


# ---------------------------------------------------------
# Build FAISS ANN Index
# ---------------------------------------------------------

def build_faiss_index(item_embeddings):

    # Inner Product is suitable because
    # our embeddings are L2-normalized.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    index.add(item_embeddings)

    return index


# ---------------------------------------------------------
# Retrieve Similar Items
# ---------------------------------------------------------

def retrieve_items(index, user_embedding):

    user_embedding = user_embedding.numpy().astype("float32")

    scores, item_indices = index.search(
        user_embedding,
        TOP_K
    )

    return scores, item_indices


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\n=== ANN RETRIEVAL ===")

    # Create Two-Tower model
    model = TwoTowerModel()

    # -----------------------------------------------------
    # Build the model before accessing the towers
    # -----------------------------------------------------

    sample_users = tf.constant(
        [0, 1],
        dtype=tf.int32
    )

    sample_items = tf.constant(
        [0, 1],
        dtype=tf.int32
    )

    model(
        (sample_users, sample_items)
    )

    # -----------------------------------------------------
    # Generate Item Embeddings
    # -----------------------------------------------------

    print("\nGenerating item embeddings...")

    item_embeddings = generate_item_embeddings(
        model
    )

    print(
        "Item embedding shape:",
        item_embeddings.shape
    )

    # -----------------------------------------------------
    # Build FAISS Index
    # -----------------------------------------------------

    print("\nBuilding FAISS index...")

    index = build_faiss_index(
        item_embeddings
    )

    print(
        "FAISS index size:",
        index.ntotal
    )

    # -----------------------------------------------------
    # Generate User Embedding
    # -----------------------------------------------------

    print("\nGenerating sample user embedding...")

    user_index = tf.constant(
        [5],
        dtype=tf.int32
    )

    user_embedding = model.user_tower(
        user_index
    )

    print(
        "User embedding shape:",
        user_embedding.shape
    )

    # -----------------------------------------------------
    # ANN Search
    # -----------------------------------------------------

    print("\nSearching for Top-10 items...")

    scores, item_indices = retrieve_items(
        index,
        user_embedding
    )

    # -----------------------------------------------------
    # Display Recommendations
    # -----------------------------------------------------

    print("\n=== RECOMMENDATIONS ===")

    for rank, (item_index, score) in enumerate(
        zip(
            item_indices[0],
            scores[0]
        ),
        start=1
    ):

        print(
            f"{rank}. "
            f"Item index: {item_index}, "
            f"similarity: {score:.4f}"
        )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print("\n=== ANN RETRIEVAL COMPLETED ===")


# ---------------------------------------------------------
# Program Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()