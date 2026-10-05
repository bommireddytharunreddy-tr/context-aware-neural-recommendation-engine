import os
import sys

import faiss
import numpy as np


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "trained_models"
)

EMBEDDINGS_PATH = os.path.join(
    MODEL_DIR,
    "item_embeddings.npz"
)

ANN_INDEX_PATH = os.path.join(
    MODEL_DIR,
    "item_embeddings.faiss"
)

ARTICLE_IDS_PATH = os.path.join(
    MODEL_DIR,
    "ann_article_ids.npy"
)

EXPECTED_ITEMS = 104_547
EMBEDDING_DIM = 64

TOP_K = 10


# ============================================================
# LOAD ITEM EMBEDDINGS
# ============================================================

def load_item_embeddings():

    print("\n=== LOADING ITEM EMBEDDINGS ===")
    print("Path:", EMBEDDINGS_PATH)

    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            "Item embeddings not found: "
            f"{EMBEDDINGS_PATH}"
        )

    data = np.load(
        EMBEDDINGS_PATH
    )

    article_ids = data["article_ids"]
    item_indices = data["item_indices"]
    embeddings = data["embeddings"]

    print(
        "Article IDs shape:",
        article_ids.shape
    )

    print(
        "Item indices shape:",
        item_indices.shape
    )

    print(
        "Embeddings shape:",
        embeddings.shape
    )

    print(
        "Embeddings dtype:",
        embeddings.dtype
    )

    if embeddings.shape != (
        EXPECTED_ITEMS,
        EMBEDDING_DIM
    ):
        raise ValueError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    if len(article_ids) != EXPECTED_ITEMS:
        raise ValueError(
            "Unexpected article ID count: "
            f"{len(article_ids)}"
        )

    if len(item_indices) != EXPECTED_ITEMS:
        raise ValueError(
            "Unexpected item index count: "
            f"{len(item_indices)}"
        )

    if not np.array_equal(
        item_indices,
        np.arange(
            EXPECTED_ITEMS,
            dtype=np.int32
        )
    ):
        raise ValueError(
            "Item indices are not sequential."
        )

    return (
        article_ids,
        item_indices,
        embeddings
    )


# ============================================================
# BUILD ANN INDEX
# ============================================================

def build_ann_index(embeddings):

    print("\n=== BUILDING FAISS ANN INDEX ===")

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    print(
        "Index type: FAISS IndexFlatIP"
    )

    print(
        "Metric: Inner Product"
    )

    print(
        "Vectors:",
        embeddings.shape[0]
    )

    print(
        "Dimension:",
        embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        EMBEDDING_DIM
    )

    index.add(
        embeddings
    )

    print(
        "Indexed vectors:",
        index.ntotal
    )

    if index.ntotal != EXPECTED_ITEMS:
        raise ValueError(
            "Unexpected number of indexed vectors: "
            f"{index.ntotal}"
        )

    return index


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    index,
    article_ids
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    faiss.write_index(
        index,
        ANN_INDEX_PATH
    )

    np.save(
        ARTICLE_IDS_PATH,
        article_ids
    )

    print("\n=== ANN INDEX SAVED ===")
    print(
        "FAISS index:",
        ANN_INDEX_PATH
    )

    print(
        "Article ID mapping:",
        ARTICLE_IDS_PATH
    )


# ============================================================
# TEST RETRIEVAL
# ============================================================

def test_retrieval(
    index,
    article_ids,
    embeddings
):

    print("\n=== TESTING ANN RETRIEVAL ===")

    query_position = 0

    query_vector = (
        embeddings[
            query_position
        ]
        .reshape(1, -1)
        .astype(np.float32)
    )

    scores, positions = index.search(
        query_vector,
        TOP_K
    )

    retrieved_positions = positions[0]
    retrieved_scores = scores[0]

    print(
        "Query item index:",
        query_position
    )

    print(
        "Query article ID:",
        article_ids[
            query_position
        ]
    )

    print(
        f"\nTop {TOP_K} nearest items:"
    )

    for rank, (
        position,
        score
    ) in enumerate(
        zip(
            retrieved_positions,
            retrieved_scores
        ),
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"item_index={position}, "
            f"article_id={article_ids[position]}, "
            f"similarity={score:.6f}"
        )

    if positions.shape != (
        1,
        TOP_K
    ):
        raise ValueError(
            "Unexpected retrieval result shape: "
            f"{positions.shape}"
        )

    if retrieved_positions[0] != query_position:
        raise ValueError(
            "Query item was not retrieved "
            "as its own nearest neighbor."
        )

    if not np.all(
        retrieved_positions >= 0
    ):
        raise ValueError(
            "Invalid ANN position returned."
        )

    print(
        "\nANN retrieval test passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("COMMIT 33 - TRAINED ANN INDEX")
    print("==========================================")

    (
        article_ids,
        item_indices,
        embeddings
    ) = load_item_embeddings()

    index = build_ann_index(
        embeddings
    )

    save_index(
        index,
        article_ids
    )

    test_retrieval(
        index,
        article_ids,
        embeddings
    )

    print(
        "\n=== COMMIT 33 TRAINED ANN "
        "INDEX PASSED ==="
    )


if __name__ == "__main__":
    main()