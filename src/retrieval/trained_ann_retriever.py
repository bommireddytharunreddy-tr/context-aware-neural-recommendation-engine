import os

import faiss
import numpy as np


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        ".."
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "trained_models"
)

ANN_INDEX_PATH = os.path.join(
    MODEL_DIR,
    "item_embeddings.faiss"
)

ARTICLE_IDS_PATH = os.path.join(
    MODEL_DIR,
    "ann_article_ids.npy"
)


# ============================================================
# CONSTANTS
# ============================================================

EMBEDDING_DIM = 64
TOTAL_ITEMS = 104_547
DEFAULT_TOP_K = 10


# ============================================================
# TRAINED ANN RETRIEVER
# ============================================================

class TrainedANNRetriever:

    def __init__(
        self,
        index_path=ANN_INDEX_PATH,
        article_ids_path=ARTICLE_IDS_PATH
    ):

        self.index_path = index_path
        self.article_ids_path = article_ids_path

        self.index = None
        self.article_ids = None

        self._load()


    # ========================================================
    # LOAD INDEX
    # ========================================================

    def _load(self):

        print("\n=== LOADING TRAINED ANN RETRIEVER ===")

        if not os.path.exists(
            self.index_path
        ):
            raise FileNotFoundError(
                "FAISS index not found: "
                f"{self.index_path}"
            )

        if not os.path.exists(
            self.article_ids_path
        ):
            raise FileNotFoundError(
                "Article ID mapping not found: "
                f"{self.article_ids_path}"
            )

        self.index = faiss.read_index(
            self.index_path
        )

        self.article_ids = np.load(
            self.article_ids_path
        )

        print(
            "FAISS index vectors:",
            self.index.ntotal
        )

        print(
            "FAISS index dimension:",
            self.index.d
        )

        print(
            "Article IDs:",
            len(self.article_ids)
        )

        if self.index.ntotal != TOTAL_ITEMS:
            raise ValueError(
                "Unexpected FAISS index size: "
                f"{self.index.ntotal}"
            )

        if self.index.d != EMBEDDING_DIM:
            raise ValueError(
                "Unexpected embedding dimension: "
                f"{self.index.d}"
            )

        if len(self.article_ids) != TOTAL_ITEMS:
            raise ValueError(
                "Unexpected article ID count: "
                f"{len(self.article_ids)}"
            )

        print(
            "Trained ANN retriever loaded successfully."
        )


    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query_embedding,
        top_k=DEFAULT_TOP_K
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = (
                query_embedding.reshape(
                    1,
                    -1
                )
            )

        if query_embedding.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                "Unexpected query embedding dimension: "
                f"{query_embedding.shape}"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        top_k = min(
            top_k,
            self.index.ntotal
        )

        scores, positions = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for position, score in zip(
            positions[0],
            scores[0]
        ):

            if position < 0:
                continue

            results.append(
                {
                    "article_id": int(
                        self.article_ids[position]
                    ),
                    "item_index": int(
                        position
                    ),
                    "similarity": float(
                        score
                    )
                }
            )

        return results


# ============================================================
# STANDALONE VALIDATION
# ============================================================

def main():

    print("\n==========================================")
    print("COMMIT 34 - TRAINED ANN RETRIEVER")
    print("==========================================")

    retriever = TrainedANNRetriever()

    # Load the same trained embeddings used
    # to construct the FAISS index.
    embeddings_path = os.path.join(
        MODEL_DIR,
        "item_embeddings.npz"
    )

    if not os.path.exists(
        embeddings_path
    ):
        raise FileNotFoundError(
            "Item embeddings not found: "
            f"{embeddings_path}"
        )

    data = np.load(
        embeddings_path
    )

    embeddings = data[
        "embeddings"
    ]

    print(
        "\nEmbedding matrix:",
        embeddings.shape
    )

    # Use item 0 as a deterministic query.
    query_embedding = embeddings[0]

    results = retriever.retrieve(
        query_embedding,
        top_k=10
    )

    print(
        "\n=== TOP 10 RETRIECOMMENDATIONS ==="
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"article_id={result['article_id']}, "
            f"item_index={result['item_index']}, "
            f"similarity={result['similarity']:.6f}"
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(results) != 10:
        raise ValueError(
            "Expected exactly 10 recommendations."
        )

    expected_article_id = int(
        data["article_ids"][0]
    )

    if results[0]["article_id"] != (
        expected_article_id
    ):
        raise ValueError(
            "Query item was not returned "
            "as the top result."
        )

    if results[0]["item_index"] != 0:
        raise ValueError(
            "Unexpected top result item index."
        )

    if results[0]["similarity"] < 0.999:
        raise ValueError(
            "Unexpected self-similarity score: "
            f"{results[0]['similarity']}"
        )

    print(
        "\nRetriever validation passed."
    )

    print(
        "\n=== COMMIT 34 TRAINED ANN "
        "RETRIEVER PASSED ==="
    )


if __name__ == "__main__":
    main()