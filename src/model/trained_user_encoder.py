import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# MODEL IMPORTS
# ============================================================

from src.model.train_real_two_tower import (
    ContextAwareTwoTowerModel
)


# ============================================================
# CONFIGURATION
# ============================================================

CONTEXT_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_models",
    "user_context.npz"
)

WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_models",
    "context_aware_two_tower.weights.h5"
)


# ============================================================
# TRAINED USER ENCODER
# ============================================================

class TrainedUserEncoder:

    def __init__(
        self,
        context_path=CONTEXT_PATH,
        weights_path=WEIGHTS_PATH
    ):

        self.context_path = Path(
            context_path
        )

        self.weights_path = Path(
            weights_path
        )

        self._load_user_context()

        self._load_trained_model()


    # ========================================================
    # LOAD USER CONTEXT
    # ========================================================

    def _load_user_context(self):

        if not self.context_path.exists():

            raise FileNotFoundError(
                f"User context file not found: "
                f"{self.context_path}"
            )

        data = np.load(
            self.context_path,
            allow_pickle=True
        )

        self.customer_ids = data[
            "customer_ids"
        ]

        self.user_indices = data[
            "user_indices"
        ]

        self.customer_total_purchases = data[
            "customer_total_purchases"
        ]

        self.customer_avg_price = data[
            "customer_avg_price"
        ]

        self.user_context = np.column_stack(
            (
                self.customer_total_purchases,
                self.customer_avg_price
            )
        ).astype(
            np.float32
        )

        print(
            f"User context loaded: "
            f"{self.user_context.shape}"
        )


    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    def _load_trained_model(self):

        if not self.weights_path.exists():

            raise FileNotFoundError(
                f"Model weights not found: "
                f"{self.weights_path}"
            )

        # The actual ContextAwareTwoTowerModel
        # takes NO constructor arguments.
        self.full_model = (
            ContextAwareTwoTowerModel()
        )

        # Build the model using the exact input
        # structure expected by train_real_two_tower.py.
        dummy_inputs = {

            "user_index": tf.constant(
                [0],
                dtype=tf.int32
            ),

            "item_index": tf.constant(
                [0],
                dtype=tf.int32
            ),

            "user_context": tf.zeros(
                (1, 2),
                dtype=tf.float32
            ),

            "item_context": tf.zeros(
                (1, 2),
                dtype=tf.float32
            )
        }

        self.full_model(
            dummy_inputs,
            training=False
        )

        # Load the persisted trained weights.
        self.full_model.load_weights(
            self.weights_path
        )

        # Reuse the trained user tower.
        self.user_tower = (
            self.full_model.user_tower
        )

        print(
            "Trained user tower loaded successfully."
        )


    # ========================================================
    # GET USER INDEX
    # ========================================================

    def get_user_index(
        self,
        customer_id
    ):

        # H&M customer IDs are strings.
        customer_id = str(
            customer_id
        )

        position = np.searchsorted(
            self.customer_ids,
            customer_id
        )

        if (
            position >= len(
                self.customer_ids
            )
            or self.customer_ids[
                position
            ] != customer_id
        ):

            raise KeyError(
                f"Customer ID not found: "
                f"{customer_id}"
            )

        return int(
            self.user_indices[
                position
            ]
        )


    # ========================================================
    # GET USER CONTEXT
    # ========================================================

    def get_user_context(
        self,
        customer_id
    ):

        user_index = (
            self.get_user_index(
                customer_id
            )
        )

        return self.user_context[
            user_index
        ]


    # ========================================================
    # GENERATE USER EMBEDDING
    # ========================================================

    def encode(
        self,
        customer_id
    ):

        user_index = (
            self.get_user_index(
                customer_id
            )
        )

        context = (
            self.get_user_context(
                customer_id
            )
        )

        user_index_tensor = tf.constant(
            [user_index],
            dtype=tf.int32
        )

        context_tensor = tf.constant(
            [context],
            dtype=tf.float32
        )

        # IMPORTANT:
        # UserTower itself does not perform the
        # log1p normalization.
        #
        # The original ContextAwareTwoTowerModel
        # performs normalization before calling
        # the user tower.
        #
        # Therefore we reproduce the exact
        # normalization used during training.

        purchase_count = (
            context_tensor[:, 0:1]
        )

        average_price = (
            context_tensor[:, 1:2]
        )

        purchase_count = tf.math.log1p(
            purchase_count
        )

        average_price = tf.math.log1p(
            average_price
        )

        normalized_context = tf.concat(
            [
                purchase_count,
                average_price
            ],
            axis=1
        )

        embedding = self.user_tower(
            user_index_tensor,
            normalized_context
        )

        return embedding.numpy()[0]


    # ========================================================
    # ENCODE WITH DETAILS
    # ========================================================

    def encode_with_details(
        self,
        customer_id
    ):

        customer_id = str(
            customer_id
        )

        user_index = (
            self.get_user_index(
                customer_id
            )
        )

        context = (
            self.get_user_context(
                customer_id
            )
        )

        embedding = (
            self.encode(
                customer_id
            )
        )

        return {

            "customer_id":
                customer_id,

            "user_index":
                user_index,

            "customer_total_purchases":
                float(context[0]),

            "customer_avg_price":
                float(context[1]),

            "embedding":
                embedding
        }


# ============================================================
# TEST
# ============================================================

def main():

    print(
        "\n=== TRAINED USER ENCODER TEST ==="
    )

    encoder = (
        TrainedUserEncoder()
    )

    # H&M customer IDs are strings.
    test_customer_id = str(
        encoder.customer_ids[5]
    )

    print(
        f"\nTest customer ID:\n"
        f"{test_customer_id}"
    )

    # --------------------------------------------------------
    # USER INDEX
    # --------------------------------------------------------

    user_index = (
        encoder.get_user_index(
            test_customer_id
        )
    )

    print(
        f"\nUser index:\n"
        f"{user_index}"
    )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = (
        encoder.get_user_context(
            test_customer_id
        )
    )

    print(
        "\nUser context:"
    )

    print(
        f"Total purchases: "
        f"{context[0]}"
    )

    print(
        f"Average price: "
        f"{context[1]}"
    )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    embedding = (
        encoder.encode(
            test_customer_id
        )
    )

    print(
        f"\nEmbedding shape: "
        f"{embedding.shape}"
    )

    print(
        f"Embedding dtype: "
        f"{embedding.dtype}"
    )

    print(
        "\nFirst 10 embedding values:"
    )

    print(
        embedding[:10]
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    norm = np.linalg.norm(
        embedding
    )

    print(
        f"\nEmbedding L2 norm: "
        f"{norm}"
    )

    assert embedding.shape == (
        64,
    ), (
        f"Expected (64,), "
        f"got {embedding.shape}"
    )

    assert np.isfinite(
        embedding
    ).all(), (
        "Embedding contains "
        "NaN or infinite values"
    )

    assert np.isclose(
        norm,
        1.0,
        atol=1e-5
    ), (
        f"Embedding is not normalized. "
        f"Norm={norm}"
    )

    print(
        "\n=== TRAINED USER ENCODER PASSED ==="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()