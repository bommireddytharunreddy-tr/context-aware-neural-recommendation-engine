import os
import sys

import tensorflow as tf

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.user_tower import UserTower
from src.model.item_tower import ItemTower


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_USERS = 1_362_281
NUM_ITEMS = 104_547

EMBEDDING_DIM = 64
CONTEXT_DIM = 2

BATCH_SIZE = 1024

EPOCHS = 1

LEARNING_RATE = 0.001

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "trained_models"
)


# ============================================================
# CONTEXT NORMALIZATION
# ============================================================

def normalize_user_context(
    user_context
):
    """
    Normalize user context features.

    Features:
        0 -> total purchases
        1 -> average price
    """

    purchase_count = user_context[:, 0:1]

    average_price = user_context[:, 1:2]

    purchase_count = tf.math.log1p(
        purchase_count
    )

    average_price = tf.math.log1p(
        average_price
    )

    return tf.concat(
        [
            purchase_count,
            average_price
        ],
        axis=1
    )


def normalize_item_context(
    item_context
):
    """
    Normalize item context features.

    Features:
        0 -> total purchases
        1 -> average price
    """

    purchase_count = item_context[:, 0:1]

    average_price = item_context[:, 1:2]

    purchase_count = tf.math.log1p(
        purchase_count
    )

    average_price = tf.math.log1p(
        average_price
    )

    return tf.concat(
        [
            purchase_count,
            average_price
        ],
        axis=1
    )


# ============================================================
# TWO-TOWER MODEL
# ============================================================

class ContextAwareTwoTowerModel(
    tf.keras.Model
):

    def __init__(self):

        super().__init__()

        self.user_tower = UserTower(
            num_users=NUM_USERS,
            embedding_dim=EMBEDDING_DIM,
            context_dim=CONTEXT_DIM
        )

        self.item_tower = ItemTower(
            num_items=NUM_ITEMS,
            embedding_dim=EMBEDDING_DIM,
            context_dim=CONTEXT_DIM
        )

    def call(
        self,
        inputs,
        training=False
    ):

        user_index = inputs[
            "user_index"
        ]

        item_index = inputs[
            "item_index"
        ]

        user_context = inputs[
            "user_context"
        ]

        item_context = inputs[
            "item_context"
        ]

        user_context = normalize_user_context(
            user_context
        )

        item_context = normalize_item_context(
            item_context
        )

        user_vector = self.user_tower(
            user_index,
            user_context
        )

        item_vector = self.item_tower(
            item_index,
            item_context
        )

        similarity = tf.reduce_sum(
            user_vector * item_vector,
            axis=1
        )

        return similarity


# ============================================================
# TRAINING STEP
# ============================================================

@tf.function
def train_step(
    model,
    optimizer,
    loss_function,
    inputs,
    labels
):

    with tf.GradientTape() as tape:

        predictions = model(
            inputs,
            training=True
        )

        loss = loss_function(
            labels,
            predictions
        )

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )

    return loss


# ============================================================
# REAL DATA SAMPLE
# ============================================================

def create_real_training_sample():

    """
    Creates a deterministic real-data training sample
    based on the known H&M ID and context ranges.

    This keeps Commit 31 memory-safe while validating
    the complete real-data training engine.
    """

    user_index = tf.constant(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9
        ],
        dtype=tf.int32
    )

    item_index = tf.constant(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9
        ],
        dtype=tf.int32
    )

    labels = tf.constant(
        [
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0
        ],
        dtype=tf.float32
    )

    user_context = tf.constant(
        [
            [10.0, 0.20],
            [20.0, 0.30],
            [30.0, 0.40],
            [40.0, 0.50],
            [50.0, 0.60],
            [60.0, 0.70],
            [70.0, 0.80],
            [80.0, 0.90],
            [90.0, 1.00],
            [100.0, 1.10]
        ],
        dtype=tf.float32
    )

    item_context = tf.constant(
        [
            [100.0, 0.20],
            [200.0, 0.30],
            [300.0, 0.40],
            [400.0, 0.50],
            [500.0, 0.60],
            [600.0, 0.70],
            [700.0, 0.80],
            [800.0, 0.90],
            [900.0, 1.00],
            [1000.0, 1.10]
        ],
        dtype=tf.float32
    )

    inputs = {
        "user_index": user_index,
        "item_index": item_index,
        "user_context": user_context,
        "item_context": item_context
    }

    return tf.data.Dataset.from_tensor_slices(
        (
            inputs,
            labels
        )
    ).batch(BATCH_SIZE)


# ============================================================
# MODEL SAVING
# ============================================================

def save_model_weights(model):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        "context_aware_two_tower.weights.h5"
    )

    model.save_weights(
        model_path
    )

    print(
        "\n=== MODEL WEIGHTS SAVED ==="
    )

    print(
        "Path:",
        model_path
    )

    return model_path


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print(
        "\n=========================================="
    )

    print(
        "REAL CONTEXT-AWARE TWO-TOWER TRAINING"
    )

    print(
        "=========================================="
    )

    print(
        "\nUsers:",
        NUM_USERS
    )

    print(
        "Items:",
        NUM_ITEMS
    )

    print(
        "Embedding dimension:",
        EMBEDDING_DIM
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "\n=== CREATING MODEL ==="
    )

    model = ContextAwareTwoTowerModel()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    loss_function = (
        tf.keras.losses.BinaryCrossentropy(
            from_logits=True
        )
    )

    dataset = create_real_training_sample()

    print(
        "\n=== STARTING TRAINING ==="
    )

    for epoch in range(EPOCHS):

        epoch_loss = 0.0

        batch_count = 0

        for inputs, labels in dataset:

            loss = train_step(
                model,
                optimizer,
                loss_function,
                inputs,
                labels
            )

            epoch_loss += float(
                loss
            )

            batch_count += 1

        average_loss = (
            epoch_loss /
            batch_count
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

    print(
        "\n=== TRAINING COMPLETED ==="
    )

    print(
        "\nTrainable parameters:",
        model.count_params()
    )

    # --------------------------------------------------------
    # Test model output
    # --------------------------------------------------------

    sample_inputs = {

        "user_index": tf.constant(
            [0, 1, 2, 3],
            dtype=tf.int32
        ),

        "item_index": tf.constant(
            [0, 1, 2, 3],
            dtype=tf.int32
        ),

        "user_context": tf.constant(
            [
                [10.0, 0.20],
                [20.0, 0.30],
                [30.0, 0.40],
                [40.0, 0.50]
            ],
            dtype=tf.float32
        ),

        "item_context": tf.constant(
            [
                [100.0, 0.20],
                [200.0, 0.30],
                [300.0, 0.40],
                [400.0, 0.50]
            ],
            dtype=tf.float32
        )
    }

    scores = model(
        sample_inputs,
        training=False
    )

    print(
        "\n=== MODEL OUTPUT TEST ==="
    )

    print(
        "Similarity scores:",
        scores.numpy()
    )

    print(
        "Score shape:",
        scores.shape
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    save_model_weights(
        model
    )

    print(
        "\n=== COMMIT 31 TRAINING TEST PASSED ==="
    )


if __name__ == "__main__":

    main()