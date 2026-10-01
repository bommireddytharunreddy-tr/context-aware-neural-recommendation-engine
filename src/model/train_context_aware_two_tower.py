import os
import sys
import tensorflow as tf

# Allow imports from src/model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from user_tower import UserTower
from item_tower import ItemTower


# ============================================================
# CONFIGURATION
# ============================================================

NUM_USERS = 1_362_281
NUM_ITEMS = 104_547

EMBEDDING_DIM = 64
CONTEXT_DIM = 2

BATCH_SIZE = 1024
EPOCHS = 1

LEARNING_RATE = 0.001


# ============================================================
# CONTEXT-AWARE TWO-TOWER MODEL
# ============================================================

class ContextAwareTwoTowerModel(tf.keras.Model):

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

    def call(self, inputs, training=False):

        user_index = inputs["user_index"]
        item_index = inputs["item_index"]

        user_context = inputs["user_context"]
        item_context = inputs["item_context"]

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
def train_step(model, optimizer, loss_function, inputs, labels):

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
# DEMO TRAINING DATA
# ============================================================

def create_demo_dataset():

    print("\n=== CREATING DEMO TRAINING DATA ===")

    user_index = tf.constant(
        [0, 1, 2, 3, 4, 5, 6, 7],
        dtype=tf.int32
    )

    item_index = tf.constant(
        [10, 20, 30, 40, 50, 60, 70, 80],
        dtype=tf.int32
    )

    labels = tf.constant(
        [1.0, 0.0, 1.0, 0.0,
         1.0, 0.0, 1.0, 0.0],
        dtype=tf.float32
    )

    # User context:
    # [customer_total_purchases,
    #  customer_avg_price]

    user_context = tf.constant(
        [
            [10.0, 0.20],
            [20.0, 0.30],
            [30.0, 0.40],
            [40.0, 0.50],
            [50.0, 0.60],
            [60.0, 0.70],
            [70.0, 0.80],
            [80.0, 0.90]
        ],
        dtype=tf.float32
    )

    # Item context:
    # [item_total_purchases,
    #  item_avg_price]

    item_context = tf.constant(
        [
            [100.0, 0.20],
            [200.0, 0.30],
            [300.0, 0.40],
            [400.0, 0.50],
            [500.0, 0.60],
            [600.0, 0.70],
            [700.0, 0.80],
            [800.0, 0.90]
        ],
        dtype=tf.float32
    )

    inputs = {
        "user_index": user_index,
        "item_index": item_index,
        "user_context": user_context,
        "item_context": item_context
    }

    dataset = tf.data.Dataset.from_tensor_slices(
        (inputs, labels)
    )

    dataset = dataset.batch(BATCH_SIZE)

    return dataset


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model():

    print("\n==========================================")
    print("CONTEXT-AWARE TWO-TOWER TRAINING")
    print("==========================================")

    model = ContextAwareTwoTowerModel()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    loss_function = tf.keras.losses.BinaryCrossentropy(
        from_logits=True
    )

    dataset = create_demo_dataset()

    print("\n=== STARTING TRAINING ===")

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

            epoch_loss += float(loss)
            batch_count += 1

        average_loss = epoch_loss / batch_count

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

    print("\n=== TRAINING COMPLETED ===")

    # --------------------------------------------------------
    # TEST MODEL
    # --------------------------------------------------------

    sample_inputs = {
        "user_index": tf.constant(
            [0, 1, 2, 3],
            dtype=tf.int32
        ),

        "item_index": tf.constant(
            [10, 20, 30, 40],
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

    scores = model(sample_inputs)

    print("\n=== MODEL OUTPUT TEST ===")

    print("User indices:")
    print(sample_inputs["user_index"].numpy())

    print("\nItem indices:")
    print(sample_inputs["item_index"].numpy())

    print("\nSimilarity scores:")
    print(scores.numpy())

    print("\nScore shape:")
    print(scores.shape)

    print("\nTrainable parameters:")
    print(model.count_params())

    print("\n=== TWO-TOWER TRAINING TEST PASSED ===")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_model()