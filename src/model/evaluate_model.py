import tensorflow as tf

from train_two_tower import (
    TwoTowerModel,
    create_training_dataset,
    NUM_USERS,
    NUM_ITEMS,
    EMBEDDING_DIM
)


BATCH_SIZE = 1024
TOP_K = 10


def train_model():
    model = TwoTowerModel()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    loss_function = tf.keras.losses.BinaryCrossentropy(
        from_logits=True
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_function
    )

    dataset = create_training_dataset()

    model.fit(
        dataset,
        epochs=1,
        verbose=1
    )

    return model


def create_item_embeddings(model):
    item_indices = tf.range(
        NUM_ITEMS,
        dtype=tf.int32
    )

    return model.item_tower(item_indices)


def calculate_recall_at_k(model, item_embeddings):
    test_users = tf.constant(
        [0, 1, 5, 35, 100],
        dtype=tf.int32
    )

    test_items = tf.constant(
        [0, 1, 5, 100, 1000],
        dtype=tf.int32
    )

    user_embeddings = model.user_tower(test_users)

    scores = tf.matmul(
        user_embeddings,
        item_embeddings,
        transpose_b=True
    )

    top_k_items = tf.math.top_k(
        scores,
        k=TOP_K
    ).indices

    hits = tf.reduce_any(
        tf.equal(
            top_k_items,
            tf.expand_dims(test_items, axis=1)
        ),
        axis=1
    )

    recall = tf.reduce_mean(
        tf.cast(hits, tf.float32)
    )

    print("\n=== MODEL EVALUATION ===")
    print("Number of test users:", len(test_users))
    print("Top-K:", TOP_K)
    print("Top-K item indices:")
    print(top_k_items.numpy())

    print("\nActual item indices:")
    print(test_items.numpy())

    print("\nRecall@10:", float(recall.numpy()))

    return recall


def main():
    print("\n=== STARTING MODEL EVALUATION ===")

    model = train_model()

    print("\nCreating item embeddings...")
    item_embeddings = create_item_embeddings(model)

    print("Item embedding shape:", item_embeddings.shape)

    calculate_recall_at_k(
        model,
        item_embeddings
    )

    print("\n=== MODEL EVALUATION COMPLETED ===")


if __name__ == "__main__":
    main()