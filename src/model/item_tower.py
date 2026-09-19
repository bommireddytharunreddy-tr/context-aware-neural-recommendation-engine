import tensorflow as tf


class ItemTower(tf.keras.Model):
    def __init__(self, num_items, embedding_dim=64):
        super().__init__()

        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_dim,
            name="item_embedding"
        )

        self.dense_1 = tf.keras.layers.Dense(
            128,
            activation="relu",
            name="item_dense_1"
        )

        self.dense_2 = tf.keras.layers.Dense(
            embedding_dim,
            activation=None,
            name="item_dense_2"
        )

    def call(self, item_index):
        item_vector = self.item_embedding(item_index)
        item_vector = self.dense_1(item_vector)
        item_vector = self.dense_2(item_vector)

        return tf.math.l2_normalize(item_vector, axis=1)


if __name__ == "__main__":
    NUM_ITEMS = 104_547
    EMBEDDING_DIM = 64

    model = ItemTower(
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM
    )

    sample_items = tf.constant(
        [0, 1, 5, 100, 1000],
        dtype=tf.int32
    )

    output = model(sample_items)

    print("\n=== ITEM TOWER ===")
    print("Input items:", sample_items.numpy())
    print("Output shape:", output.shape)
    print("Embedding dimension:", EMBEDDING_DIM)
    print("Model parameters:", model.count_params())

    print("\n=== ITEM TOWER TEST PASSED ===")