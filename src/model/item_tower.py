import tensorflow as tf


class ItemTower(tf.keras.Model):

    def __init__(
        self,
        num_items,
        embedding_dim=64,
        context_dim=2
    ):
        super().__init__()

        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_dim,
            name="item_embedding"
        )

        self.context_dense = tf.keras.layers.Dense(
            32,
            activation="relu",
            name="item_context_dense"
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

    def call(
        self,
        item_index,
        item_context=None
    ):

        item_vector = self.item_embedding(
            item_index
        )

        if item_context is not None:

            context_vector = self.context_dense(
                item_context
            )

            item_vector = tf.concat(
                [
                    item_vector,
                    context_vector
                ],
                axis=1
            )

        item_vector = self.dense_1(
            item_vector
        )

        item_vector = self.dense_2(
            item_vector
        )

        return tf.math.l2_normalize(
            item_vector,
            axis=1
        )


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

    sample_context = tf.constant(
        [
            [100.0, 0.25],
            [200.0, 0.50],
            [300.0, 0.75],
            [400.0, 1.00],
            [500.0, 1.25]
        ],
        dtype=tf.float32
    )

    output = model(
        sample_items,
        sample_context
    )

    print("\n=== CONTEXT-AWARE ITEM TOWER ===")

    print(
        "Input items:",
        sample_items.numpy()
    )

    print(
        "Context shape:",
        sample_context.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    print(
        "Embedding dimension:",
        EMBEDDING_DIM
    )

    print(
        "Model parameters:",
        model.count_params()
    )

    print(
        "\n=== ITEM TOWER TEST PASSED ==="
    )