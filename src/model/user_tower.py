import tensorflow as tf


class UserTower(tf.keras.Model):

    def __init__(
        self,
        num_users,
        embedding_dim=64,
        context_dim=2
    ):
        super().__init__()

        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_dim,
            name="user_embedding"
        )

        self.context_dense = tf.keras.layers.Dense(
            32,
            activation="relu",
            name="user_context_dense"
        )

        self.dense_1 = tf.keras.layers.Dense(
            128,
            activation="relu",
            name="user_dense_1"
        )

        self.dense_2 = tf.keras.layers.Dense(
            embedding_dim,
            activation=None,
            name="user_dense_2"
        )

    def call(
        self,
        user_index,
        user_context=None
    ):

        user_vector = self.user_embedding(
            user_index
        )

        if user_context is not None:

            context_vector = self.context_dense(
                user_context
            )

            user_vector = tf.concat(
                [
                    user_vector,
                    context_vector
                ],
                axis=1
            )

        user_vector = self.dense_1(
            user_vector
        )

        user_vector = self.dense_2(
            user_vector
        )

        return tf.math.l2_normalize(
            user_vector,
            axis=1
        )


if __name__ == "__main__":

    NUM_USERS = 1_362_281
    EMBEDDING_DIM = 64

    model = UserTower(
        num_users=NUM_USERS,
        embedding_dim=EMBEDDING_DIM
    )

    sample_users = tf.constant(
        [0, 1, 5, 35, 100],
        dtype=tf.int32
    )

    sample_context = tf.constant(
        [
            [10.0, 0.25],
            [20.0, 0.50],
            [30.0, 0.75],
            [40.0, 1.00],
            [50.0, 1.25]
        ],
        dtype=tf.float32
    )

    output = model(
        sample_users,
        sample_context
    )

    print("\n=== CONTEXT-AWARE USER TOWER ===")
    print(
        "Input users:",
        sample_users.numpy()
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
        "\n=== USER TOWER TEST PASSED ==="
    )