import tensorflow as tf


def create_tf_dataset(
    user_indices,
    item_indices,
    labels,
    batch_size=1024,
    shuffle=False
):
    """
    Convert user, item and label arrays
    into a TensorFlow Dataset.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "user_index": user_indices,
                "item_index": item_indices
            },
            labels
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(
                len(user_indices),
                100000
            ),
            seed=42
        )

    dataset = (
        dataset
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


def inspect_dataset(
    dataset,
    num_batches=1
):

    print("\n=== TENSORFLOW DATASET SAMPLE ===")

    for batch_inputs, batch_labels in dataset.take(
        num_batches
    ):

        print("\nUser indices:")
        print(
            batch_inputs["user_index"]
            .numpy()[:10]
        )

        print("\nItem indices:")
        print(
            batch_inputs["item_index"]
            .numpy()[:10]
        )

        print("\nLabels:")
        print(
            batch_labels.numpy()[:10]
        )

        print("\nBatch size:")
        print(
            batch_inputs["user_index"]
            .shape[0]
        )

        print(
            "\n=== TENSORFLOW DATASET TEST PASSED ==="
        )


if __name__ == "__main__":

    print(
        "=== TENSORFLOW INPUT PIPELINE TEST ==="
    )

    user_indices = tf.constant(
        [0, 1, 2, 3, 4, 5],
        dtype=tf.int32
    )

    item_indices = tf.constant(
        [10, 20, 30, 40, 50, 60],
        dtype=tf.int32
    )

    labels = tf.constant(
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        dtype=tf.float32
    )

    dataset = create_tf_dataset(
        user_indices,
        item_indices,
        labels,
        batch_size=2,
        shuffle=True
    )

    inspect_dataset(dataset)