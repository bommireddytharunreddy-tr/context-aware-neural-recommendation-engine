import tensorflow as tf

from user_tower import UserTower
from item_tower import ItemTower


NUM_USERS = 1_362_281
NUM_ITEMS = 104_547

EMBEDDING_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 1


class TwoTowerModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.user_tower = UserTower(
            num_users=NUM_USERS,
            embedding_dim=EMBEDDING_DIM
        )

        self.item_tower = ItemTower(
            num_items=NUM_ITEMS,
            embedding_dim=EMBEDDING_DIM
        )

    def call(self, inputs):
        user_index, item_index = inputs

        user_vector = self.user_tower(user_index)
        item_vector = self.item_tower(item_index)

        similarity = tf.reduce_sum(
            user_vector * item_vector,
            axis=1
        )

        return similarity


def create_training_dataset():

    user_index = tf.random.uniform(
        shape=(100_000,),
        minval=0,
        maxval=NUM_USERS,
        dtype=tf.int32
    )

    item_index = tf.random.uniform(
        shape=(100_000,),
        minval=0,
        maxval=NUM_ITEMS,
        dtype=tf.int32
    )

    labels = tf.ones(
        shape=(100_000,),
        dtype=tf.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        ((user_index, item_index), labels)
    )

    dataset = dataset.shuffle(10_000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():

    print("\n=== TWO-TOWER MODEL TRAINING ===")

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
        epochs=EPOCHS
    )

    print("\n=== MODEL TEST ===")

    sample_users = tf.constant(
        [0, 1, 5, 35],
        dtype=tf.int32
    )

    sample_items = tf.constant(
        [0, 1, 5, 100],
        dtype=tf.int32
    )

    scores = model(
        (sample_users, sample_items)
    )

    print("User indices:", sample_users.numpy())
    print("Item indices:", sample_items.numpy())
    print("Similarity scores:", scores.numpy())
    print("Score shape:", scores.shape)

    print("\n=== TWO-TOWER TRAINING COMPLETED ===")


if __name__ == "__main__":
    main()