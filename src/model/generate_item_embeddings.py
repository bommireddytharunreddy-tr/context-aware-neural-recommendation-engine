import tensorflow as tf

from train_two_tower import TwoTowerModel


NUM_ITEMS = 104_547
EMBEDDING_DIM = 64


def generate_item_embeddings(model):
    item_indices = tf.range(
        NUM_ITEMS,
        dtype=tf.int32
    )

    item_embeddings = model.item_tower(item_indices)

    return item_embeddings


def main():
    print("\n=== GENERATING ITEM EMBEDDINGS ===")

    model = TwoTowerModel()

    # Build the model before accessing the Item Tower.
    sample_items = tf.constant([0, 1, 2], dtype=tf.int32)
    model((sample_items, sample_items))

    item_embeddings = generate_item_embeddings(model)

    print("\n=== ITEM EMBEDDINGS ===")
    print("Number of items:", NUM_ITEMS)
    print("Embedding dimension:", EMBEDDING_DIM)
    print("Embedding shape:", item_embeddings.shape)

    print("\n=== FIRST 5 ITEM EMBEDDINGS ===")
    print(item_embeddings[:5].numpy())

    print("\n=== ITEM EMBEDDING GENERATION COMPLETED ===")


if __name__ == "__main__":
    main()