from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from negative_sampling import create_training_dataset
from create_id_mappings import (
    create_user_mapping,
    create_item_mapping
)


DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMPrepareTwoTowerTrainingData")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def load_transactions(spark):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(
            f"{DATA_DIR}/transactions_train.csv"
        )
    )


def prepare_two_tower_training_data(
    transactions
):

    print("\n=== CREATING TRAINING DATA ===")

    training_data = create_training_dataset(
        transactions
    )

    print(
        "\n=== CREATING ID MAPPINGS ==="
    )

    user_mapping = create_user_mapping(
        transactions
    )

    item_mapping = create_item_mapping(
        transactions
    )

    print(
        "\n=== JOINING USER MAPPING ==="
    )

    training_data = (
        training_data
        .join(
            user_mapping,
            on="customer_id",
            how="inner"
        )
    )

    print(
        "\n=== JOINING ITEM MAPPING ==="
    )

    training_data = (
        training_data
        .join(
            item_mapping,
            on="article_id",
            how="inner"
        )
    )

    training_data = (
        training_data
        .select(
            "user_index",
            "item_index",
            "label"
        )
    )

    return training_data


if __name__ == "__main__":

    spark = create_spark_session()

    try:

        transactions = load_transactions(
            spark
        )

        training_data = (
            prepare_two_tower_training_data(
                transactions
            )
        )

        print(
            "\n=== TRAINING DATA SCHEMA ==="
        )

        training_data.printSchema()

        print(
            "\n=== LABEL DISTRIBUTION ==="
        )

        training_data.groupBy(
            "label"
        ).count().show()

        print(
            "\n=== SAMPLE TRAINING DATA ==="
        )

        training_data.show(
            10,
            truncate=False
        )

        print(
            "\n=== UNIQUE USERS ==="
        )

        print(
            training_data
            .select("user_index")
            .distinct()
            .count()
        )

        print(
            "\n=== UNIQUE ITEMS ==="
        )

        print(
            training_data
            .select("item_index")
            .distinct()
            .count()
        )

        print(
            "\n=== TWO-TOWER TRAINING DATA TEST PASSED ==="
        )

    finally:

        spark.stop()