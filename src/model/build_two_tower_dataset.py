from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMTwoTowerDataset")
        .getOrCreate()
    )


def load_transactions(spark):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{DATA_DIR}/transactions_train.csv")
    )


def create_user_mapping(transactions):
    window = Window.orderBy("customer_id")

    return (
        transactions
        .select("customer_id")
        .distinct()
        .withColumn(
            "user_index",
            F.row_number().over(window) - 1
        )
    )


def create_item_mapping(transactions):
    window = Window.orderBy("article_id")

    return (
        transactions
        .select("article_id")
        .distinct()
        .withColumn(
            "item_index",
            F.row_number().over(window) - 1
        )
    )


def build_training_dataset(transactions, user_mapping, item_mapping):
    return (
        transactions
        .select(
            "customer_id",
            "article_id",
            "price",
            "sales_channel_id"
        )
        .join(user_mapping, on="customer_id", how="inner")
        .join(item_mapping, on="article_id", how="inner")
        .select(
            "user_index",
            "item_index",
            "price",
            "sales_channel_id"
        )
        .withColumn("label", F.lit(1))
    )


def main():
    spark = create_spark_session()

    transactions = load_transactions(spark)

    user_mapping = create_user_mapping(transactions)
    item_mapping = create_item_mapping(transactions)

    training_dataset = build_training_dataset(
        transactions,
        user_mapping,
        item_mapping
    )

    print("\n=== TWO-TOWER TRAINING DATA SAMPLE ===")
    training_dataset.show(10, truncate=False)

    print("\n=== TWO-TOWER TRAINING DATA SCHEMA ===")
    training_dataset.printSchema()

    print("\n=== TRAINING DATA COUNT ===")
    print(training_dataset.count())



    spark.stop()


if __name__ == "__main__":
    main()