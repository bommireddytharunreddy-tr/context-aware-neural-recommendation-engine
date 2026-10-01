from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMTrainingContextFeatures")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def load_transactions(spark):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{DATA_DIR}/transactions_train.csv")
        .withColumn("t_dat", F.to_date("t_dat"))
    )


def create_user_context_features(transactions):
    """
    Create context features at user level.
    These features can be used for both positive and negative interactions.
    """

    user_features = (
        transactions
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("customer_total_purchases"),
            F.avg("price").alias("customer_avg_price")
        )
    )

    return user_features


def create_item_context_features(transactions):
    """
    Create context features at item level.
    These features can be used for both positive and negative interactions.
    """

    item_features = (
        transactions
        .groupBy("article_id")
        .agg(
            F.count("*").alias("item_total_purchases"),
            F.avg("price").alias("item_avg_price")
        )
    )

    return item_features


def create_training_context_features(transactions):
    """
    Create separate user-level and item-level context features.
    """

    user_features = create_user_context_features(transactions)
    item_features = create_item_context_features(transactions)

    return user_features, item_features


if __name__ == "__main__":
    spark = create_spark_session()

    try:
        transactions = load_transactions(spark)

        user_features, item_features = create_training_context_features(
            transactions
        )

        print("\n=== USER CONTEXT FEATURES ===")
        user_features.printSchema()
        user_features.show(10, truncate=False)

        print("\n=== ITEM CONTEXT FEATURES ===")
        item_features.printSchema()
        item_features.show(10, truncate=False)

        print("\n=== USER FEATURE COUNT ===")
        print(user_features.count())

        print("\n=== ITEM FEATURE COUNT ===")
        print(item_features.count())

        print("\n=== TRAINING CONTEXT FEATURE TEST PASSED ===")

    finally:
        spark.stop()