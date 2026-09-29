from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMRealInteractions")
        .getOrCreate()
    )


def load_transactions(spark):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{DATA_DIR}/transactions_train.csv")
    )


def prepare_real_interactions(transactions):

    interactions = (
        transactions
        .select(
            "customer_id",
            "article_id",
            "t_dat",
            "price",
            "sales_channel_id"
        )
        .filter(F.col("customer_id").isNotNull())
        .filter(F.col("article_id").isNotNull())
        .filter(F.col("t_dat").isNotNull())
        .filter(F.col("price").isNotNull())
        .filter(F.col("price") > 0)
    )

    # Aggregate repeated purchases of the same item by the same customer.
    user_item_interactions = (
        interactions
        .groupBy("customer_id", "article_id")
        .agg(
            F.count("*").alias("interaction_count"),
            F.min("t_dat").alias("first_purchase_date"),
            F.max("t_dat").alias("last_purchase_date"),
            F.avg("price").alias("average_price"),
            F.max("sales_channel_id").alias("sales_channel_id")
        )
    )

    # Every observed customer-item pair represents a positive interaction.
    user_item_interactions = user_item_interactions.withColumn(
        "label",
        F.lit(1.0)
    )

    return user_item_interactions


if __name__ == "__main__":

    spark = create_spark_session()

    try:
        transactions = load_transactions(spark)

        interactions = prepare_real_interactions(transactions)

        print("\n=== REAL INTERACTION DATA ===")

        print("\nTotal unique customer-item interactions:")
        print(interactions.count())

        print("\nNumber of unique customers:")
        print(
            interactions
            .select("customer_id")
            .distinct()
            .count()
        )

        print("\nNumber of unique articles:")
        print(
            interactions
            .select("article_id")
            .distinct()
            .count()
        )

        print("\nSample interactions:")
        interactions.show(10, truncate=False)

        print("\n=== REAL INTERACTION PREPARATION TEST PASSED ===")

    finally:
        spark.stop()