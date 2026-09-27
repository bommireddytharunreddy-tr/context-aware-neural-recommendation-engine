from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMContextFeatures")
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


def create_context_features(transactions):

    # Time-based features
    context_features = (
        transactions
        .withColumn("purchase_year", F.year("t_dat"))
        .withColumn("purchase_month", F.month("t_dat"))
        .withColumn("purchase_day", F.dayofmonth("t_dat"))
        .withColumn("purchase_day_of_week", F.dayofweek("t_dat"))
    )

    # Customer-level purchase behavior
    customer_features = (
        context_features
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("customer_total_purchases"),
            F.avg("price").alias("customer_avg_price"),
            F.max("t_dat").alias("customer_last_purchase_date")
        )
    )

    # Item popularity
    item_features = (
        context_features
        .groupBy("article_id")
        .agg(
            F.count("*").alias("item_total_purchases"),
            F.avg("price").alias("item_avg_price"),
            F.max("t_dat").alias("item_last_purchase_date")
        )
    )

    # Monthly item popularity
    monthly_item_popularity = (
        context_features
        .groupBy(
            "article_id",
            "purchase_year",
            "purchase_month"
        )
        .agg(
            F.count("*").alias("monthly_item_purchases")
        )
    )

    return (
        context_features,
        customer_features,
        item_features,
        monthly_item_popularity
    )


if __name__ == "__main__":

    spark = create_spark_session()

    try:
        transactions = load_transactions(spark)

        (
            context_features,
            customer_features,
            item_features,
            monthly_item_popularity
        ) = create_context_features(transactions)

        print("\n=== CONTEXT FEATURES ===")
        context_features.select(
            "customer_id",
            "article_id",
            "purchase_year",
            "purchase_month",
            "purchase_day_of_week"
        ).show(5)

        print("\n=== CUSTOMER FEATURES ===")
        customer_features.show(5)

        print("\n=== ITEM FEATURES ===")
        item_features.show(5)

        print("\n=== MONTHLY ITEM POPULARITY ===")
        monthly_item_popularity.show(5)

        print("\n=== CONTEXT FEATURE ENGINEERING TEST PASSED ===")

    finally:
        spark.stop()