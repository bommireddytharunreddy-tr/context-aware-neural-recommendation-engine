from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMFeatureEngineering")
        .getOrCreate()
    )


def load_transactions(spark):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{DATA_DIR}/transactions_train.csv")
    )


def create_features(transactions):
    customer_frequency = (
        transactions
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("purchase_frequency")
        )
    )

    article_popularity = (
        transactions
        .groupBy("article_id")
        .agg(
            F.count("*").alias("purchase_count")
        )
    )

    customer_article = (
        transactions
        .groupBy("customer_id", "article_id")
        .agg(
            F.max("t_dat").alias("last_purchase_date"),
            F.count("*").alias("customer_article_purchase_count")
        )
    )

    max_date = transactions.select(
        F.max("t_dat").alias("max_date")
    ).first()["max_date"]

    customer_article = customer_article.withColumn(
        "recency_days",
        F.datediff(F.lit(max_date), F.col("last_purchase_date"))
    )

    return customer_frequency, article_popularity, customer_article


if __name__ == "__main__":
    spark = create_spark_session()

    transactions = load_transactions(spark)

    customer_frequency, article_popularity, customer_article = create_features(
        transactions
    )

    print("\n=== CUSTOMER FREQUENCY ===")
    customer_frequency.show(5)

    print("\n=== ARTICLE POPULARITY ===")
    article_popularity.show(5)

    print("\n=== CUSTOMER-ARTICLE FEATURES ===")
    customer_article.show(5)

    spark.stop()