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
        .csv(
            f"{DATA_DIR}/transactions_train.csv"
        )
        .withColumn(
            "t_dat",
            F.to_date("t_dat")
        )
    )


def create_training_context_features(
    transactions
):

    # --------------------------------------------------
    # Time-based features
    # --------------------------------------------------

    context_features = (
        transactions
        .withColumn(
            "purchase_year",
            F.year("t_dat")
        )
        .withColumn(
            "purchase_month",
            F.month("t_dat")
        )
        .withColumn(
            "purchase_day",
            F.dayofmonth("t_dat")
        )
        .withColumn(
            "purchase_day_of_week",
            F.dayofweek("t_dat")
        )
    )

    # --------------------------------------------------
    # Customer-level features
    # --------------------------------------------------

    customer_features = (
        context_features
        .groupBy("customer_id")
        .agg(
            F.count("*").alias(
                "customer_total_purchases"
            ),
            F.avg("price").alias(
                "customer_avg_price"
            ),
            F.max("t_dat").alias(
                "customer_last_purchase_date"
            )
        )
    )

    # --------------------------------------------------
    # Item-level features
    # --------------------------------------------------

    item_features = (
        context_features
        .groupBy("article_id")
        .agg(
            F.count("*").alias(
                "item_total_purchases"
            ),
            F.avg("price").alias(
                "item_avg_price"
            ),
            F.max("t_dat").alias(
                "item_last_purchase_date"
            )
        )
    )

    # --------------------------------------------------
    # Join customer + item features
    # --------------------------------------------------

    training_features = (
        context_features
        .join(
            customer_features,
            on="customer_id",
            how="left"
        )
        .join(
            item_features,
            on="article_id",
            how="left"
        )
    )

    # --------------------------------------------------
    # Select model-relevant contextual features
    # --------------------------------------------------

    training_features = (
        training_features
        .select(
            "customer_id",
            "article_id",
            "t_dat",

            "purchase_year",
            "purchase_month",
            "purchase_day",
            "purchase_day_of_week",

            "customer_total_purchases",
            "customer_avg_price",

            "item_total_purchases",
            "item_avg_price"
        )
    )

    return training_features


if __name__ == "__main__":

    spark = create_spark_session()

    try:

        transactions = load_transactions(
            spark
        )

        training_features = (
            create_training_context_features(
                transactions
            )
        )

        print(
            "\n=== TRAINING CONTEXT FEATURES ==="
        )

        training_features.printSchema()

        print(
            "\n=== FEATURE SAMPLE ==="
        )

        training_features.show(
            10,
            truncate=False
        )

        print(
            "\n=== FEATURE COUNT ==="
        )

        print(
            training_features.count()
        )

        print(
            "\n=== TRAINING CONTEXT FEATURE TEST PASSED ==="
        )

    finally:

        spark.stop()