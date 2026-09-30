from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMNegativeSampling")
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
    )


def create_positive_interactions(transactions):
    """
    Create unique customer-item interactions.

    label = 1.0 means the customer actually purchased
    the article.
    """

    return (
        transactions
        .select(
            "customer_id",
            "article_id"
        )
        .dropDuplicates()
        .withColumn(
            "label",
            F.lit(1.0)
        )
    )


def create_negative_interactions(
    transactions,
    negative_ratio=1,
    seed=42
):
    """
    Generate negative customer-item interactions.

    This implementation DOES NOT create a full
    customer x item Cartesian product.

    Instead, a deterministic hash is used to select
    candidate articles for each positive interaction.
    """

    # ---------------------------------------------------------
    # 1. Positive interactions
    # ---------------------------------------------------------

    positive = create_positive_interactions(
        transactions
    )

    # ---------------------------------------------------------
    # 2. Create item index
    # ---------------------------------------------------------

    items = (
        transactions
        .select("article_id")
        .distinct()
    )

    item_window = Window.orderBy("article_id")

    items = (
        items
        .withColumn(
            "item_index",
            F.row_number().over(item_window) - 1
        )
    )

    item_count = items.count()

    print("\nTotal unique items:", item_count)

    # ---------------------------------------------------------
    # 3. Generate deterministic candidate item
    # ---------------------------------------------------------

    candidates = (
        positive
        .select(
            "customer_id",
            "article_id"
        )
        .withColumn(
            "hash_value",
            F.abs(
                F.xxhash64(
                    F.col("customer_id"),
                    F.col("article_id"),
                    F.lit(seed)
                )
            )
        )
        .withColumn(
            "candidate_index",
            F.pmod(
                F.col("hash_value"),
                F.lit(item_count)
            )
        )
    )

    # ---------------------------------------------------------
    # 4. Join candidate index with item table
    #
    # Aliases are important because both DataFrames
    # contain article_id.
    # ---------------------------------------------------------

    candidate_df = candidates.alias("c")
    item_df = items.alias("i")

    candidates = (
        candidate_df
        .join(
            item_df,
            F.col("c.candidate_index")
            == F.col("i.item_index"),
            "inner"
        )
        .select(
            F.col("c.customer_id").alias(
                "customer_id"
            ),
            F.col("c.article_id").alias(
                "positive_article_id"
            ),
            F.col("i.article_id").alias(
                "negative_article_id"
            )
        )
    )

    # ---------------------------------------------------------
    # 5. Remove candidates that were actually purchased
    # ---------------------------------------------------------

    purchased = (
        positive
        .select(
            "customer_id",
            F.col("article_id").alias(
                "purchased_article_id"
            )
        )
        .dropDuplicates()
    )

    candidate_df = candidates.alias("c")
    purchased_df = purchased.alias("p")

    negative_candidates = (
        candidate_df
        .join(
            purchased_df,
            (
                F.col("c.customer_id")
                == F.col("p.customer_id")
            )
            &
            (
                F.col("c.negative_article_id")
                == F.col("p.purchased_article_id")
            ),
            "left_anti"
        )
        .select(
            F.col("c.customer_id").alias(
                "customer_id"
            ),
            F.col("c.negative_article_id").alias(
                "article_id"
            )
        )
        .dropDuplicates()
    )

    # ---------------------------------------------------------
    # 6. Count positive interactions per customer
    # ---------------------------------------------------------

    positive_counts = (
        positive
        .groupBy("customer_id")
        .agg(
            F.count("*").alias(
                "positive_count"
            )
        )
    )

    # ---------------------------------------------------------
    # 7. Select negative samples
    # ---------------------------------------------------------

    negative_window = (
        Window
        .partitionBy("customer_id")
        .orderBy(
            F.xxhash64(
                F.col("customer_id"),
                F.col("article_id"),
                F.lit(seed)
            )
        )
    )

    negative = (
        negative_candidates
        .join(
            positive_counts,
            on="customer_id",
            how="inner"
        )
        .withColumn(
            "negative_rank",
            F.row_number().over(
                negative_window
            )
        )
        .filter(
            F.col("negative_rank")
            <= (
                F.col("positive_count")
                * F.lit(negative_ratio)
            )
        )
        .select(
            "customer_id",
            "article_id"
        )
        .withColumn(
            "label",
            F.lit(0.0)
        )
    )

    return negative


def create_training_dataset(transactions):
    """
    Create final training dataset containing:

    label = 1.0 -> positive interaction
    label = 0.0 -> negative interaction
    """

    positive = create_positive_interactions(
        transactions
    )

    negative = create_negative_interactions(
        transactions,
        negative_ratio=1
    )

    training_data = (
        positive
        .unionByName(negative)
    )

    return training_data


if __name__ == "__main__":

    spark = create_spark_session()

    try:

        # -----------------------------------------------------
        # Load H&M transactions
        # -----------------------------------------------------

        transactions = load_transactions(
            spark
        )

        # -----------------------------------------------------
        # Positive interactions
        # -----------------------------------------------------

        print(
            "\n=== POSITIVE INTERACTIONS ==="
        )

        positive = create_positive_interactions(
            transactions
        )

        positive_count = positive.count()

        print(
            "Positive interactions:",
            positive_count
        )

        # -----------------------------------------------------
        # Negative interactions
        # -----------------------------------------------------

        print(
            "\n=== GENERATING NEGATIVE INTERACTIONS ==="
        )

        negative = create_negative_interactions(
            transactions,
            negative_ratio=1
        )

        negative_count = negative.count()

        print(
            "Negative interactions:",
            negative_count
        )

        # -----------------------------------------------------
        # Final training dataset
        # -----------------------------------------------------

        print(
            "\n=== TRAINING DATA ==="
        )

        training_data = (
            positive
            .unionByName(negative)
        )

        total_count = training_data.count()

        print(
            "Total training interactions:",
            total_count
        )

        # -----------------------------------------------------
        # Label distribution
        # -----------------------------------------------------

        print(
            "\n=== LABEL DISTRIBUTION ==="
        )

        training_data.groupBy(
            "label"
        ).count().show()

        # -----------------------------------------------------
        # Sample records
        # -----------------------------------------------------

        print(
            "\n=== SAMPLE TRAINING DATA ==="
        )

        training_data.show(
            10,
            truncate=False
        )

        print(
            "\n=== NEGATIVE SAMPLING TEST PASSED ==="
        )

    finally:

        spark.stop()