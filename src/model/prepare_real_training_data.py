from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.model.negative_sampling import (
    create_positive_interactions,
    create_negative_interactions
)

from src.model.temporal_split import (
    create_temporal_split
)

from src.model.create_id_mappings import (
    create_user_mapping,
    create_item_mapping
)

from src.features.training_context_features import (
    create_training_context_features
)


DATA_DIR = "data/raw"


def create_spark_session():

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMRealTwoTowerTrainingData")
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


def create_train_interactions(train):

    print(
        "\n=== CREATING POSITIVE INTERACTIONS ==="
    )

    positive = create_positive_interactions(
        train
    )

    positive_count = positive.count()

    print(
        "Positive interactions:",
        positive_count
    )

    print(
        "\n=== CREATING NEGATIVE INTERACTIONS ==="
    )

    negative = create_negative_interactions(
        train,
        negative_ratio=1
    )

    negative_count = negative.count()

    print(
        "Negative interactions:",
        negative_count
    )

    training_interactions = (
        positive.unionByName(negative)
    )

    return training_interactions


def add_id_mappings(
    training_interactions,
    user_mapping,
    item_mapping
):

    print(
        "\n=== ADDING USER MAPPING ==="
    )

    data = (
        training_interactions
        .join(
            user_mapping,
            on="customer_id",
            how="inner"
        )
    )

    print(
        "\n=== ADDING ITEM MAPPING ==="
    )

    data = (
        data
        .join(
            item_mapping,
            on="article_id",
            how="inner"
        )
    )

    return data


def add_context_features(
    training_data,
    train
):

    print(
        "\n=== CREATING USER AND ITEM CONTEXT FEATURES ==="
    )

    user_features, item_features = (
        create_training_context_features(
            train
        )
    )

    print(
        "\n=== JOINING USER CONTEXT FEATURES ==="
    )

    training_data = (
        training_data
        .join(
            user_features,
            on="customer_id",
            how="left"
        )
    )

    print(
        "\n=== JOINING ITEM CONTEXT FEATURES ==="
    )

    training_data = (
        training_data
        .join(
            item_features,
            on="article_id",
            how="left"
        )
    )

    return training_data


def check_context_nulls(training_data):

    print(
        "\n=== CHECKING CONTEXT NULL VALUES ==="
    )

    null_counts = (
        training_data
        .select(
            F.sum(
                F.when(
                    F.col(
                        "customer_total_purchases"
                    ).isNull(),
                    1
                ).otherwise(0)
            ).alias(
                "null_customer_total_purchases"
            ),

            F.sum(
                F.when(
                    F.col(
                        "customer_avg_price"
                    ).isNull(),
                    1
                ).otherwise(0)
            ).alias(
                "null_customer_avg_price"
            ),

            F.sum(
                F.when(
                    F.col(
                        "item_total_purchases"
                    ).isNull(),
                    1
                ).otherwise(0)
            ).alias(
                "null_item_total_purchases"
            ),

            F.sum(
                F.when(
                    F.col(
                        "item_avg_price"
                    ).isNull(),
                    1
                ).otherwise(0)
            ).alias(
                "null_item_avg_price"
            )
        )
    )

    null_counts.show()

    return null_counts


def select_model_features(training_data):

    return (
        training_data
        .select(
            "user_index",
            "item_index",

            "customer_total_purchases",
            "customer_avg_price",

            "item_total_purchases",
            "item_avg_price",

            "label"
        )
    )


def main():

    print(
        "\n=========================================="
    )

    print(
        "REAL TWO-TOWER TRAINING DATA"
    )

    print(
        "=========================================="
    )

    spark = create_spark_session()

    try:

        # --------------------------------------------------
        # 1. Load transactions
        # --------------------------------------------------

        print(
            "\n=== LOADING H&M TRANSACTIONS ==="
        )

        transactions = load_transactions(
            spark
        )

        # --------------------------------------------------
        # 2. Temporal split
        # --------------------------------------------------

        print(
            "\n=== CREATING TEMPORAL SPLIT ==="
        )

        train, validation, test = (
            create_temporal_split(
                transactions
            )
        )

        # --------------------------------------------------
        # 3. Positive + negative interactions
        # --------------------------------------------------

        training_interactions = (
            create_train_interactions(
                train
            )
        )

        # --------------------------------------------------
        # 4. Create ID mappings
        # --------------------------------------------------

        print(
            "\n=== CREATING USER MAPPING ==="
        )

        user_mapping = create_user_mapping(
            transactions
        )

        print(
            "\n=== CREATING ITEM MAPPING ==="
        )

        item_mapping = create_item_mapping(
            transactions
        )

        # --------------------------------------------------
        # 5. Add user/item IDs
        # --------------------------------------------------

        training_data = add_id_mappings(
            training_interactions,
            user_mapping,
            item_mapping
        )

        # --------------------------------------------------
        # 6. Add context features
        # --------------------------------------------------

        training_data = add_context_features(
            training_data,
            train
        )

        # --------------------------------------------------
        # 7. Check NULL values
        # --------------------------------------------------

        check_context_nulls(
            training_data
        )

        # --------------------------------------------------
        # 8. Select final model features
        # --------------------------------------------------

        training_data = select_model_features(
            training_data
        )

        # --------------------------------------------------
        # 9. Final schema
        # --------------------------------------------------

        print(
            "\n=== FINAL TWO-TOWER TRAINING SCHEMA ==="
        )

        training_data.printSchema()

        # --------------------------------------------------
        # 10. Sample
        # --------------------------------------------------

        print(
            "\n=== FINAL TRAINING SAMPLE ==="
        )

        training_data.show(
            10,
            truncate=False
        )

        # --------------------------------------------------
        # 11. Label distribution
        # --------------------------------------------------

        print(
            "\n=== LABEL DISTRIBUTION ==="
        )

        training_data.groupBy(
            "label"
        ).count().show()

        # --------------------------------------------------
        # 12. Context statistics
        # --------------------------------------------------

        print(
            "\n=== CONTEXT FEATURE STATISTICS ==="
        )

        training_data.select(
            "customer_total_purchases",
            "customer_avg_price",
            "item_total_purchases",
            "item_avg_price"
        ).describe().show()

        # --------------------------------------------------
        # 13. Final test
        # --------------------------------------------------

        print(
            "\n=== REAL TWO-TOWER INPUT PIPELINE TEST PASSED ==="
        )

    finally:

        spark.stop()


if __name__ == "__main__":

    main()