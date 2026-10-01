from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


DATA_DIR = "data/raw"


def create_spark_session():

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMCreateIDMappings")
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


def create_user_mapping(transactions):

    window = Window.orderBy(
        "customer_id"
    )

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

    window = Window.orderBy(
        "article_id"
    )

    return (
        transactions
        .select("article_id")
        .distinct()
        .withColumn(
            "item_index",
            F.row_number().over(window) - 1
        )
    )


def main():

    spark = create_spark_session()

    try:

        transactions = load_transactions(
            spark
        )

        user_mapping = create_user_mapping(
            transactions
        )

        item_mapping = create_item_mapping(
            transactions
        )

        print(
            "\n=== USER MAPPING SAMPLE ==="
        )

        user_mapping.show(
            10,
            truncate=False
        )

        print(
            "\n=== ITEM MAPPING SAMPLE ==="
        )

        item_mapping.show(
            10,
            truncate=False
        )

        print(
            "\n=== UNIQUE USERS ==="
        )

        print(
            user_mapping.count()
        )

        print(
            "\n=== UNIQUE ITEMS ==="
        )

        print(
            item_mapping.count()
        )

    finally:

        spark.stop()


if __name__ == "__main__":

    main()