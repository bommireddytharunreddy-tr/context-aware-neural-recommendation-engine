from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMCreateIDMappings")
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
    return (
        transactions
        .select("customer_id")
        .distinct()
        .orderBy("customer_id")
        .withColumn(
            "user_index",
            F.row_number().over(
                __import__("pyspark").sql.Window.orderBy("customer_id")
            ) - 1
        )
    )


def create_item_mapping(transactions):
    return (
        transactions
        .select("article_id")
        .distinct()
        .orderBy("article_id")
        .withColumn(
            "item_index",
            F.row_number().over(
                __import__("pyspark").sql.Window.orderBy("article_id")
            ) - 1
        )
    )


def main():
    spark = create_spark_session()

    transactions = load_transactions(spark)

    user_mapping = create_user_mapping(transactions)
    item_mapping = create_item_mapping(transactions)

    print("\n=== USER MAPPING SAMPLE ===")
    user_mapping.show(10, truncate=False)

    print("\n=== ITEM MAPPING SAMPLE ===")
    item_mapping.show(10, truncate=False)

    print("\n=== UNIQUE USERS ===")
    print(user_mapping.count())

    print("\n=== UNIQUE ITEMS ===")
    print(item_mapping.count())

    spark.stop()


if __name__ == "__main__":
    main()