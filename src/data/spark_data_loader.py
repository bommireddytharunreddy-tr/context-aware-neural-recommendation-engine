from pyspark.sql import SparkSession

DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMRecommendationEngine")
        .getOrCreate()
    )


def load_datasets(spark):
    articles = spark.read.option("header", True).option("inferSchema", True).csv(
        f"{DATA_DIR}/articles.csv"
    )

    customers = spark.read.option("header", True).option("inferSchema", True).csv(
        f"{DATA_DIR}/customers.csv"
    )

    transactions = spark.read.option("header", True).option("inferSchema", True).csv(
        f"{DATA_DIR}/transactions_train.csv"
    )

    return articles, customers, transactions


if __name__ == "__main__":
    spark = create_spark_session()

    articles, customers, transactions = load_datasets(spark)

    print("\n=== ARTICLES ===")
    articles.printSchema()

    print("\n=== CUSTOMERS ===")
    customers.printSchema()

    print("\n=== TRANSACTIONS ===")
    transactions.printSchema()

    spark.stop()