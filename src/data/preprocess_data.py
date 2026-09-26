from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMDataPreprocessing")
        .getOrCreate()
    )


def load_raw_datasets(spark):
    articles = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{DATA_DIR}/articles.csv")
    )

    customers = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{DATA_DIR}/customers.csv")
    )

    transactions = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{DATA_DIR}/transactions_train.csv")
    )

    return articles, customers, transactions


def preprocess_articles(articles):
    cleaned = (
        articles
        .dropDuplicates(["article_id"])
        .filter(F.col("article_id").isNotNull())
        .filter(F.col("product_code").isNotNull())
    )

    return cleaned


def preprocess_customers(customers):
    cleaned = (
        customers
        .dropDuplicates(["customer_id"])
        .filter(F.col("customer_id").isNotNull())
    )

    return cleaned


def preprocess_transactions(transactions):
    cleaned = (
        transactions
        .dropDuplicates()
        .filter(F.col("customer_id").isNotNull())
        .filter(F.col("article_id").isNotNull())
        .filter(F.col("price").isNotNull())
        .filter(F.col("price") > 0)
        .withColumn(
            "t_dat",
            F.to_date(F.col("t_dat"))
        )
        .filter(F.col("t_dat").isNotNull())
    )

    return cleaned


def print_dataset_statistics(
    articles,
    customers,
    transactions
):
    print("\n=== PREPROCESSED DATASET STATISTICS ===")

    print("\nArticles:")
    print("Rows:", articles.count())
    print("Columns:", len(articles.columns))

    print("\nCustomers:")
    print("Rows:", customers.count())
    print("Columns:", len(customers.columns))

    print("\nTransactions:")
    print("Rows:", transactions.count())
    print("Columns:", len(transactions.columns))

    print("\nTransaction date range:")
    transactions.select(
        F.min("t_dat").alias("min_date"),
        F.max("t_dat").alias("max_date")
    ).show()


def preprocess_datasets(spark):
    articles, customers, transactions = load_raw_datasets(spark)

    print("\n=== RAW DATA LOADED ===")

    articles_clean = preprocess_articles(articles)
    customers_clean = preprocess_customers(customers)
    transactions_clean = preprocess_transactions(transactions)

    print("\n=== DATA PREPROCESSING COMPLETED ===")

    print_dataset_statistics(
        articles_clean,
        customers_clean,
        transactions_clean
    )

    return (
        articles_clean,
        customers_clean,
        transactions_clean
    )


if __name__ == "__main__":
    spark = create_spark_session()

    try:
        preprocess_datasets(spark)
        print("\n=== PREPROCESSING TEST PASSED ===")
    finally:
        spark.stop()