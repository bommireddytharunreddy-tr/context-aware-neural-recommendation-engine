from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed/two_tower_training"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMTrainingDataPreparation")
        .getOrCreate()
    )


def load_transactions(spark):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{DATA_DIR}/transactions_train.csv")
    )


def prepare_training_data(transactions):

    training_data = (
        transactions
        .select(
            "customer_id",
            "article_id",
            "t_dat",
            "price",
            "sales_channel_id"
        )
        .withColumn("label", F.lit(1))
    )

    return training_data


def main():

    spark = create_spark_session()

    transactions = load_transactions(spark)

    training_data = prepare_training_data(transactions)

    print("\n=== TRAINING DATA SAMPLE ===")
    training_data.show(10, truncate=False)

    print("\n=== TRAINING DATA SCHEMA ===")
    training_data.printSchema()

    print("\n=== TRAINING DATA COUNT ===")
    print(training_data.count())

    spark.stop()


if __name__ == "__main__":
    main()