from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_DIR = "data/raw"


def create_spark_session():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HMTemporalSplit")
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


def create_temporal_split(transactions):

    min_date = (
        transactions
        .select(F.min("t_dat"))
        .first()[0]
    )

    max_date = (
        transactions
        .select(F.max("t_dat"))
        .first()[0]
    )

    print("\nMinimum transaction date:", min_date)
    print("Maximum transaction date:", max_date)

    total_days = (
        max_date - min_date
    ).days

    train_days = int(total_days * 0.70)
    validation_days = int(total_days * 0.15)

    train_end = (
        min_date
        + __import__("datetime").timedelta(
            days=train_days
        )
    )

    validation_end = (
        train_end
        + __import__("datetime").timedelta(
            days=validation_days
        )
    )

    print("\n=== TEMPORAL SPLIT DATES ===")
    print("Train end:", train_end)
    print("Validation end:", validation_end)
    print("Test end:", max_date)

    train = (
        transactions
        .filter(
            F.col("t_dat") < F.lit(train_end)
        )
    )

    validation = (
        transactions
        .filter(
            (F.col("t_dat") >= F.lit(train_end))
            &
            (F.col("t_dat") < F.lit(validation_end))
        )
    )

    test = (
        transactions
        .filter(
            F.col("t_dat") >= F.lit(validation_end)
        )
    )

    return train, validation, test


def print_split_statistics(
    train,
    validation,
    test
):

    print("\n=== TRAINING SET ===")
    print("Rows:", train.count())

    train.select(
        F.min("t_dat").alias("min_date"),
        F.max("t_dat").alias("max_date")
    ).show()

    print("\n=== VALIDATION SET ===")
    print("Rows:", validation.count())

    validation.select(
        F.min("t_dat").alias("min_date"),
        F.max("t_dat").alias("max_date")
    ).show()

    print("\n=== TEST SET ===")
    print("Rows:", test.count())

    test.select(
        F.min("t_dat").alias("min_date"),
        F.max("t_dat").alias("max_date")
    ).show()


if __name__ == "__main__":

    spark = create_spark_session()

    try:

        transactions = load_transactions(
            spark
        )

        train, validation, test = (
            create_temporal_split(
                transactions
            )
        )

        print_split_statistics(
            train,
            validation,
            test
        )

        print(
            "\n=== TEMPORAL SPLIT TEST PASSED ==="
        )

    finally:

        spark.stop()