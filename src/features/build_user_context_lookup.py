import os

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

TRANSACTIONS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "transactions_train.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "trained_models"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "user_context.npz"
)

CHUNK_SIZE = 500_000


# ============================================================
# BUILD USER CONTEXT
# ============================================================

def build_user_context():

    print("\n==========================================")
    print("BUILD USER CONTEXT LOOKUP")
    print("==========================================")

    if not os.path.exists(TRANSACTIONS_PATH):
        raise FileNotFoundError(
            f"Transactions file not found: {TRANSACTIONS_PATH}"
        )

    user_purchase_counts = {}
    user_price_sums = {}

    total_rows = 0

    print("\nReading transactions in chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TRANSACTIONS_PATH,
            usecols=[
                "customer_id",
                "price"
            ],
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        chunk["price"] = pd.to_numeric(
            chunk["price"],
            errors="coerce"
        )

        chunk = chunk.dropna(
            subset=[
                "customer_id",
                "price"
            ]
        )

        grouped = (
            chunk
            .groupby("customer_id")["price"]
            .agg(
                purchase_count="count",
                price_sum="sum"
            )
        )

        for customer_id, row in grouped.iterrows():

            user_purchase_counts[customer_id] = (
                user_purchase_counts.get(
                    customer_id,
                    0
                )
                + int(row["purchase_count"])
            )

            user_price_sums[customer_id] = (
                user_price_sums.get(
                    customer_id,
                    0.0
                )
                + float(row["price_sum"])
            )

        total_rows += len(chunk)

        print(
            f"Chunk {chunk_number}: "
            f"{total_rows:,} rows processed"
        )

    # --------------------------------------------------------
    # Sort customers exactly like create_id_mappings.py
    # --------------------------------------------------------

    customer_ids = sorted(
        user_purchase_counts.keys()
    )

    user_indices = np.arange(
        len(customer_ids),
        dtype=np.int32
    )

    purchase_counts = np.array(
        [
            user_purchase_counts[customer_id]
            for customer_id in customer_ids
        ],
        dtype=np.float32
    )

    price_sums = np.array(
        [
            user_price_sums[customer_id]
            for customer_id in customer_ids
        ],
        dtype=np.float64
    )

    average_prices = (
        price_sums
        / purchase_counts
    )

    average_prices = average_prices.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected_users = 1_362_281

    print("\n=== USER CONTEXT SUMMARY ===")
    print("Users:", len(customer_ids))
    print(
        "User indices:",
        user_indices.shape
    )
    print(
        "Purchase counts:",
        purchase_counts.shape
    )
    print(
        "Average prices:",
        average_prices.shape
    )

    if len(customer_ids) != expected_users:
        raise ValueError(
            "Unexpected user count: "
            f"{len(customer_ids)}"
        )

    if not np.array_equal(
        user_indices,
        np.arange(expected_users)
    ):
        raise ValueError(
            "User indices are not sequential."
        )

    if np.any(purchase_counts <= 0):
        raise ValueError(
            "Invalid purchase count detected."
        )

    if not np.all(
        np.isfinite(average_prices)
    ):
        raise ValueError(
            "Invalid average price detected."
        )

    # --------------------------------------------------------
    # Save lookup
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        customer_ids=np.array(
            customer_ids
        ),
        user_indices=user_indices,
        customer_total_purchases=purchase_counts,
        customer_avg_price=average_prices
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\n=== USER CONTEXT LOOKUP PASSED ===")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    build_user_context()