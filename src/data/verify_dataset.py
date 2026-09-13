from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

files = [
    "articles.csv",
    "customers.csv",
    "transactions_train.csv",
]

for file in files:
    path = DATA_DIR / file

    print(f"\nChecking: {file}")

    if not path.exists():
        print("❌ File not found")
        continue

    df = pd.read_csv(path, nrows=5)

    print(f"✅ File found")
    print(f"Rows sampled: {len(df)}")
    print(f"Columns: {list(df.columns)}")