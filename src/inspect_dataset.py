from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "dataset" / "original" / "data tinyMl"

for filename in ["test.csv", "accdata.csv", "sessions.csv", "data_subjects_info.csv"]:
    path = base / filename
    print(f"\n--- {filename} ---")
    if filename == "test.csv":
        df = pd.read_csv(path)
        print("shape:", df.shape)
        print("target:", "Activity")
        print("classes:", sorted(df["Activity"].dropna().unique()))
        print("feature count:", len([c for c in df.columns if c not in ("Activity", "subject")]))
    elif filename == "accdata.csv":
        df = pd.read_csv(path)
        print("shape:", df.shape)
        print(df.head())
    else:
        sep = ";" if filename == "sessions.csv" else ","
        df = pd.read_csv(path, sep=sep)
        print("shape:", df.shape)
        print(df.head())
