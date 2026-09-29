from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "activity_model.joblib"
DATA = ROOT / "dataset" / "original" / "data tinyMl" / "test.csv"

bundle = joblib.load(MODEL_PATH)
model = bundle["pipeline"]
features = bundle["feature_names"]

df = pd.read_csv(DATA)
sample = df.iloc[[0]].drop(columns=["Activity", "subject"], errors="ignore")
sample = sample.reindex(columns=features)

prediction = model.predict(sample)[0]
print(f"Predicted activity: {prediction}")

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(sample)[0]
    ranked = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )
    print("\nClass probabilities:")
    for label, probability in ranked:
        print(f"  {label:20s} {probability:.4f}")
