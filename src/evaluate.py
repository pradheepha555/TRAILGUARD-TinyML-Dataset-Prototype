from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "activity_model.joblib"
DATA = ROOT / "dataset" / "original" / "data tinyMl" / "test.csv"

bundle = joblib.load(MODEL_PATH)
model = bundle["pipeline"]
features = bundle["feature_names"]

df = pd.read_csv(DATA)
X = df.drop(columns=["Activity", "subject"], errors="ignore")
X = X.reindex(columns=features)
y = df["Activity"].astype(str)

pred = model.predict(X)

print(classification_report(y, pred))
print("Confusion matrix:")
print(confusion_matrix(y, pred, labels=model.classes_))
