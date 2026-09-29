from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "original" / "data tinyMl" / "test.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
target = "Activity"
if target not in df.columns:
    raise ValueError(f"Expected '{target}' column was not found.")

# The original dataset contains a subject identifier; keep it out of the classifier.
X = df.drop(columns=[target, "subject"], errors="ignore")
y = df[target].astype(str)

# Keep numeric sensor-derived features only.
X = X.select_dtypes(include=["number"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, n_jobs=None))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, pred)
report = classification_report(y_test, pred)

joblib.dump(
    {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "target": target,
        "classes": sorted(y.unique().tolist()),
        "dataset_file": str(DATA.relative_to(ROOT)),
        "note": "Prototype activity classifier trained from the supplied dataset."
    },
    MODEL_DIR / "activity_model.joblib"
)

(RESULTS_DIR / "metrics.txt").write_text(
    f"Accuracy: {accuracy:.4f}\n\n{report}",
    encoding="utf-8"
)

print(f"Dataset rows: {len(df)}")
print(f"Numeric features: {X.shape[1]}")
print(f"Classes: {sorted(y.unique())}")
print(f"Holdout accuracy: {accuracy:.4f}")
print(f"Saved: {MODEL_DIR / 'activity_model.joblib'}")
