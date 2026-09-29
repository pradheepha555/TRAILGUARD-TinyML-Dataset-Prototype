# TRAILGUARD TinyML Dataset Prototype

This is a **separate dataset/ML prototype** for TRAILGUARD.

## Important: original dataset preserved

The supplied ZIP was extracted into:

`dataset/original/data tinyMl/`

The original files are kept as supplied. No rows, columns, values, or filenames were rewritten.

Included original files:

- `accdata.csv`
- `accdata.zip`
- `data_subjects_info.csv`
- `PPG_Dataset.csv`
- `sessions.csv`
- `test.csv`
- `train.csv`

## What the supplied dataset actually contains

The supplied `test.csv` is a labeled human-activity dataset with 563 columns. Its target column is `Activity`, with classes including:

- LAYING
- SITTING
- STANDING
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS

The supplied archive also contains accelerometer and PPG-related data.

**This archive does not itself provide a labeled fall class. Therefore this repository does not claim that the supplied dataset trains a fall detector.** The included Python ML code demonstrates activity classification from the supplied `test.csv`.

## ML pipeline

```text
Original dataset
      ↓
Load test.csv
      ↓
Remove Activity + subject from inputs
      ↓
Numeric sensor features
      ↓
Median imputation
      ↓
Standardization
      ↓
Logistic Regression
      ↓
Activity prediction
      ↓
Evaluation
```

## Run

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train:

```powershell
python src/train_model.py
```

Predict one sample:

```powershell
python src/predict.py
```

Evaluate:

```powershell
python src/evaluate.py
```

Inspect the supplied data:

```powershell
python src/inspect_dataset.py
```

## Outputs

After training:

```text
models/activity_model.joblib
results/metrics.txt
```

## Relationship to TRAILGUARD

The activity classifier can serve as an experimental preprocessing component for wearable sensing. It should not be presented as a clinically validated fall detector.

For an actual TRAILGUARD fall-detection model, a dataset containing verified fall and non-fall events is required.

## Prototype limitations

- The supplied dataset is not a clinical dataset.
- Activity classification is not equivalent to fall detection.
- Model performance here is not evidence of real-world medical or emergency accuracy.
- Deployment to nRF52840/TinyML requires model-size, latency, memory, and power validation.
