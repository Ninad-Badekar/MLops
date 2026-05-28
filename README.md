# Water Potability Prediction MLOps Pipeline

MLOps pipeline for predicting water potability using DVC and a FastAPI serving layer.

## Pipeline

| Stage | Description |
|---|---|
| `data_collection` | Split raw CSV into train/test sets |
| `pre_processing` | Fill missing values with median |
| `model_building` | Train a RandomForestClassifier |
| `model_evaluation` | Evaluate accuracy, precision, recall, F1 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

## Run pipeline

```bash
dvc repro
```

## Serve API

```bash
cd src && uvicorn main:app --reload
```

## Project structure

```
.
├── .dvc/          # DVC internal config
├── data/
│   ├── raw/       # Raw train/test splits (DVC-tracked)
│   └── processed/ # Processed data (DVC-tracked)
├── src/           # Python source code
├── dvc.yaml       # DVC pipeline definition
├── dvc.lock       # DVC pipeline lock file
└── model.pkl      # Trained model (DVC-tracked)
```
