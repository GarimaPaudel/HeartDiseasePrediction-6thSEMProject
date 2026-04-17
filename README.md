# Heart Disease Prediction

A machine learning project to predict the presence or absence of heart disease using clinical parameters. Built with a full MLOps pipeline — from data ingestion to model registration — and served via a FastAPI REST API.

---

## Project Structure

```
├── app.py                  # FastAPI prediction server
├── run_pipeline.py         # CLI to run the ML pipeline
├── config/
│   ├── config.yaml         # Paths and pipeline config
│   └── params.yaml         # Model hyperparameter grids
├── src/
│   ├── components/         # Pipeline stage implementations
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluator.py
│   │   └── model_registry.py
│   ├── pipeline/           # Stage orchestration scripts
│   └── utils/              # Logging, YAML/JSON/pickle helpers
├── notebooks/              # Exploratory analysis notebooks
├── artifacts/
│   ├── data/raw/           # Ingested raw data
│   ├── data/processed/     # Preprocessed + feature-engineered data
│   ├── models/registry/    # Versioned model files (v1–v5)
│   ├── reports/            # Evaluation reports
│   └── logs/               # Pipeline run logs
└── mlruns/                 # MLflow experiment tracking
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML | scikit-learn (DecisionTree, RandomForest, KNN, LogisticRegression) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Config | YAML + python-box |
| Logging | Loguru |
| Dependency Management | uv |

---

## Setup

**Prerequisites:** Python 3.13+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/GarimaPaudel/HeartDiseasePrediction-6thSEMProject.git
cd HeartDiseasePrediction-6thSEMProject

# Install dependencies
uv sync
```

---

## Running the Pipeline

The pipeline has 6 stages: ingestion → preprocessing → feature engineering → training → evaluation → model registration.

```bash
# Run the full pipeline (train + register model)
uv run python run_pipeline.py run-all

# Run a specific stage only
uv run python run_pipeline.py run-stage --stage ingestion
uv run python run_pipeline.py run-stage --stage preprocessing
uv run python run_pipeline.py run-stage --stage feature-engineering
uv run python run_pipeline.py run-stage --stage training
uv run python run_pipeline.py run-stage --stage evaluation
uv run python run_pipeline.py run-stage --stage registration

# List all registered model versions
uv run python run_pipeline.py list-models
```

To change the active model, edit `config/params.yaml`:
```yaml
active_model: "DecisionTree"   # Options: DecisionTree, RandomForest, LogisticRegression, KNN
```

---

## Starting the API

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check if API and model are loaded |
| POST | `/predict` | Get a heart disease prediction |

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 50,
    "gender": "male",
    "height": 170,
    "weight": 80,
    "ap_hi": 120,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": false,
    "alco": false,
    "active": true
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "label": "Absence of Heart Disease"
}
```

---

## Input Parameters

| Field | Type | Description |
|---|---|---|
| `age` | float | Age in years |
| `gender` | string | `"male"` or `"female"` |
| `height` | float | Height in cm |
| `weight` | float | Weight in kg |
| `ap_hi` | int | Systolic blood pressure |
| `ap_lo` | int | Diastolic blood pressure |
| `cholesterol` | int | 1 = normal, 2 = above normal, 3 = well above normal |
| `gluc` | int | 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke` | bool | Smoker |
| `alco` | bool | Alcohol intake |
| `active` | bool | Physically active |

---

## Model Performance

Four algorithms were trained and compared — DecisionTree, RandomForest, LogisticRegression, and KNN — each tuned via GridSearchCV with 5-fold StratifiedKFold cross-validation on 70,000 patient records from the [Kaggle Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).

| Model | Accuracy | Sensitivity | Specificity | Objective |
|---|---|---|---|---|
| DecisionTree | 73.26% | 63.61% | 82.70% | 0.8917 |
| **RandomForest** | **73.55%** | **69.35%** | 77.57% | **0.9049** |
| LogisticRegression | 73.14% | 67.72% | 78.44% | 0.9007 |
| KNN | 71.39% | 68.96% | 73.77% | 0.8863 |

> **Objective** = `accuracy + 0.25 × sensitivity` — sensitivity is weighted higher because false negatives (missing a sick patient) are more costly in a medical context.

`artifacts/reports/evaluation_report.json` always stores the **best-performing model's results** by objective value. It is only overwritten when a new run produces a higher objective score.

---

## Experiment Tracking

MLflow is used to track all training runs. To view the MLflow UI:

```bash
uv run mlflow ui --backend-store-uri mlruns
```

Then open `http://localhost:5000` in your browser.
