# FraudShield - Production-Grade MLOps Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![React](https://img.shields.io/badge/Frontend-React+Tailwind-61DAFB?logo=react)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

FraudShield is an end-to-end MLOps project for credit card fraud detection using the IEEE-CIS Fraud Detection dataset. It is designed to demonstrate the full lifecycle of a production-style ML system: data ingestion, validation, feature engineering, model training, experiment tracking, drift simulation, drift detection, automated retraining, API serving, and a live monitoring dashboard.

The dataset is static, so FraudShield includes a controlled drift simulation engine. Instead of pretending that new live data exists, the project deliberately shifts the inference data distribution over time to mimic fraud campaigns, changing transaction amounts, fraud prevalence, behavioral velocity, and categorical patterns. In a real deployment, this simulated feed would be replaced by a streaming source or production transaction database.

**Dataset:** [IEEE-CIS Fraud Detection - Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

Dataset facts verified locally:

- `train_transaction.csv`: 590,540 rows, 394 columns
- `train_identity.csv`: 144,233 rows, 41 columns
- `test_transaction.csv`: 506,691 rows, 393 columns
- `test_identity.csv`: 141,907 rows, 41 columns
- Training fraud rate: approximately 3.5%

## Important Dataset Notes

The IEEE-CIS dataset does not contain clean business columns such as `user_id`, `merchant_id`, `amount`, or `card_type` directly. FraudShield maps the raw schema into production-friendly feature names during ingestion and feature engineering.

Key raw columns:

| Concept | IEEE-CIS source |
|---|---|
| Transaction ID | `TransactionID` |
| Target | `isFraud` |
| Transaction amount | `TransactionAmt` |
| Transaction time | `TransactionDT` |
| Product/category | `ProductCD` |
| Card/network signals | `card1`, `card2`, `card3`, `card4`, `card5`, `card6` |
| Address signals | `addr1`, `addr2` |
| Email domain signals | `P_emaildomain`, `R_emaildomain` |
| Count/time/anonymized signals | `C*`, `D*`, `M*`, `V*` |
| Identity/device signals | `id_*`, `DeviceType`, `DeviceInfo` |

Planned feature proxies:

| Production-style feature | Derived from |
|---|---|
| `amount` | `TransactionAmt` |
| `hour`, `day_of_week`, `is_weekend` | `TransactionDT` converted from relative seconds |
| `card_network` | `card4` |
| `card_type` | `card6` |
| `user_key` | stable combination of card/address fields such as `card1`, `card2`, `card3`, `card5`, `addr1` |
| `merchant_key` | proxy from `ProductCD` plus selected transaction/card/email patterns |
| `velocity_1hr` | rolling count by `user_key` over the previous hour |
| aggregate features | point-in-time rolling aggregates by `user_key` and `merchant_key` |

## Architecture

```text
IEEE-CIS CSV files
        |
        v
Data ingestion and validation
        |
        v
Feature engineering and preprocessing
        |
        v
XGBoost training + MLflow tracking + SHAP evaluation
        |
        v
FastAPI inference service + PostgreSQL logging
        |
        v
Drift detection + drift simulation + retraining pipeline
        |
        v
React dashboard + Grafana monitoring
```

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Language | Python 3.11 | Core pipeline |
| Modeling | XGBoost | Fraud classification on tabular data |
| Explainability | SHAP | Global and per-prediction explanations |
| Experiment tracking | MLflow | Run logging, model artifacts, model registry |
| Data validation | Pandas and custom checks | Schema, null, duplicate, and dtype checks |
| Drift simulation | Custom Python module | Controlled distribution shift on static data |
| Drift detection | Evidently and statistical tests | Numeric and categorical drift reports |
| API | FastAPI | Prediction, metrics, drift, retraining, health endpoints |
| Database | PostgreSQL | Features, predictions, drift logs, retraining logs |
| Monitoring | Grafana Cloud | Operational and model monitoring panels |
| Frontend | React, Tailwind, Recharts | Live fraud monitoring dashboard |
| Orchestration | GitHub Actions | CI/CD and scheduled retraining workflows |
| Containers | Docker Compose | Local development stack |

## Project Structure

```text
fraudshield/
|-- src/
|   |-- data_ingestion.py
|   |-- feature_engineering.py
|   |-- data_preprocessing.py
|   |-- model_training.py
|   |-- model_evaluation.py
|   |-- drift_simulator.py
|   |-- drift_detector.py
|   |-- retraining_pipeline.py
|   `-- api/
|       `-- main.py
|-- scripts/
|   |-- generate_inference_batch.py
|   |-- train.py
|   |-- evaluate.py
|   `-- check_drift.py
|-- dashboard/
|   `-- src/
|       |-- components/
|       `-- App.jsx
|-- .github/
|   `-- workflows/
|       |-- ci.yml
|       |-- deploy.yml
|       `-- retrain.yml
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- inference_batches/
|-- models/
|-- notebooks/
|-- tests/
|-- reports/
|-- logs/
|-- config/
|   `-- config.yaml
|-- docker/
|-- docker-compose.yml
|-- Makefile
|-- requirements.txt
|-- .env.example
`-- README.md
```

Raw Kaggle CSV files are intentionally gitignored. Locally, place them under `data/raw/`:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
data/raw/test_transaction.csv
data/raw/test_identity.csv
data/raw/sample_submission.csv
```

## Pipeline Walkthrough

### 1. Training Pipeline

```text
Raw IEEE-CIS CSV files
  -> validate required columns, dtypes, nulls, duplicates
  -> join transaction and identity tables
  -> derive temporal, card, user-proxy, merchant-proxy, velocity, and aggregate features
  -> split train/validation/test with stratification
  -> apply SMOTE only to the training split
  -> train XGBoost with class imbalance handling
  -> track metrics and artifacts in MLflow
  -> evaluate with precision, recall, F1, ROC-AUC, confusion matrix, and SHAP
  -> register the best model
```

### 2. Serving Pipeline

```text
POST /predict
  -> validate request payload
  -> build or fetch features
  -> load the production model
  -> predict fraud probability
  -> compute SHAP explanation
  -> log prediction to PostgreSQL
  -> return prediction response
```

### 3. Monitoring and Retraining Pipeline

```text
Scheduled GitHub Action
  -> generate drifted inference batch
  -> compare batch against training baseline
  -> save drift report
  -> trigger retraining if drift is detected
  -> compare new model against current production model
  -> promote only if performance improves beyond the configured threshold
  -> log retraining event
```

## Drift Simulation

FraudShield simulates drift so the monitoring and retraining system can be exercised without a live production stream.

Planned shifts:

| Feature area | Shift applied |
|---|---|
| Amount | Increase `TransactionAmt` distribution by configurable intensity |
| Fraud prevalence | Sample batches with higher fraud concentration over time |
| Velocity | Increase rolling transaction counts for selected user proxies |
| Time behavior | Shift fraudulent transactions toward different hours |
| Categorical patterns | Change selected card/product/address distributions |

Configuration example:

```yaml
drift_simulation:
  base_intensity: 0.1
  intensity_increment: 0.2
  max_intensity: 1.0
  drift_threshold: 0.05
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Fraud probability and SHAP explanation |
| `GET` | `/metrics` | Current model F1, AUC, precision, recall |
| `GET` | `/drift` | Latest drift report |
| `GET` | `/drift/history` | Drift detection history |
| `POST` | `/retrain` | Manually trigger retraining |
| `GET` | `/model/info` | Model version, metrics, and promotion history |
| `GET` | `/transactions/recent` | Recent prediction logs |
| `GET` | `/grafana-metrics` | JSON metrics for Grafana |
| `GET` | `/health` | API, database, and model health |

Sample prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "T_12345",
    "transaction_amt": 4200.00,
    "product_cd": "W",
    "card4": "visa",
    "card6": "credit",
    "transaction_dt": 86400
  }'
```

Sample response:

```json
{
  "transaction_id": "T_12345",
  "fraud_probability": 0.87,
  "prediction": "FRAUD",
  "confidence": "HIGH",
  "model_version": "v1",
  "shap_explanation": {
    "transaction_amt": 0.34,
    "velocity_1hr": 0.18,
    "card6_credit": 0.09,
    "product_cd_W": -0.04
  }
}
```

## Database Schema

```sql
CREATE TABLE features (
  transaction_id TEXT PRIMARY KEY,
  amount FLOAT,
  hour INT,
  day_of_week INT,
  is_weekend BOOLEAN,
  user_key TEXT,
  merchant_key TEXT,
  user_tx_count_30d INT,
  user_avg_amount_30d FLOAT,
  merchant_tx_count_30d INT,
  velocity_1hr INT,
  card_network TEXT,
  card_type TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE predictions (
  id SERIAL PRIMARY KEY,
  transaction_id TEXT,
  fraud_probability FLOAT,
  prediction TEXT,
  model_version TEXT,
  shap_values JSONB,
  timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE drift_logs (
  id SERIAL PRIMARY KEY,
  run_date DATE,
  feature_name TEXT,
  drift_score FLOAT,
  p_value FLOAT,
  drift_detected BOOLEAN,
  drift_intensity FLOAT,
  timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE retrain_logs (
  id SERIAL PRIMARY KEY,
  triggered_at TIMESTAMP,
  trigger_reason TEXT,
  old_f1 FLOAT,
  new_f1 FLOAT,
  promoted BOOLEAN,
  notes TEXT,
  timestamp TIMESTAMP DEFAULT NOW()
);
```

## Getting Started

### Prerequisites

- Python 3.11
- Docker and Docker Compose
- Node.js 18+
- Kaggle account and Kaggle API credentials

### Download the Dataset

```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/
```

### Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### Run the Stack

```bash
make up
make train
make drift
make retrain
make test
```

## Planned Model Performance

Initial targets after feature engineering and tuning:

| Metric | Target |
|---|---|
| ROC-AUC | 0.90+ |
| F1 Score | To be established from validation experiments |
| Precision | Tuned based on operating threshold |
| Recall | Tuned based on operating threshold |

Final metrics will be reported from reproducible MLflow runs rather than hard-coded in the README.

## Roadmap

- [ ] Create project structure and config files
- [ ] Implement IEEE-CIS data ingestion and validation
- [ ] Build schema-aware feature engineering
- [ ] Train baseline XGBoost model
- [ ] Add MLflow tracking and model registry
- [ ] Add SHAP global and per-prediction explanations
- [ ] Add drift simulation and drift detection
- [ ] Build retraining pipeline
- [ ] Build FastAPI service
- [ ] Build React dashboard
- [ ] Add Docker Compose local stack
- [ ] Add CI/CD workflows
- [ ] Deploy API and dashboard

## Author

Devansh Gupta  
B.Tech Information Technology - DJSCE Mumbai (2027)

## License

MIT
