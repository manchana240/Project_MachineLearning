## Machine Learning Pipeline

### 1. Data Preprocessing
- Removed 12 duplicate rows
- Handled unknown values across job, marital, education, housing, loan, and default features
- Treated pdays=999 special code as binary indicator
- Capped campaign outliers using IQR method

### 2. Feature Engineering
Six new features were engineered:
- `economic_index` — PCA composite of three correlated macroeconomic indicators
- `age_group` — Age segmentation based on subscription behaviour patterns
- `was_contacted_before` — Binary flag from pdays special code
- `poutcome_success` — Prior campaign success binary indicator
- `is_cellular` — Contact method binary indicator
- `is_risk` — Credit risk binary indicator from default column

### 3. Models Compared

| Model | Category | Tuning Method |
|---|---|---|
| Logistic Regression | Linear | RandomizedSearchCV |
| KNN | Distance-based | RandomizedSearchCV |
| Random Forest | Tree-based | RandomizedSearchCV |
| XGBoost | Ensemble Boosting | RandomizedSearchCV |
| ANN (Keras) | Deep Learning | Early Stopping |

### 4. Best Model
**XGBoost** — selected for deployment based on superior ROC-AUC, 
F1-score, sub-millisecond inference speed, and SHAP-based interpretability.
Decision threshold tuned to 0.30 to prioritise recall on the minority subscriber class.

---

## Getting Started

### Prerequisites
- Python 3.12
- Docker Desktop
- AWS CLI (for cloud deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bank-marketing-prediction.git
cd bank-marketing-prediction

# Install dependencies
pip install -r requirements.txt
```

### Run the API locally with Docker

```bash
# Build the Docker image
docker build -t bank-marketing-prediction .

# Run the container
docker run -p 80:80 bank-marketing-prediction

# Access the API documentation
# Open browser at: http://localhost/docs
```

### Run automated tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — returns model status |
| `/predict` | POST | Returns subscription probability for a customer |

### Sample prediction request

```json
{
  "job": "admin.",
  "marital": "married",
  "education": "university.degree",
  "housing": "yes",
  "loan": "no",
  "month": "may",
  "day_of_week": "mon",
  "campaign": 1,
  "previous": 0,
  "cons_price_idx": 93.994,
  "cons_conf_idx": -36.4,
  "was_contacted_before": 0,
  "economic_index": 0.97,
  "is_risk": 0,
  "poutcome_success": 0,
  "is_cellular": 1,
  "age_group": "working_age"
}
```

### Sample prediction response

```json
{
  "subscription_probability": 0.1823,
  "prediction": "no",
  "recommended_to_call": false,
  "message": "Low probability customer — not recommended"
}
```

---

## Cloud Deployment — AWS

### Architecture
- **Amazon S3** — Data and model artefact storage
- **Amazon ECR** — Docker image registry
- **Amazon SageMaker** — Model training, registry, and endpoint
- **AWS CloudWatch** — Logging, monitoring, and alerting
- **AWS CodePipeline** — CI/CD automation
- **MLflow** — Experiment tracking and model versioning

### MLOps Pipeline
- Data versioning via S3 bucket versioning
- Experiment tracking via MLflow
- CI/CD via GitHub Actions (4-stage pipeline)
- Automated testing via pytest
- Model monitoring via SageMaker Model Monitor
- Automated retraining via EventBridge monthly schedule

---

## Experiment Tracking

MLflow is used for experiment tracking. To view the experiment dashboard:

```bash
cd notebooks
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open browser at: `http://localhost:5000`

---

## Key Results

| Model | ROC-AUC | F1-Score | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.783215 | 0.451229 | 0.455838 |
| KNN | 0.783215 | 0.366957 | 0.455838 |
| Random Forest | 0.783215 | 0.457453 | 0.455838 |
| XGBoost | 0.783215 | 0.472256 | 0.455838 |
| ANN | NaN | 0.461598 | NaN |

---

## Technologies Used

- **Python 3.12**
- **scikit-learn 1.9.0** — preprocessing and classical ML models
- **XGBoost 2.0.3** — gradient boosting
- **TensorFlow/Keras** — ANN deep learning model
- **SHAP** — model explainability
- **MLflow** — experiment tracking
- **FastAPI** — REST API serving
- **Docker** — containerisation
- **AWS S3, ECR, SageMaker** — cloud deployment
- **GitHub Actions** — CI/CD pipeline

---

## References

- Moro, S., Cortez, P. and Rita, P. (2014) 'A data-driven approach to predict 
  the success of bank telemarketing', Decision Support Systems, 62, pp. 22–31.
- UCI Machine Learning Repository: 
  https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## Author
Sachini Manchanayake
Undergraduate, MSc Data Science 
Coventry University / NIBM
Module: NIB 7072 Machine Learning
