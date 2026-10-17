# Heart Disease Prediction — MLOps Assignment 01

**Course:** Machine Learning Operations (MLOps), AIMLCZG523  
**Project:** Reproducible training, experiment tracking, API serving, CI, containerization and monitoring.

> This is an educational demonstration, not a medical device. Predictions must not be used for diagnosis or treatment.

## Project layout

- `src/data.py` downloads and validates the UCI Heart Disease Cleveland data.
- `src/train.py` runs stratified cross-validation, compares Logistic Regression and Random Forest, logs MLflow runs, and saves the selected pipeline.
- `src/api.py` serves `/health`, `/predict`, and `/metrics` with FastAPI.
- `notebooks/eda_and_inference.ipynb` holds the EDA and an inference example.
- `monitoring/` has Prometheus config and a provisioned Grafana dashboard.
- `tests/` covers data validation and API input handling.
- `.github/workflows/ci.yml` runs lint, tests, and a training smoke test.
- `Dockerfile` builds an isolated API image; `docker-compose.yml` starts the API and Prometheus.
- `k8s/` includes Kubernetes Deployment and Service manifests.
- `report.md` is a report draft. Replace all marked evidence placeholders with results/screenshots from your own run before submission.

## Google Colab quick start

For the easiest model-training workflow, upload `notebooks/colab_end_to_end.ipynb` to Google Colab and choose **Runtime → Run all**. It installs the notebook dependencies, downloads the UCI Cleveland data (with an upload fallback), runs EDA, compares Logistic Regression and Random Forest, logs experiments to local MLflow files, evaluates the selected pipeline, and exports model/evidence artifacts as a downloadable ZIP.

The notebook makes model development easier, but it does not itself prove Docker, GitHub Actions, Kubernetes, or a public deployment. Those tasks still need to be executed and evidenced separately using the project files below.

## 1. Setup

Python 3.11 is recommended (tested locally on 3.10 and in Docker on 3.11). Dependencies are pinned in `requirements.txt` because the pickled model must be loaded with the same scikit-learn version that trained it.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download and inspect the dataset

```bash
python -m src.data --output data/heart.csv
```

The script fetches the UCI processed Cleveland file. The target is converted to binary: `0` means no disease in this dataset and values `1–4` are grouped as `1` (disease present). The source uses `?` for missing values; these are treated as missing, not as valid measurements.

## 3. Train and track experiments

```bash
python -m src.train --data data/heart.csv --output models/heart_pipeline.joblib --cv 5
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the MLflow UI at `http://127.0.0.1:5000`. The training script logs parameters, cross-validation metrics, held-out test metrics, confusion matrix, ROC curve, and the fitted pipeline. Results are intentionally calculated when you run the project rather than hard-coded into this repository.

## 4. Run tests and API locally

```bash
pytest -q
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Check `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`. Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":3}'
```

The API expects the 13 Cleveland input features. It returns the predicted class and model probability for the positive class. If the model file does not exist, the API returns a clear 503 error rather than silently training at startup.

## 5. Docker

```bash
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```

Train the model first: the Dockerfile copies `models/` into the image. If host port 8000 is busy use `-p 8010:8000`.

```bash
API_PORT=8000 docker compose up -d --build   # API :API_PORT, Prometheus :9090, Grafana :3000 (admin/admin)
```

Grafana auto-loads the "Heart Disease API" dashboard (request rate, p95 latency, totals, errors). Raw metrics are at `/metrics`.

## 6. Kubernetes (local cluster)

Build the image in the cluster's image environment (with Rancher Desktop/moby `docker build` is enough; model is baked into the image). **Always pin the context so you do not hit another cluster:**

```bash
kubectl --context rancher-desktop apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl --context rancher-desktop get pods,svc
curl localhost:8080/health   # LoadBalancer on port 8080
```

The included service is `LoadBalancer` (port 8080) which Rancher Desktop/Docker Desktop expose on localhost; on Minikube run `minikube tunnel`. For a public cloud, configure an approved LoadBalancer/Ingress, registry image, TLS, secrets, resource limits, and access controls. Do not expose a patient-data endpoint publicly without a security review.

## 7. Evidence and submission checklist

- [ ] Run the download, EDA/training, and test commands successfully.
- [ ] Review MLflow runs and save screenshots to `screenshots/`.
- [ ] Capture API `/docs`, successful `/predict`, Docker build/run, CI workflow, and Kubernetes status screenshots.
- [ ] Replace placeholders in `report.md` with your measured metrics and actual screenshots.
- [ ] Add your repository URL and actual deployed URL (or local access instructions).
- [ ] Record a short video showing data download → training/MLflow → tests/CI → Docker/API → monitoring/deployment.
- [ ] Be ready to explain the target mapping, leakage prevention, cross-validation, model choice, and limitations.

## Reproducibility notes

A fixed random seed is used where applicable. Preprocessing is fitted inside an sklearn `Pipeline` and `ColumnTransformer`, so imputers and scaling are trained only on each training fold. A final untouched test split is used for the final comparison. The Cleveland subset is relatively small and should not be interpreted as representative of all populations.
# MLOPS-Assignment-1-2025AE05075
