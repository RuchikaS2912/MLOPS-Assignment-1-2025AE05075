"""Train and compare reproducible classification pipelines with MLflow tracking."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data import FEATURES, load_dataset

SEED = 42
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}


def make_pipeline(model):
    prep = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                FEATURES,
            )
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", prep), ("classifier", model)])


def save_eda(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for col in FEATURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")
        fig.tight_layout()
        fig.savefig(out / f"hist_{col}.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title("Feature and target correlation")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="target", ax=ax)
    ax.set_title("Target class balance (0=no disease, 1=disease)")
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=140)
    plt.close(fig)


def evaluate(y_true, pred, proba):
    return {
        "test_accuracy": float(accuracy_score(y_true, pred)),
        "test_precision": float(precision_score(y_true, pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_true, proba)),
    }


def main():
    import mlflow
    import mlflow.sklearn

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--output", default="models/heart_pipeline.joblib")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    df = pd.read_csv(args.data) if Path(args.data).exists() else load_dataset()
    missing = set(FEATURES + ["target"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    save_eda(df, Path("artifacts/eda"))
    X, y = df[FEATURES], df["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("heart-disease-classification")
    candidates = {
        "logistic_regression": (
            make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=SEED)),
            {"classifier__n_estimators": [150, 300], "classifier__max_depth": [None, 5, 10]},
        ),
    }
    results = []
    for name, (pipe, params) in candidates.items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(pipe, params, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            cv_scores = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
            pred = best.predict(X_test)
            proba = best.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, pred, proba)
            mlflow.log_param("model", name)
            mlflow.log_param("cv_folds", args.cv)
            mlflow.log_param("seed", SEED)
            mlflow.log_params({f"best_{k}": str(v) for k, v in search.best_params_.items()})
            mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
            for metric, vals in cv_scores.items():
                mlflow.log_metric(f"cv_{metric}_mean", float(vals.mean()))
                mlflow.log_metric(f"cv_{metric}_std", float(vals.std()))
            mlflow.log_metrics(metrics)
            run_dir = Path("artifacts") / name
            run_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
            fig.tight_layout()
            cm_path = run_dir / "confusion_matrix.png"
            fig.savefig(cm_path, dpi=140)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
            fig.tight_layout()
            roc_path = run_dir / "roc_curve.png"
            fig.savefig(roc_path, dpi=140)
            plt.close(fig)
            report = classification_report(y_test, pred, output_dict=True, zero_division=0)
            (run_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(run_dir / "classification_report.json"))
            mlflow.sklearn.log_model(best, artifact_path="model", serialization_format="cloudpickle")
            results.append(
                {
                    "name": name,
                    "pipeline": best,
                    "cv_auc": float(search.best_score_),
                    "metrics": metrics,
                    "params": search.best_params_,
                }
            )
    winner = max(results, key=lambda x: x["cv_auc"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner["pipeline"], out)
    summary = {
        "selected_model": winner["name"],
        "selection_basis": "highest mean cross-validated ROC-AUC on training split",
        "best_params": winner["params"],
        "test_metrics": winner["metrics"],
        "seed": SEED,
        "feature_order": FEATURES,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
