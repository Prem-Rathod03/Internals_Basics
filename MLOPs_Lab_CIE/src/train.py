"""
train.py - experiment tracking with mlflow
trains LinearRegression and GradientBoosting, logs to mlflow, picks best one
"""

import json
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXPERIMENT_NAME = "audittrail-audit-completion-days"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["audit_completion_days"])
    y = df["audit_completion_days"]
    return X, y


def train_and_log(model, model_name, X_train, X_test, y_train, y_test):
    """train a single model and log everything to mlflow"""
    with mlflow.start_run(run_name=model_name):
        params = model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("priority", "high")

        # log model artifact to mlflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id

        print(f"{model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, run_id={run_id}")

        return {
            "name": model_name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "run_id": run_id,
            "model": model,
        }


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    models = [
        (LinearRegression(), "LinearRegression"),
        (GradientBoostingRegressor(random_state=42), "GradientBoosting"),
    ]

    results = []
    for model, name in models:
        result = train_and_log(model, name, X_train, X_test, y_train, y_test)
        results.append(result)

    # pick best by MAE
    best = min(results, key=lambda r: r["mae"])

    # save best model
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best["model"], best_model_path)
    print(f"\nBest model saved to {best_model_path}")

    # save metadata so other scripts can use it
    meta = {
        "best_model_name": best["name"],
        "best_run_id": best["run_id"],
        "best_mae": best["mae"],
        "best_rmse": best["rmse"],
    }
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # save results
    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"]} for r in results],
        "best_model": best["name"],
        "best_metric_name": "mae",
        "best_metric_value": best["mae"],
    }

    output_path = os.path.join(RESULTS_DIR, "step1_s1.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
