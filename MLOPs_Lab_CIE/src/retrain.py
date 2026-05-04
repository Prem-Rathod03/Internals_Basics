"""
retrain.py - retraining pipeline
merges old + new data, retrains the best model type,
compares with the champion, promotes if it's better
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
TRAINING_DATA = os.path.join(BASE_DIR, "data", "training_data.csv")
NEW_DATA = os.path.join(BASE_DIR, "data", "new_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
META_PATH = os.path.join(MODELS_DIR, "best_model_meta.json")
EXPERIMENT_NAME = "audittrail-audit-completion-days"

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_model(name):
    """returns a fresh model instance based on the name"""
    if name == "LinearRegression":
        return LinearRegression()
    elif name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    # load info about current best model
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    champion_name = meta["best_model_name"]
    champion_mae = meta["best_mae"]
    print(f"Champion: {champion_name}, MAE: {champion_mae}")

    # load and combine datasets
    df_train = pd.read_csv(TRAINING_DATA)
    df_new = pd.read_csv(NEW_DATA)

    original_rows = len(df_train)
    new_rows = len(df_new)

    df_combined = pd.concat([df_train, df_new], ignore_index=True)
    combined_rows = len(df_combined)
    print(f"Data: {original_rows} + {new_rows} = {combined_rows} rows")

    X = df_combined.drop(columns=["audit_completion_days"])
    y = df_combined["audit_completion_days"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # retrain same model type on combined data
    mlflow.set_experiment(EXPERIMENT_NAME)
    retrained_model = get_model(champion_name)

    with mlflow.start_run(run_name=f"{champion_name}_retrained"):
        params = retrained_model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)

        retrained_model.fit(X_train, y_train)
        y_pred = retrained_model.predict(X_test)

        retrained_mae = float(mean_absolute_error(y_test, y_pred))
        retrained_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        mlflow.log_metric("mae", retrained_mae)
        mlflow.log_metric("rmse", retrained_rmse)
        mlflow.set_tag("priority", "high")
        mlflow.set_tag("stage", "retrained")
        mlflow.sklearn.log_model(retrained_model, artifact_path="model")

        print(f"Retrained MAE: {retrained_mae:.4f}, RMSE: {retrained_rmse:.4f}")

    # compare and decide
    retrained_mae = round(retrained_mae, 4)
    improvement = round(champion_mae - retrained_mae, 4)

    if improvement > 0:
        action = "promoted"
        best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        joblib.dump(retrained_model, best_model_path)
        print(f"Retrained model PROMOTED (improved by {improvement})")
    else:
        action = "kept_champion"
        print(f"Champion kept (no improvement, diff: {improvement})")

    output = {
        "original_data_rows": original_rows,
        "new_data_rows": new_rows,
        "combined_data_rows": combined_rows,
        "champion_mae": champion_mae,
        "retrained_mae": retrained_mae,
        "improvement": improvement,
        "min_improvement_threshold": 0,
        "action": action,
        "comparison_metric": "mae",
    }

    output_path = os.path.join(RESULTS_DIR, "step4_s8.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
