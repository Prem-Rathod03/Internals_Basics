"""
register_model.py - registers the best model in mlflow model registry
uses metadata from train.py to find the right run
"""

import json
import os
import warnings

import mlflow
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_META_PATH = os.path.join(BASE_DIR, "models", "best_model_meta.json")
MODEL_NAME = "audittrail-audit-completion-days-predictor"

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    with open(MODELS_META_PATH, "r") as f:
        meta = json.load(f)

    run_id = meta["best_run_id"]
    best_mae = meta["best_mae"]

    print(f"Registering model from run_id: {run_id}")
    print(f"Best MAE: {best_mae}")

    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)

    version = int(result.version)
    print(f"Registered '{MODEL_NAME}' version {version}")

    output = {
        "registered_model_name": MODEL_NAME,
        "version": version,
        "run_id": run_id,
        "source_metric": "mae",
        "source_metric_value": best_mae,
    }

    output_path = os.path.join(RESULTS_DIR, "step3_s6.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
