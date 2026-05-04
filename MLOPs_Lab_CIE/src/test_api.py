"""
test_api.py - tests the fastapi endpoints and saves results
run this after starting the api server (python api.py)
"""

import json
import os
import sys

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

API_BASE = "http://localhost:8080"

TEST_INPUT = {
    "controls_count": 86,
    "evidence_items": 341,
    "auditor_experience": 7,
    "is_regulatory": 0,
}


def main():
    # check health first
    print("Testing /heartbeat...")
    try:
        resp = requests.get(f"{API_BASE}/heartbeat", timeout=5)
        resp.raise_for_status()
        health_response = resp.json()
        print(f"  Health: {health_response}")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Make sure the API server is running on port 8080!")
        sys.exit(1)

    # test prediction
    print("Testing /score...")
    resp = requests.post(f"{API_BASE}/score", json=TEST_INPUT, timeout=5)
    resp.raise_for_status()
    score_response = resp.json()
    prediction = score_response["prediction"]
    print(f"  Prediction: {prediction}")

    # save results
    output = {
        "health_endpoint": "/heartbeat",
        "predict_endpoint": "/score",
        "port": 8080,
        "health_response": health_response,
        "test_input": TEST_INPUT,
        "prediction": prediction,
    }

    output_path = os.path.join(RESULTS_DIR, "step2_s4.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
