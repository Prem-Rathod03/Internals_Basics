"""
api.py - serves the best model using fastapi
endpoints: /heartbeat (health check), /score (prediction)
"""

import os

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# load the model at startup
model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")


app = FastAPI(title="AuditTrail Prediction API", version="1.0.0")


class AuditInput(BaseModel):
    controls_count: int = Field(..., ge=10, le=200, description="Number of controls (10-200)")
    evidence_items: int = Field(..., ge=20, le=500, description="Number of evidence items (20-500)")
    auditor_experience: int = Field(..., ge=1, le=20, description="Auditor experience in years (1-20)")
    is_regulatory: int = Field(..., ge=0, le=1, description="Regulatory flag (0 or 1)")


class PredictionOutput(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/heartbeat", response_model=HealthResponse)
def heartbeat():
    return HealthResponse(status="healthy", model_loaded=model is not None)


@app.post("/score", response_model=PredictionOutput)
def score(input_data: AuditInput):
    """takes audit features and returns predicted completion days"""
    features = np.array([[
        input_data.controls_count,
        input_data.evidence_items,
        input_data.auditor_experience,
        input_data.is_regulatory,
    ]])
    prediction = float(model.predict(features)[0])
    return PredictionOutput(prediction=round(prediction, 4))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
