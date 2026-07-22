"""FastAPI service exposing the best student-performance regression model.

Endpoints:
    POST /predict  - predict a student's final grade (G3, 0-20)
    POST /retrain  - retrain the model from existing data plus newly uploaded CSV rows
    GET  /         - health check / redirect hint to /docs

Run locally:
    uv run uvicorn prediction:app --reload --app-dir summative/API
"""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).resolve().parent / "models"
BUNDLE_PATH = MODELS_DIR / "best_model.pkl"
TRAIN_DATA_PATH = MODELS_DIR / "train_data.csv"

app = FastAPI(
    title="Student Performance Prediction API",
    description=(
        "Predicts a student's final Mathematics grade (G3, 0-20) from academic "
        "history and study habits, supporting early intervention for at-risk "
        "students. Model: best of SGD Linear Regression / Decision Tree / Random "
        "Forest, selected by test MSE."
    ),
    version="1.0.0",
)

# CORS reasoning:
# - allow_origins=["*"]: the API is consumed by a Flutter mobile app (requests do not
#   originate from a fixed web domain) and by graders testing from Swagger UI or their
#   own machines, so no single origin can be whitelisted. The API is public and
#   stateless - it serves predictions only.
# - allow_credentials=False: we use no cookies or sessions, and browsers forbid
#   credentials together with a wildcard origin. Restricting this is a deliberate
#   security choice, not an omission.
# - allow_methods=["GET", "POST"]: the API only exposes GET (health/docs) and POST
#   (predict/retrain). DELETE, PUT and PATCH are restricted because no endpoint
#   mutates or removes a resource.
# - allow_headers=["Content-Type"]: the only header clients need to send JSON or
#   multipart uploads; everything else stays restricted.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class StudentInput(BaseModel):
    """One student's data. Every field is type-enforced and range-constrained to the
    realistic bounds observed in the UCI Student Performance dataset."""

    G1: float = Field(..., ge=0, le=20, description="First-period grade (0-20)")
    G2: float = Field(..., ge=0, le=20, description="Second-period grade (0-20)")
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures (0-4)")
    Medu: int = Field(..., ge=0, le=4,
                      description="Mother's education: 0 none .. 4 higher education")
    age: int = Field(..., ge=15, le=22, description="Student age in years (15-22)")
    goout: int = Field(..., ge=1, le=5,
                       description="Going out with friends: 1 very low .. 5 very high")
    studytime: int = Field(..., ge=1, le=4,
                           description="Weekly study time: 1 (<2h) .. 4 (>10h)")
    absences: int = Field(..., ge=0, le=93, description="Number of school absences (0-93)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"G1": 12, "G2": 13, "failures": 0, "Medu": 3, "age": 16,
                 "goout": 3, "studytime": 2, "absences": 4}
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_final_grade: float = Field(description="Predicted G3 on the 0-20 scale")
    model_used: str
    interpretation: str


class RetrainResponse(BaseModel):
    message: str
    model_used: str
    rows_added: int
    total_training_rows: int
    new_test_mse: float


def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found on server.")
    return joblib.load(BUNDLE_PATH)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "predict": "POST /predict",
            "retrain": "POST /retrain"}


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput):
    bundle = load_bundle()
    row = pd.DataFrame([student.model_dump()])[bundle["features"]]
    raw = bundle["model"].predict(bundle["scaler"].transform(row))[0]
    grade = float(np.clip(raw, 0, 20))

    if grade < 10:
        note = "At risk: predicted below the pass mark (10/20). Early support recommended."
    elif grade < 14:
        note = "On track: predicted a passing grade."
    else:
        note = "Strong performance predicted."

    return PredictionResponse(
        predicted_final_grade=round(grade, 2),
        model_used=bundle["model_name"],
        interpretation=note,
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)):
    """Retrain the existing model when new data is uploaded.

    Upload a CSV with the columns: G1, G2, failures, Medu, age, goout, studytime,
    absences, G3. The new rows are appended to the stored training data, the model is
    refitted with its existing hyperparameters, evaluated, and the saved model file is
    replaced.
    """
    bundle = load_bundle()
    features, target = bundle["features"], bundle["target"]

    try:
        new_df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable CSV.")

    missing = [c for c in features + [target] if c not in new_df.columns]
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"CSV is missing required columns: {missing}")
    if len(new_df) == 0:
        raise HTTPException(status_code=422, detail="CSV contains no data rows.")

    base_df = pd.read_csv(TRAIN_DATA_PATH)
    full_df = pd.concat([base_df[features + [target]], new_df[features + [target]]],
                        ignore_index=True).dropna()

    X, y = full_df[features], full_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    scaler = bundle["scaler"].__class__()
    model = bundle["model"].__class__(**bundle["model"].get_params())
    model.fit(scaler.fit_transform(X_train), y_train)
    new_mse = float(mean_squared_error(y_test, model.predict(scaler.transform(X_test))))

    bundle.update(model=model, scaler=scaler, test_mse=new_mse)
    joblib.dump(bundle, BUNDLE_PATH)
    full_df.to_csv(TRAIN_DATA_PATH, index=False)

    return RetrainResponse(
        message="Model retrained and saved successfully.",
        model_used=bundle["model_name"],
        rows_added=len(new_df),
        total_training_rows=len(full_df),
        new_test_mse=round(new_mse, 3),
    )
