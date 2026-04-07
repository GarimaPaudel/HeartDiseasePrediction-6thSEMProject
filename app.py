import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.components.feature_engineering import FeatureEngineer
from src.components.model_registry import ModelRegistry
from src.utils.common import read_yaml



_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    config = read_yaml(config_path)
    registry = ModelRegistry(config.model_registry)
    try:
        _state["model"] = registry.load_production_model()
        _state["fe"] = FeatureEngineer()
    except FileNotFoundError as exc:
        # App starts but /predict will return 503 until a model is registered
        _state["model"] = None
        _state["fe"] = FeatureEngineer()
        _state["startup_error"] = str(exc)
    yield
    _state.clear()


app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    age: float = Field(..., gt=0, description="Age in years")
    gender: str = Field(..., pattern="^(male|female)$")
    height: float = Field(..., gt=0, description="Height in cm")
    weight: float = Field(..., gt=0, description="Weight in kg")
    systolic_bp: int = Field(..., alias="ap_hi", gt=0)
    diastolic_bp: int = Field(..., alias="ap_lo", gt=0)
    cholesterol: int = Field(..., ge=1, le=3, description="1=normal, 2=medium, 3=high")
    gluc: int = Field(..., ge=1, le=3, description="1=normal, 2=medium, 3=high")
    smoke: bool
    alco: bool
    active: bool

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    label: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Endpoints

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("model") is not None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(body: PredictRequest):
    model = _state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=_state.get("startup_error", "Model not loaded."),
        )

    fe: FeatureEngineer = _state["fe"]
    bmi = fe.compute_bmi(body.weight, body.height)
    bmi_class = fe.compute_bmi_class(bmi)
    male = 1 if body.gender == "male" else 0

    features = [
        body.age, male, body.height, body.weight,
        body.systolic_bp, body.diastolic_bp,
        body.cholesterol, body.gluc,
        int(body.smoke), int(body.alco), int(body.active),
        bmi, bmi_class,
    ]
    df = pd.DataFrame(
        [features],
        columns=["age", "male", "height", "weight", "ap_hi", "ap_lo",
                 "cholesterol", "gluc", "smoke", "alco", "active", "bmi", "bmi_class"],
    )

    prediction = int(model.predict(df)[0])
    label = "Presence of Heart Disease" if prediction == 1 else "Absence of Heart Disease"
    return PredictResponse(prediction=prediction, label=label)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
