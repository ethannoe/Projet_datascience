"""
API REST d'inférence — Marketing ROI Prediction Service.

Architecture : Front (Dashboard) → API REST → Modèle sérialisé
Cette séparation reproduit une architecture professionnelle réelle.

Endpoints :
  GET  /health        Santé du service + état du modèle
  POST /predict       Prédiction des ventes marketing
  GET  /model-info    Informations sur les modèles disponibles

Usage :
    uvicorn api.main:app --reload
    Documentation interactive : http://localhost:8000/docs (Swagger)
"""

import os
import sys
from datetime import datetime
from typing import Literal, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = "saved_models"

app = FastAPI(
    title="Marketing ROI Prediction API",
    description=(
        "Service d'inférence pour la prédiction des ventes marketing. "
        "Retourne les ventes estimées et le ROI associé à une combinaison budgétaire."
    ),
    version="1.0.0",
)

_model_cache: dict = {}


def _load_model(task: str, name: str):
    key = f"{task}_{name}"
    if key not in _model_cache:
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(MODEL_DIR, f"{task}_{safe}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle non trouvé : {path}. Lancez train.py d'abord.")
        _model_cache[key] = joblib.load(path)
    return _model_cache[key]


# ─── Schémas Pydantic ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    TV: float = Field(..., ge=0, description="Budget TV en M€")
    Radio: float = Field(..., ge=0, description="Budget Radio en M€")
    Social_Media: float = Field(..., ge=0, description="Budget Social Media en M€")
    Influencer: Literal["Mega", "Macro", "Micro", "Nano"] = Field(
        ..., description="Type d'influenceur"
    )
    model: Optional[str] = Field(
        "Gradient Boosting",
        description="Nom du modèle : 'Linear Regression' | 'Random Forest' | 'Gradient Boosting' | 'MLP (Deep Learning)'",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "TV": 120.5,
            "Radio": 18.0,
            "Social_Media": 12.3,
            "Influencer": "Macro",
            "model": "Gradient Boosting",
        }
    }}


class PredictResponse(BaseModel):
    predicted_sales_M: float = Field(..., description="Ventes prédites en M€")
    roi_estimate: float = Field(..., description="ROI = Ventes / Budget Total")
    total_budget_M: float = Field(..., description="Budget total en M€")
    profit_estimate_M: float = Field(..., description="Profit estimé = Ventes - Budget")
    model_used: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    models_available: bool
    available_models: list
    timestamp: str


class ModelInfoResponse(BaseModel):
    available_regression_models: list
    available_classification_models: list
    best_regression_model: Optional[str]
    best_classification_model: Optional[str]
    target_regression: str
    target_classification: str
    input_features: list


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Vérifie que le service est actif et que les modèles sont disponibles."""
    meta_exists = os.path.exists(os.path.join(MODEL_DIR, "metadata.joblib"))
    available = []
    for name in ["Linear Regression", "Random Forest", "Gradient Boosting", "MLP (Deep Learning)"]:
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        if os.path.exists(os.path.join(MODEL_DIR, f"regression_{safe}.joblib")):
            available.append(name)
    return HealthResponse(
        status="healthy" if meta_exists else "degraded — lancez train.py",
        models_available=meta_exists,
        available_models=available,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inférence"])
def predict(request: PredictRequest):
    """
    Prédit les ventes pour une combinaison budgétaire donnée.

    Retourne les ventes en M€, le ROI estimé, le profit et les métadonnées du modèle.
    """
    model_name = request.model or "Gradient Boosting"
    try:
        pipeline = _load_model("regression", model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    input_df = pd.DataFrame([{
        "TV": request.TV,
        "Radio": request.Radio,
        "Social_Media": request.Social_Media,
        "Influencer": request.Influencer,
    }])

    try:
        predicted_sales = float(pipeline.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    total_budget = request.TV + request.Radio + request.Social_Media
    roi = predicted_sales / total_budget if total_budget > 0 else 0.0
    profit = predicted_sales - total_budget

    return PredictResponse(
        predicted_sales_M=round(predicted_sales, 4),
        roi_estimate=round(roi, 4),
        total_budget_M=round(total_budget, 4),
        profit_estimate_M=round(profit, 4),
        model_used=model_name,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Modèles"])
def model_info():
    """Retourne les informations sur les modèles disponibles et le meilleur modèle."""
    meta_path = os.path.join(MODEL_DIR, "metadata.joblib")
    if not os.path.exists(meta_path):
        raise HTTPException(
            status_code=404,
            detail="Aucun modèle entraîné. Lancez train.py d'abord.",
        )
    meta = joblib.load(meta_path)

    reg_available, clf_available = [], []
    for name in ["Linear Regression", "Random Forest", "Gradient Boosting", "MLP (Deep Learning)"]:
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        if os.path.exists(os.path.join(MODEL_DIR, f"regression_{safe}.joblib")):
            reg_available.append(name)
        if os.path.exists(os.path.join(MODEL_DIR, f"classification_{safe}.joblib")):
            clf_available.append(name)
    # Ajout Logistic Regression classification
    if os.path.exists(os.path.join(MODEL_DIR, "classification_Logistic_Regression.joblib")):
        clf_available.insert(0, "Logistic Regression")

    return ModelInfoResponse(
        available_regression_models=reg_available,
        available_classification_models=clf_available,
        best_regression_model=meta.get("best_reg_model"),
        best_classification_model=meta.get("best_clf_model"),
        target_regression="Sales (M€)",
        target_classification="Performance (High / Medium / Low)",
        input_features=["TV", "Radio", "Social_Media", "Influencer"],
    )
