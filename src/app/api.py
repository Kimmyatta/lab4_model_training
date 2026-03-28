# src/app/api.py
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json

EXAMPLE_FEATURES = [
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]
# Explicit request schema for breast cancer dataset (30 features)
class CancerRequest(BaseModel):
    features: list[float]= Field(
        default=EXAMPLE_FEATURES,
        description="List of 30 breast cancer features",
        examples=[EXAMPLE_FEATURES]
    )
def create_app(model_path: str = "models/model.pkl"):
    """
    Creates a FastAPI app that serves predictions for the Breast Cancer model.

    Example values for the 30-feature breast cancer input:
      - features: [
          17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
          1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
          25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]
        """
    # Helpful guard so students get a clear error if they forgot to train first
    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Train the model first (run the DAG or scripts/train_model.py)."
        )

    model= joblib.load(model_path)
    # Fix if tuple was saved
    if isinstance(model, tuple):
        model = model[0]    
    app = FastAPI(title="Breast Cancer Model API")

    # Map numeric predictions to class names
    target_names = {0: "malignant", 1: "benign"}

    @app.get("/")
    def root():
        return {
            "message": "Breast Cancer model is ready for inference!",
            "classes": target_names,
        }

    @app.post("/predict")
    def predict(request: CancerRequest):
        try:
             # Expect exactly 30 features
            if len(request.features) != 30:
                raise ValueError("Expected 30 features for breast cancer model")
                
            X = np.array([request.features])
            idx = int(model.predict(X)[0])
            
        except Exception as e:
            # Surface any shape/validation issues as a 400 instead of a 500
            raise HTTPException(status_code=400, detail=str(e))
        return {"prediction": target_names[idx], "class_index": idx}

    @app.get("/model/info")
    def model_info():
        with open("models/metadata.json") as f:
            metadata = json.load(f)
        return metadata
        
    # return the FastAPI app
    return app
