import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
import pandas as pd
import gradio as gr 

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.model_loader import model, feature_names, THRESHOLD
from src.serving.schema import ChurnRequest, ChurnResponse
from src.serving.validator import validate_churn_request
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.serving.gradio_app import demo 

app = FastAPI(title="Telco Churn Prediction API")

@app.get("/health")
def root():
    return {"Status": "ok"}

@app.post("/predict", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest):
    is_valid, validation_errors = validate_churn_request(request)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Validation failed",
                "errors": validation_errors
            }
        )
    
    try:
        df = pd.DataFrame([request.model_dump()])
        
        df_processed = preprocess_data(df)
        df_built = build_features(df_processed)
        
        if 'Churn' in df_built.columns:
            df_built = df_built.drop(columns='Churn')

        for col in feature_names:
            if col not in df_built.columns:
                df_built[col]=0

        df_built = df_built[feature_names]
        
        for col in df_built.columns:
            if df_built[col].dtype == 'object':
                df_built[col] = pd.to_numeric(df_built[col], errors='coerce').fillna(0)
        df_built = df_built.astype(float)  

        proba = model.predict_proba(df_built)[0][1]
        prediction = "Yes" if proba >= THRESHOLD else "No"

        return ChurnResponse(
            churn_probability=float(proba),
            churn_prediction=prediction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

app = gr.mount_gradio_app(app, demo, path="/")
    #uvicorn app:app --reload 

    

#     {
#   "gender": "Male",
#   "SeniorCitizen": 0,
#   "Partner": "No",
#   "Dependents": "No",
#   "tenure": 1,
#   "PhoneService": "Yes",
#   "MultipleLines": "No",
#   "InternetService": "Fiber optic",
#   "OnlineSecurity": "No",
#   "OnlineBackup": "No",
#   "DeviceProtection": "No",
#   "TechSupport": "No",
#   "StreamingTV": "Yes",
#   "StreamingMovies": "Yes",
#   "Contract": "Month-to-month",
#   "PaperlessBilling": "Yes",
#   "PaymentMethod": "Electronic check",
#   "MonthlyCharges": 99.0,
#   "TotalCharges": 99.0
# }