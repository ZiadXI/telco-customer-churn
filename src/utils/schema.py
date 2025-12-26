"""
Data validation using Pydantic
"""
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional


class User(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: Optional[str] = None
    InternetService: str
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: str
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    Churn: Optional[str] = None
