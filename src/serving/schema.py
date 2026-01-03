from pydantic import BaseModel, Field
from typing import Optional


class ChurnRequest(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1, description="Must be 0 or 1")
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=72, description="Must be between 0 and 72")
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
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0, description="Must be non-negative")
    TotalCharges: float = Field(ge=0, description="Must be non-negative")


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
