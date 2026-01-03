"""
API-specific validation for ChurnRequest
"""
from typing import List, Tuple
from src.serving.schema import ChurnRequest


def validate_churn_request(request: ChurnRequest) -> Tuple[bool, List[str]]:
    """
    Validate a ChurnRequest object against business rules.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    allowed_gender = ["Male", "Female"]
    if request.gender not in allowed_gender:
        errors.append(f"Invalid gender: '{request.gender}'. Must be one of: {allowed_gender}")
    
    if request.SeniorCitizen not in [0, 1]:
        errors.append(f"Invalid SeniorCitizen: {request.SeniorCitizen}. Must be 0 or 1")
    
    if not (0 <= request.tenure <= 72):
        errors.append(f"Invalid tenure: {request.tenure}. Must be between 0 and 72")
    
    binary_fields = {
        "Partner": request.Partner,
        "Dependents": request.Dependents,
        "PhoneService": request.PhoneService,
        "PaperlessBilling": request.PaperlessBilling
    }
    
    for field_name, field_value in binary_fields.items():
        if field_value not in ["Yes", "No"]:
            errors.append(f"Invalid {field_name}: '{field_value}'. Must be 'Yes' or 'No'")
    
    optional_binary_fields = {
        "MultipleLines": request.MultipleLines,
        "OnlineSecurity": request.OnlineSecurity,
        "OnlineBackup": request.OnlineBackup,
        "DeviceProtection": request.DeviceProtection,
        "TechSupport": request.TechSupport,
        "StreamingTV": request.StreamingTV,
        "StreamingMovies": request.StreamingMovies
    }
    
    for field_name, field_value in optional_binary_fields.items():
        if field_value is not None and field_value not in ["Yes", "No", "No phone service", "No internet service"]:
            errors.append(
                f"Invalid {field_name}: '{field_value}'. "
                f"Must be 'Yes', 'No', 'No phone service', or 'No internet service'"
            )
    
    allowed_internet = ["DSL", "Fiber optic", "No"]
    if request.InternetService not in allowed_internet:
        errors.append(
            f"Invalid InternetService: '{request.InternetService}'. "
            f"Must be one of: {allowed_internet}"
        )
    
    allowed_contract = ["Month-to-month", "One year", "Two year"]
    if request.Contract not in allowed_contract:
        errors.append(
            f"Invalid Contract: '{request.Contract}'. "
            f"Must be one of: {allowed_contract}"
        )
    
    allowed_payment = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    if request.PaymentMethod not in allowed_payment:
        errors.append(
            f"Invalid PaymentMethod: '{request.PaymentMethod}'. "
            f"Must be one of: {allowed_payment}"
        )
    
    if request.MonthlyCharges < 0:
        errors.append(f"MonthlyCharges cannot be negative: {request.MonthlyCharges}")
    
    if request.TotalCharges < 0:
        errors.append(f"TotalCharges cannot be negative: {request.TotalCharges}")
    
    if request.TotalCharges < request.MonthlyCharges:
        errors.append(
            f"TotalCharges ({request.TotalCharges}) must be >= MonthlyCharges ({request.MonthlyCharges})"
        )
    
    if request.InternetService == "No":
        internet_dependent_fields = {
            "OnlineSecurity": request.OnlineSecurity,
            "OnlineBackup": request.OnlineBackup,
            "DeviceProtection": request.DeviceProtection,
            "TechSupport": request.TechSupport,
            "StreamingTV": request.StreamingTV,
            "StreamingMovies": request.StreamingMovies
        }
        
        for field_name, field_value in internet_dependent_fields.items():
            if field_value not in [None, "No internet service"]:
                errors.append(
                    f"Cannot have InternetService='No' and {field_name}='{field_value}'. "
                    f"{field_name} must be None or 'No internet service'"
                )
    
    if request.PhoneService == "No" and request.MultipleLines not in [None, "No phone service"]:
        errors.append(
            f"Cannot have PhoneService='No' and MultipleLines='{request.MultipleLines}'. "
            f"MultipleLines must be None or 'No phone service'"
        )
    
    return len(errors) == 0, errors

