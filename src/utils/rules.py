import pandas as pd
from src.utils.schema import User




def validate_rules(row:User):
    total = getattr(row, "TotalCharges")
    monthly = getattr(row, "MonthlyCharges")

    if total < monthly:
     raise ValueError(f"TotalCharges ({total}) must be >= MonthlyCharges ({monthly})")


    
    allowed_gender =["Male","Female"]
    allowed_sen_citizen =[0,1]
    if row.gender not in allowed_gender:
        raise ValueError (f"Invalid Gender: {row.gender}")
    if not (0 <= row.tenure <= 72):
        raise ValueError (f"Tenure must be between 0 & 72: {row.tenure}")
    # if row.Dependents =="Yes" and row.Partner=="No":
    #     raise ValueError("Dependents cannot be 'Yes' when Partner is 'No'")
    if row.SeniorCitizen not in [0, 1]:
        raise ValueError(f"Invalid SeniorCitizen value: {row.SeniorCitizen}")
    # if row.InternetService == "No":
    #     internet_cols =[
    #         "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    #     "TechSupport", "StreamingTV", "StreamingMovies"
    #     ]
    #     for col in internet_cols:
    #         if getattr(row,col) not in ["No",None]:
    #             raise ValueError(f"Invalid, Can't have NO internet and have {col}")

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        value = getattr(row,col)
        if value <0:
            raise ValueError(f"{col} Can't be negative")

    binary_cols = ["Partner", "Dependents", "PhoneService"]
    for col in binary_cols:
      if getattr(row, col) not in ["Yes", "No"]:
          raise ValueError(f"{col} must be Yes or No")
    
    