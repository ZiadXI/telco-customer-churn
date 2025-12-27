import os
import pandas as pd
import sys,os


# sys.path.append(os.path.abspath("src"))
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.dataframe_validator import validate_dataframe


data_path = "C:\\Users\\Wind\\Downloads\\archive (10)\\WA_Fn-UseC_-Telco-Customer-Churn.csv"
target_col = "Churn"

def main():
    print("Loading Data...\n")
    df = load_data(data_path)   
    print(df.head(3))
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # Drop rows with missing numeric values
    df = df.dropna(subset=["TotalCharges", "MonthlyCharges"])

    print("Validating Data\n")
    valid,errors = validate_dataframe(df)
    if errors:
        print("Printing first 5 errors...")
        for idx,err in errors[:5]:
            print(f"Row: {idx}, Error: {err}")
    

    print("Preprocessing Data...\n")
    df_clean = preprocess_data(df,target_col=target_col)
    print(df_clean.head(3))
    
    print("Building Features\n")
    df_features = build_features(df_clean,target_col=target_col)
    print(df_features.head(3))
    print("Phase one done !")

    output_path = "data/processed/train_processed.csv"
    df_features.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
if __name__ == "__main__":
    main()

