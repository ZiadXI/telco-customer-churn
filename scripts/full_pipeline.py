#!/usr/bin/env python3
"""
Production-ready Telco Churn Pipeline
"""

import os
import sys
import argparse
from xml.parsers.expat import features
import mlflow
import mlflow.xgboost
import joblib
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from mlflow.models.signature import infer_signature


project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.utils.dataframe_validator import validate_dataframe
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model

# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# mlflow.set_tracking_uri(args.mlflow_uri)
# mlflow.set_experiment(args.experiment)



def main(args):
 print("Starting telco churn analysis")
 mlflow.set_tracking_uri(args.mlflow_uri)
 mlflow.set_experiment(args.experiment)
 with mlflow.start_run():
    try:
        mlflow.log_param("input_file", args.input)
        mlflow.log_param("target_column", args.target)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("random_state", 42)
        
        print("Loading data...\n")

        file_path = args.input
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        df = load_data(file_path)

        print("Data Loaded Succesfully")

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges", "MonthlyCharges"])

        mlflow.log_metric("total_samples",df.shape[0])
        mlflow.log_metric("total_features",df.shape[1])

        if args.verbose:
            print(df.head(3).to_string())
            print()

        print("Validating data...\n")
           
        valid,errors = validate_dataframe(df)

        mlflow.log_metric("validation_total_rows",len(df))
        mlflow.log_metric("validation_valid_rows",len(valid))
        mlflow.log_metric("validation_failed_rows",len(errors))
        mlflow.set_tag("validation_status","passed" if len(errors)==0 else "failed")
        
    
        print("Validation Complete")
        if errors:
         failure_dir = project_root/"data"/"validation_failures"
         failure_path = failure_dir / "full_validation_errors.json"
         print(f"Saved validation errors in {failure_path}")

         failure_dir.mkdir(parents=True, exist_ok=True)
         with open(failure_path,"w") as f:
            json.dump(errors,f,indent=2)
        

         print("Printing first 5 errors")
         for error in errors[:5]:
            print(error)

         mlflow.log_dict({
            "failed_row_count":len(errors),
            "sample_errors":errors[:10]
         },
          "validation_errors.json"
         )   
         raise ValueError(f"Failed {len(errors)} out of {len(df)} rows")

        print("Preprocessing the data..\n")

        df_clean = preprocess_data(df,target_col = args.target)

        preprocessed_dir = project_root/"data"/"processed"
        preprocessed_dir.mkdir(parents=True,exist_ok=True)
        preprocessed_path = preprocessed_dir / "train_preprocessed.csv"
     
        df_clean.to_csv(preprocessed_path,index=False)

        mlflow.log_artifact(str(preprocessed_path),artifact_path="preprocessed_data")
        mlflow.log_param("preprocessed_data_path", str(preprocessed_path))
        
        print("Data Sucessfully Preprocessed\n")

        print("Building Data Features\n")

        df_features = build_features(df_clean)

        mlflow.log_metric("dropped_rows_after_feature_engineering",df.shape[0] - df_features.shape[0])
        mlflow.log_metric("features_count_after_feature_engineering",df_features.shape[1])
  
        print("Saving the training features")
        
        train_features = [feature for feature in df_features.columns]

        artifacts_dir = project_root/"artifacts"
        artifacts_dir.mkdir(parents=True,exist_ok=True)
        artifacts_path = artifacts_dir / f"train_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(artifacts_path,"w") as f:
            json.dump(train_features,f,indent=2)
        
        mlflow.log_artifact(str(artifacts_path),artifact_path="feature_engineered_data")
        print(f"Training feature names saved at {artifacts_path} as a .json file")

        preprocessed_dir = project_root/"data"/"processed"
        preprocessed_dir.mkdir(parents=True,exist_ok=True)
        preprocessed_path = preprocessed_dir / "train_feature_engineered.csv"
        df_features.to_csv(preprocessed_path,index=False)

        print(f"Feature engineered data saved at {preprocessed_path} succesfully")

        mlflow.log_artifact(str(preprocessed_path),artifact_path="feature_engineered_data")
        mlflow.log_param("feature_engineered_data_path", str(preprocessed_path))


     #train model
        print("Training Model...\n")

        model,X_test,y_test,train_metrics,params = train_model(df_features,
        target_col=args.target,
        test_size=args.test_size,
        random_state=42,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators)

        mlflow.log_params(params)
        mlflow.log_metrics(train_metrics)
        
        
        x_sig = X_test.copy()

        int_cols = x_sig.select_dtypes(include=["int","int64"]).columns
        x_sig[int_cols] = x_sig[int_cols].astype("float64")

        y_pred = model.predict(x_sig)
        signature = infer_signature(x_sig, y_pred)

        mlflow.xgboost.log_model(model,signature=signature,name="xgboost_model")

        artifacts_dir = project_root/"artifacts"/"xgboost_model"
        artifacts_dir.mkdir(parents=True,exist_ok=True)

        local_model_path = artifacts_dir / f"trained_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        model.save_model(local_model_path)

        print("Evaluating Model...\n")
 
        evaluate_results = evaluate_model(model,X_test,y_test,args.threshold)

        mlflow.log_metrics(evaluate_results)

    except Exception as e:
      mlflow.set_tag("pipeline_status","failed")
      mlflow.log_param("error",str(e))
      raise e

            
        
         
        
 

        



    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telco Churn Prediction Pipeline"
    )

    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--target", type=str, default="Churn", help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="XGBoost learning rate")
    parser.add_argument("--max_depth", type=int, default=15, help="Max tree depth")
    parser.add_argument("--n_estimators", type=int, default=150, help="Number of trees")
    parser.add_argument("--experiment", type=str, default="Telco_Churn_Production", help="MLflow experiment name")
    parser.add_argument("--mlflow_uri", type=str, default="http://127.0.0.1:5000", help="MLflow tracking URI")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")

    args = parser.parse_args()
    main(args)



# python scripts/full_pipeline.py --input "C:\\Users\\Wind\\Downloads\\archive (10)\\WA_Fn-UseC_-Telco-Customer-Churn.csv"

# mlflow server --backend-store-uri sqlite:///C:/mlflow_server/mlflow.db --default-artifact-root file:///C:/mlflow_server/artifacts --host 127.0.0.1 --port 5000



