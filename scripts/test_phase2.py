import os
import pandas as pd
import sys



from mlflow.models import infer_signature
import mlflow
import mlflow.xgboost

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data.load_data import load_data
from models.evaluate import evaluate_model
from models.train import train_model

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("First Experiment")


def main():
 df = load_data("data/processed/train_processed.csv")
 print("Data loaded for modelling!")
 model,X_test,y_test,train_metrics,params = train_model(df,target_col="Churn")
#  X_test_float = X_test.astype("float64")
 evaluate_results = evaluate_model(model,X_test,y_test)
 signature = infer_signature(X_test,y_test)


 
 with mlflow.start_run():
    mlflow.log_params(params)

    for param, value in evaluate_results.items():
     if param != "classification_threshold":
        mlflow.log_metric(param,value)

    mlflow.log_param("classification_threshold",evaluate_results["classification_threshold"])
    mlflow.xgboost.log_model(model,name="xgboost_model",signature=signature)


if __name__=="__main__":
    main()


# mlflow server --backend-store-uri sqlite:///C:/mlflow_server/mlflow.db --default-artifact-root file:///C:/mlflow_server/artifacts --host 127.0.0.1 --port 5000