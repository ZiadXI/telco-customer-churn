import mlflow
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import recall_score




def train_model(df:pd.DataFrame,
    target_col: str,
    scale_pos_weight: float = None,
    test_size: float = 0.2,
    random_state: int = 42,
    learning_rate: float = 0.01,
    max_depth: int = 15,
    n_estimators: int = 150):

    X=df.drop(target_col,axis=1)
    y=df[target_col]
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


    if scale_pos_weight is None:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Auto calculated scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        print(f"Using provided scale_pos_weight: {scale_pos_weight:.2f}")

    params = {
        'colsample_bytree': 0.6,
        'gamma': 0.2,
        'learning_rate': learning_rate,
        'max_depth': 15,
        'min_child_weight': 5,
        'n_estimators': n_estimators,
        'subsample': 0.6,
        'n_jobs': -1,
        'random_state': random_state,
        'scale_pos_weight': scale_pos_weight
    }
    model = XGBClassifier(**params)

    print(" Training XGBoost model...")
    model.fit(X_train,y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test,preds)
    rec = recall_score(y_test, preds)
    
    print(f"   Training complete!")
    print(f"   Train Accuracy: {acc:.4f}")
    print(f"   Train Recall:   {rec:.4f}")
    
    train_metrics = {
        "train_accuracy": acc,
        "train_recall": rec
    }


    return model,X_test,y_test,train_metrics,params
   

