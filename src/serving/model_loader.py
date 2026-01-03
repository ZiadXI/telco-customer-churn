from pathlib import Path
import json
import xgboost as xgb


PROJECT_ROOT = Path(__file__).parent.parent.parent

Model_DIR = PROJECT_ROOT/"artifacts"/"xgboost_model"
Features_DIR = PROJECT_ROOT/"artifacts"


THRESHOLD=0.5

def load_model():
 model_files = sorted(Model_DIR.glob("trained_model_*.json"))
 if not model_files:
    raise FileNotFoundError("No trained model found")

 model = xgb.XGBClassifier()
 model.load_model(model_files[-1])
 return model

def load_feature_names():
    feature_files = sorted(Features_DIR.glob("train_features_*.json"))
    if not feature_files:
        raise FileNotFoundError("Feature Names not found")
    
    with open(feature_files[-1], "r") as f:
        features = json.load(f)
    
    # Remove 'Churn' if present (it's the target variable, not a feature)
    if 'Churn' in features:
        features = [f for f in features if f != 'Churn']
    
    return features

model = load_model()
feature_names = load_feature_names()



 