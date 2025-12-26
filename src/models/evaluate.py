import mlflow
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold (default: 0.5)
    """
    
    # get predictions
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    if threshold != 0.5:
        preds = (proba >= threshold).astype(int)
        
    # calculate metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)
    
   
    
    # calculate confusion matrix
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    
    
    # lalculate additional business metrics for churn
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    
    
    # Print results
    print("\n" + "="*60)
    print(" MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\n Core Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   ROC AUC:   {roc_auc:.4f}")
    
    print(f"\n Business Impact:")
    print(f"   False Negative Rate: {false_negative_rate:.4f} (Missed churners)")
    print(f"   False Positive Rate: {false_positive_rate:.4f} (False alarms)")
    
    print(f"\n Confusion Matrix:")
    print(f"   True Negatives:  {tn} | False Positives: {fp}")
    print(f"   False Negatives: {fn} | True Positives:  {tp}")
    
    print("\n Detailed Classification Report:")
    print(classification_report(y_test, preds, digits=4))
    print("="*60 + "\n")
    
    return {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "classification_threshold": threshold
    }