import pandas as pd



def preprocess_data (df: pd.DataFrame,target_col: str="Churn") -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    # Only drop customerID if it exists (for training data, not inference)
    if 'customerID' in df.columns:
        df = df.drop(columns='customerID')
    



    if target_col in df.columns and df[target_col].dtype=="object":
        df[target_col] = df[target_col].str.strip().map({"No":0,"Yes":1})

    return df

