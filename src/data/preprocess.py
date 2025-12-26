import pandas as pd



def preprocess_data (df: pd.DataFrame,target_col: str="Churn") -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df = df.drop(columns='customerID')
    


    if target_col in df.columns and df[target_col].dtype=="object":
        df[target_col] = df[target_col].str.strip().map({"No":0,"Yes":1})

    return df

