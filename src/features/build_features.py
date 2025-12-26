import pandas as pd

def _map_binary_series(s:pd.Series)->pd.Series:

  vals = list(s.dropna().unique().astype(str))
  valset = set(vals)


  if valset == {"Yes","No"}:
    return s.map({"Yes":1,"No":0}).astype("Int64")

  if valset== {"Male","Female"}:
    return s.map({"Male":1,"Female":0})

  if len(vals)==2:
    sorted_vals = sorted(vals)
    return s.astype(str).map({sorted_vals[0]:0,sorted_vals[1]:1}).astype("Int64")

  return s

def build_features(df:pd.DataFrame,target_col:str="Churn")->pd.DataFrame:
    df=df.copy()

    # if "TotalCharges" in df.columns:
    #     df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    #     #fill the new nans with 0 
    #     df["TotalCharges"] = df["TotalCharges"].fillna(0)
    # df["TotalCharges"] = df["TotalCharges"].astype(float)
    object_cols=[ c for c in df.select_dtypes(include='object').columns if c != target_col]
    numeric_colns=df.select_dtypes(include=["Int64","Float64"]).columns.tolist()
    
    print(f"Found:{len(object_cols)} categorical cols and {len(numeric_colns)} numerical")

    binary_features = [c for c in object_cols if df[c].dropna().nunique() == 2]
    multi_colns = [c for c in object_cols if df[c].dropna().nunique() > 2]

    print(f"Found:{len(binary_features)} binary features and {len(multi_colns)} multi colns")

    for c in binary_features:
         original = df[c].dtype
         df[c] = _map_binary_series(df[c].astype(str))
         print(f"{c}: {original} → binary (0/1)")
    
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    if bool_cols:
            df[bool_cols] = df[bool_cols].astype(int)
            print(f"Converted {len(bool_cols)} boolean columns to int: {bool_cols}")
    
    if multi_colns:
        print("Applying One-Hot to multi colns")
        df =  pd.get_dummies(df,columns=multi_colns,drop_first=True,dtype=int)
        

        
    for c in binary_features:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    print(f" Feature engineering complete: {df.shape[1]} final features")
    return df        

    

