import pandas as pd
from src.utils.schema import User
from src.utils.rules import validate_rules

def validate_dataframe(df:pd.DataFrame):
    valid=[]
    errors=[]
    for idx, row in df.iterrows():
      try:
        obj = User(**row.to_dict())
        rules = validate_rules(obj)
        valid.append(obj)
      except Exception as e:
        errors.append((idx,str(e)))
    
    

    

    print(f"Valid rows: {len(valid)}")
    print(f"Failed rows: {len(errors)}\n")
    return valid,errors
