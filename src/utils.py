import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=['policy_number', 'policy_bind_date', 'incident_date',
                     'insured_zip', '_c39'], inplace=True, errors='ignore')
    
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df