import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 建立副本並清理
    df_clean = df.dropna().copy()
    
    # 處理資料
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['total'] = df_clean['quantity'] * df_clean['price']
    
    return df_clean
