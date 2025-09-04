# tests/test_modules.py
import pandas as pd
import sys
import os

# 確保可以找到 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data_cleaning import clean_data

def test_clean_data():
    df = pd.DataFrame({
        'date': ['2025-01-01', None],
        'product': ['A', 'B'],
        'quantity': [10, None],
        'price': [100, 200]
    })
    cleaned = clean_data(df)
    assert 'total' in cleaned.columns
    assert cleaned.shape[0] == 1
    print("✅ clean_data 測試通過！")

if __name__ == "__main__":
    test_clean_data()
