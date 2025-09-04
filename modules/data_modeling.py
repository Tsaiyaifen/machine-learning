from sklearn.linear_model import LinearRegression
import pandas as pd

def train_model(df: pd.DataFrame):
    X = df[['quantity', 'price']]
    y = df['total']
    model = LinearRegression()
    model.fit(X, y)
    return model
