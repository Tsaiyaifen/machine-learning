import PySimpleGUI as sg
import pandas as pd
from modules.data_modeling import train_model  # 你的 ML 模組

# 1️⃣ 載入清洗好的數據並訓練模型
df_clean = pd.read_csv('data/processed/sales_clean.csv')
model = train_model(df_clean)

# 2️⃣ 定義 GUI 版面
layout = [
    [sg.Text("Quantity"), sg.Input(key="quantity")],
    [sg.Text("Price"), sg.Input(key="price")],
    [sg.Button("Predict"), sg.Text("", key="result")],
    [sg.Button("Exit")]
]

window = sg.Window("Sales Prediction", layout)

# 3️⃣ 事件迴圈
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Exit":
        break
    if event == "Predict":
        try:
            qty = float(values["quantity"])
            price = float(values["price"])
            total = model.predict([[qty, price]])[0]
            window["result"].update(f"Predicted Total: {total:.2f}")
        except Exception as e:
            window["result"].update(f"Error: {e}")

window.close()
