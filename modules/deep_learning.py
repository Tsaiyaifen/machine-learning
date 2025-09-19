import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

# 定義模型 (簡單的兩層神經網路)
class SalesNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(SalesNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_pytorch_model(df: pd.DataFrame, epochs=100, lr=0.01, model_path="models/sales_net.pth"):
    # 使用 quantity 和 price 預測 total
    X = df[['quantity', 'price']].values.astype(np.float32)
    y = df[['total']].values.astype(np.float32)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    model = SalesNet(input_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 建立資料夾並儲存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ PyTorch Model saved at {model_path}")

    return model


def load_pytorch_model(model_path="models/sales_net.pth"):
    model = SalesNet(input_dim=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"✅ PyTorch Model loaded from {model_path}")
    return model

def predict_with_pytorch(model, quantity, price):
    model.eval()
    input_tensor = torch.tensor([[quantity, price]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction
