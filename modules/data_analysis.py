import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summary_statistics(df: pd.DataFrame):
    return df.describe()

def sales_by_product(df: pd.DataFrame):
    return df.groupby('product')['total'].sum().sort_values(ascending=False)

def plot_sales_report(df: pd.DataFrame):
    """
    一次生成三種銷售分析圖表：
    1. 銷售額隨時間變化折線圖
    2. 各產品銷售額柱狀圖
    3. 銷售數量與單價散點圖
    """
    # 1️⃣ 銷售額隨時間變化折線圖
    daily_sales = df.groupby('date')['total'].sum()
    plt.figure(figsize=(8,5))
    plt.plot(daily_sales.index, daily_sales.values, marker='o')
    plt.title("Daily Sales Total")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.grid(True)
    plt.show()

    # 2️⃣ 各產品銷售額柱狀圖
    product_sales = df.groupby('product')['total'].sum().reset_index()
    plt.figure(figsize=(6,4))
    sns.barplot(x='product', y='total', data=product_sales, palette='viridis')
    plt.title("Total Sales by Product")
    plt.xlabel("Product")
    plt.ylabel("Total Sales")
    plt.show()

    # 3️⃣ 銷售數量與單價散點圖
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='quantity', y='price', hue='product', size='total', sizes=(50,200), data=df)
    plt.title("Quantity vs Price with Total Sales")
    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.show()
