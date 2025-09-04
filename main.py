import pandas as pd
from db.setup_tables import setup
from modules.data_cleaning import clean_data
from modules.data_modeling import train_model
from modules.data_analysis import summary_statistics, sales_by_product, plot_sales_report

# 1️⃣ 建立資料表
setup()

# 2️⃣ 匯入 CSV 並清洗
df = pd.read_csv('data/raw/sales.csv')
df_clean = clean_data(df)
df_clean.to_csv('data/processed/sales_clean.csv', index=False)

# 3️⃣ 從清洗後的 CSV 讀取，確保分析資料一致
df_clean = pd.read_csv('data/processed/sales_clean.csv')

# 4️⃣ 分析與建模
print(summary_statistics(df_clean))
print(sales_by_product(df_clean))
model = train_model(df_clean)
print("Model trained successfully!")

# 5️⃣ 生成分析圖表報表
plot_sales_report(df_clean)
