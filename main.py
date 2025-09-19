import pandas as pd
from db.setup_tables import setup
from modules.data_cleaning import clean_data
from modules.data_modeling import train_model
from modules.data_analysis import summary_statistics, sales_by_product, plot_sales_report
from modules.deep_learning import train_pytorch_model, load_pytorch_model  # ⬅️ 新增

def main():
    print("=" * 50)
    print("Step 1. 建立資料表") 
    setup()

    print("=" * 50)
    print("Step 2. 匯入與清洗資料") 
    df = pd.read_csv('data/raw/sales.csv')
    df_clean = clean_data(df)
    df_clean.to_csv('data/processed/sales_clean.csv', index=False)
    print("Data cleaned and saved to data/processed/sales_clean.csv") 

    print("=" * 50)
    print("Step 3. 載入清洗後資料") 
    df_clean = pd.read_csv('data/processed/sales_clean.csv')
    print(df_clean.head())

    print("=" * 50)
    print("Step 4. 分析報表") 
    print("\n--- Summary Statistics ---")
    print(summary_statistics(df_clean))

    print("\n--- Sales by Product ---")
    print(sales_by_product(df_clean))

    print("=" * 50)
    print("Step 5. 視覺化報表") 
    plot_sales_report(df_clean)

    print("=" * 50)
    print("Step 6. 機器學習模型 (Scikit-learn)") 
    model = train_model(df_clean)
    print("Scikit-learn Model trained successfully!") 

    print("=" * 50)
    print("Step 7. 深度學習模型 (PyTorch)") 
    dl_model = train_pytorch_model(df_clean, epochs=100, lr=0.01)

    print("=" * 50)
    print("Step 8. 測試載入已儲存的 PyTorch 模型") 
    loaded_model = load_pytorch_model()

    print("=" * 50)
    print("Step 9. 模型預測 (Prediction)") 
    test_quantity, test_price = 10, 200
    sklearn_pred = model.predict([[test_quantity, test_price]])[0]
    pytorch_pred = predict_with_pytorch(loaded_model, test_quantity, test_price)
    print(f"Scikit-learn 預測: quantity={test_quantity}, price={test_price} → total={sklearn_pred:.2f}")
    print(f"PyTorch 預測: quantity={test_quantity}, price={test_price} → total={pytorch_pred:.2f}")

    print("=" * 50)
    print("Step 10. 互動輸入模式 (輸入 q=quit 離開)") 
    while True:
        user_input = input("請輸入數量(quantity) 和 價格(price)，例如 5 100: ")
        if user_input.lower() in ["q", "quit", "exit"]:
            print("離開互動模式") 
            break
        try:
            quantity, price = map(float, user_input.split())
            sklearn_pred = model.predict([[quantity, price]])[0]
            pytorch_pred = predict_with_pytorch(loaded_model, quantity, price)
            print(f"➡ Scikit-learn 預測 total = {sklearn_pred:.2f}")
            print(f"➡ PyTorch 預測 total = {pytorch_pred:.2f}")
        except Exception as e:
            print("⚠ 輸入格式錯誤，請重新輸入 (例如 5 100)")
            
if __name__ == "__main__":
    main()
