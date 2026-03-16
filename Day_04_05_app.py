import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# 設定繁體中文顯示（若在 Colab 或本地環境需安裝字體，此處以標籤英文為主確保執行成功）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀入資料
df_final = pd.read_csv('sales_rfm.csv')


#%%
# 1、建構RFM欄位

# 1. 將字串轉換為 datetime 格式 (這是最關鍵的一步)
df_final['OrderDate'] = pd.to_datetime(df_final['OrderDate'])

# 假設分析基準日為 2026/01/01
analysis_date = datetime(2026, 1, 1)

# 將「成千上萬筆的原始交易流水帳」壓縮成「以客戶為單位」的商業價值指標。
rfm_table = df_final.groupby('CustomerID').agg({
    # R: 距離基準日幾天 (越小越新)
    'OrderDate': lambda x: (analysis_date - x.max()).days,
    # F: 總購買次數
    'Order_ID': 'count',
    # M: 總消費金額
    'Amount': 'sum'
})

# 重新命名欄位，讓意義明確
rfm_table.rename(columns={
    'OrderDate': 'Recency',
    'Order_ID': 'Frequency',
    'Amount': 'Monetary'
}, inplace=True)


#%%

# 執行 K-means 分群
# 我們直接對 rfm_table 進行運算，並假設分為 4 群。


from sklearn.cluster import KMeans

# 1. 設定群數 (假設分 4 群)
# 2. 初始化模型 (n_init=10 是為了避免警告並確保穩定性)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)


# 3. 執行分群並取得標籤 (使用 fit_predict)
# 注意：這會回傳一個 1 維陣列，長度等於你的客戶數
rfm_table['Cluster'] = kmeans.fit_predict(rfm_table)

# 4. 檢查結果
print("--- 分群後的資料摘要 ---")
print(rfm_table.head())

# 5. 查看各群的平均值，理解分群邏輯
cluster_analysis = rfm_table.groupby('Cluster').mean()


#%%

# 定義名稱

# 建立一個對照表
mapping = {1: '鑽石客戶 (高貢獻)', 3: '黃金 (潛力高)', 0: '一般客戶', 2: '小資 (流失客戶)'}
rfm_table['Segment'] = rfm_table['Cluster'].map(mapping)

#%%

# 頁面標題與設定
st.set_page_config(page_title="RFM 商務決策 Dashboard", layout="wide")
st.title("📊 RFM 客戶分群 What-if 模擬看板")

# --- 側邊欄控制項 (What-if 參數) ---
st.sidebar.header("🔍 What-if 參數設定")

# 問題一參數
st.sidebar.subheader("1. 鑽石客防流失 (C1)")
r_threshold = st.sidebar.slider("高風險 Recency 門檻 (天)", 30, 120, 60)
retention_rate = st.sidebar.slider("免運券挽回率 (%)", 0, 100, 50) / 100

# 問題二參數
st.sidebar.subheader("2. 黃金客升級 (C3)")
upsell_target = st.sidebar.slider("預期客單價提升 (%)", 0, 50, 15) / 100

# 問題三參數
st.sidebar.subheader("3. 預算挪移優化 (C2 -> C0)")
cost_per_sms = st.sidebar.number_input("每封簡訊成本 ($)", value=2.0)
conversion_lift = st.sidebar.slider("一般客轉換率提升 (%)", 0.0, 1.0, 0.1, step=0.05) / 100

# --- 資料模擬 (假設你已經載入了 rfm_table) ---
# 此處建議先載入你的 rfm_table，這裡範例模擬關鍵數據
# rfm_table = pd.read_csv('your_processed_rfm.csv') 

# --- 計算邏輯 ---

# 1. 鑽石客效益
c1_data = rfm_table[rfm_table['Cluster'] == 1]
high_risk_c1 = c1_data[c1_data['Recency'] > r_threshold]
saved_revenue = high_risk_c1['Monetary'].sum() * 0.2 * retention_rate

# 2. 黃金客效益
c3_data = rfm_table[rfm_table['Cluster'] == 3]
revenue_lift = c3_data['Monetary'].sum() * upsell_target

# 3. 預算挪移效益
c2_data = rfm_table[rfm_table['Cluster'] == 2]
c0_data = rfm_table[rfm_table['Cluster'] == 0]
saved_budget = len(c2_data) * cost_per_sms
lost_rev_c2 = (len(c2_data) * 0.005) * c2_data['Monetary'].mean()
gained_rev_c0 = (len(c0_data) * conversion_lift) * c0_data['Monetary'].mean()
net_impact = gained_rev_c0 - lost_rev_c2 + (saved_budget / 0.2) # 簡化假設預算與營收比

# --- 介面佈局 ---

# 第一排：關鍵指標 (KPI Cards)
col1, col2, col3 = st.columns(3)
col1.metric("預計救回金額 (C1)", f"${saved_revenue:,.0f}", delta=f"{len(high_risk_c1)} 人風險")
col2.metric("預計增長營收 (C3)", f"${revenue_lift:,.0f}", delta=f"+{upsell_target*100:.0%}")
col3.metric("預算挪移淨效益", f"${net_impact:,.0f}", delta=f"${saved_budget} 省下成本")

st.markdown("---")

# 第二排：視覺化圖表
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("客戶價值分佈 (Monetary vs Recency)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=rfm_table, x='Recency', y='Monetary', hue='Cluster', palette='viridis', ax=ax)
    plt.axvline(r_threshold, color='red', linestyle='--', label='風險線')
    st.pyplot(fig)

with right_col:
    st.subheader("各群組平均貢獻")
    fig2, ax2 = plt.subplots()
    avg_m = rfm_table.groupby('Segment')['Monetary'].mean().sort_values()
    avg_m.plot(kind='barh', color='skyblue', ax=ax2)
    st.pyplot(fig2)

st.info("💡 調整左側邊欄參數，儀表板將即時更新運算結果。")