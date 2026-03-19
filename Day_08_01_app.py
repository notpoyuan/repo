# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:28:33 2026

@author: user
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb

# 設定頁面資訊
st.set_page_config(page_title="流失預警與自動決策系統", layout="wide")

# --- 1. 資料加載與處理 (Cache 確保速度) ---
@st.cache_data
def load_and_process_data():
    df_b = pd.read_csv('user_behavior_30.csv', index_col='User_ID')
    df_v = pd.read_csv('user_value_ltv.csv', index_col='User_ID')
    df_m = pd.read_csv('marketing_cost.csv')
    df = df_b.join(df_v)
    
    # PCA 降維
    X = df.filter(regex='Q')
    scaler = StandardScaler()
    X_pca = PCA(n_components=2).fit_transform(scaler.fit_transform(X))
    df['活躍維度'] = X_pca[:, 0]
    df['摩擦維度'] = X_pca[:, 1]
    
    # K-Means 分群
    cluster_features = df[['活躍維度', '摩擦維度', 'Total_Spend']]
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(cluster_features))
    cluster_map = {0: '潛力新客', 1: '核心高價', 2: '價格敏感', 3: '羊毛黨'}
    df['客群名稱'] = df['Cluster'].map(cluster_map)
    
    # XGBoost 預測
    X_model = df[['活躍維度', '摩擦維度', 'Total_Spend', 'Avg_Order_Value']]
    model = xgb.XGBClassifier(eval_metric='logloss').fit(X_model, df['Is_Churn'])
    df['Churn_Prob'] = model.predict_proba(X_model)[:, 1]
    
    return df, df_m

df_full, df_marketing = load_and_process_data()

# --- 2. 側邊欄控制 (Sidebar) ---
st.sidebar.title("🛠️ 決策參數設定")
st.sidebar.markdown("---")
churn_threshold = st.sidebar.slider("流失機率預警門檻", 0.0, 1.0, 0.5)
recovery_rate = st.sidebar.slider("發券後預期流失降幅 (%)", 0, 100, 30) / 100

# --- 3. 頁面設計 ---
st.title("📊 流失預警與自動化預防系統")
st.markdown("本系統整合 **PCA 降維、K-Means 分群與 XGBoost 預測**，自動產出最優化補償建議。")

# 區塊 A：關鍵指標 (Key Metrics)
col1, col2, col3, col4 = st.columns(4)
col1.metric("總樣本數", len(df_full))
col2.metric("平均流失風險", f"{df_full['Churn_Prob'].mean():.1%}")
col3.metric("高風險人數", len(df_full[df_full['Churn_Prob'] > churn_threshold]))
col4.metric("核心高價人數", len(df_full[df_full['客群名稱'] == '核心高價']))

st.markdown("---")

# 區塊 B：視覺化分析
tab1, tab2 = st.tabs(["🎯 客群分佈 (PCA & Cluster)", "📉 流失風險分佈"])

with tab1:
    fig_pca, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_full, x='活躍維度', y='摩擦維度', hue='客群名稱', alpha=0.6, ax=ax)
    plt.title("客群心理維度分佈圖 (PCA Results)")
    st.pyplot(fig_pca)

with tab2:
    fig_churn, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df_full['Churn_Prob'], bins=30, kde=True, color='red', ax=ax)
    plt.axvline(churn_threshold, color='black', linestyle='--')
    plt.title("流失機率分佈 (Red Line = Threshold)")
    st.pyplot(fig_churn)

# 區塊 C：自動化決策輸出
st.markdown("### 🚀 自動化補償預案清單")

# 執行 EV 決策邏輯 (使用滑桿參數)
def streamlit_ev_logic(row):
    p_churn = row['Churn_Prob']
    p_stay = 1 - p_churn
    ltv = row['Total_Spend']
    # 這裡連結 df_marketing 的成本
    m_row = df_marketing[df_marketing['Cluster_Name'] == row['客群名稱']].iloc[0]
    cost = m_row['Coupon_Cost']
    redeem_r = m_row['Redeem_Rate']

    if row['客群名稱'] == '核心高價':
        if p_churn > 0.7: return '專人電話聯繫', 0
        elif p_churn > 0.5: return '升級 VIP 專屬權益 (無券)', 0
        else: return '無須干預 (維持現狀)', 0

    ev_no_action = p_stay * ltv
    ev_with_coupon = (p_stay + (p_churn * recovery_rate)) * ltv - (cost * redeem_r)
    
    if ev_with_coupon > ev_no_action and p_churn > churn_threshold:
        return '自動發送折扣券', cost
    else:
        return '無須干預 (維持現狀)', 0

# 應用決策
df_final = df_full.reset_index()
# 併入成本表資訊
df_final = pd.merge(df_final, df_marketing, left_on='客群名稱', right_on='Cluster_Name', how='left')

df_final[['Suggested_Action', 'Coupon_Value']] = df_final.apply(
    lambda row: pd.Series(streamlit_ev_logic(row)), axis=1
)

# 顯示最終清單
final_report = df_final[df_final['Suggested_Action'] != '無須干預 (維持現狀)'][['User_ID', '客群名稱', 'Churn_Prob', 'Suggested_Action', 'Coupon_Value']]

st.dataframe(final_report.style.format({'Churn_Prob': '{:.2%}'}), use_container_width=True)

# 下載按鈕
csv = final_report.to_csv(index=False).encode('utf-8-sig')
st.download_button("📥 下載行動名單 CSV", data=csv, file_name='action_list.csv', mime='text/csv')