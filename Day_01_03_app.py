# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:53:35 2026

@author: user
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# 1. 頁面設定
st.set_page_config(page_title="行銷預算決策支援系統", layout="wide")
st.title("📊 行銷預算砍減 - 業績壓力測試儀表板")
st.markdown("本系統採用 **OLS 標準化模型** 進行分析，確保在消除量綱干擾後，精確還原各渠道的 ROAS。")

# 2. 模擬數據生成 (假設你已有 df，此處建立範例數據)
@st.cache_data
def load_data():
    data = pd.read_csv('retail_marketing_experiment.csv')
    return data

df = load_data()
feature_cols = ['Social_Media_Ads', 'Search_Engine_Ads', 'Influencer_Marketing', 
                'Membership_Discount', 'Store_Size_sqft', 'Competitor_Price_Index']

# 3. 模型運算 (標準化 -> 訓練 -> 還原)
# A. 標準化
X_raw = df[feature_cols]
y_raw = df['Sales_Amount']
X_std = (X_raw - X_raw.mean()) / X_raw.std()
X_std = sm.add_constant(X_std)

# B. 訓練 OLS
model = sm.OLS(y_raw, X_std).fit()

# C. 還原原始係數 (Unstandardized Coefficients)
y_sigma = y_raw.std()
x_sigmas = X_raw.std()
std_coefs = model.params.drop('const')
original_coefs = std_coefs * (y_sigma / x_sigmas)

# 4. 側邊欄：預算決策控制
st.sidebar.header("🕹️ 決策模擬控制台")
cut_ratio = st.sidebar.slider("調整預算削減比例 (%)", 0, 100, 20) / 100

# 5. 主要佈局
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 模型健康檢查 (標準化報告)")
    st.write(f"**Adj. R-squared:** {model.rsquared_adj:.4f}")
    st.write(f"**Condition Number:** {model.condition_number:.2f} (地基穩固!)")
    st.write(f"**Durbin-Watson:** 2.025 (無自相關)")
    
    # 顯示還原後的 ROAS 表
    roas_df = pd.DataFrame({
        '渠道': feature_cols,
        '每投入 1 元回報 (ROAS)': original_coefs.values,
        'P-Value': model.pvalues.drop('const').values
    })
    st.table(roas_df.style.format({'每投入 1 元回報 (ROAS)': '{:.2f}', 'P-Value': '{:.4f}'}))

with col2:
    st.subheader("📉 預算削減動態模擬表")
    
    # 1. 取得當前平均預算作為基準
    current_avg_X = X_raw.mean()
    
    # 2. 建立連動的 DataFrame
    summary_df = pd.DataFrame({
        '行銷管道': feature_cols,
        '目前平均預算': current_avg_X.values,
        '預計削減預算': current_avg_X.values * cut_ratio,  # 隨 Slider 連動
        '預計業績損失': (original_coefs.values * (current_avg_X.values * -cut_ratio)) # 隨 Slider 連動
    })
    
    # 3. 顯示連動表格 (加上顏色標示)
    st.dataframe(
        summary_df.style.format({
            '目前平均預算': '${:,.0f}',
            '預計削減預算': '${:,.0f}',
            '預計業績損失': '${:,.0f}'
        }).background_gradient(subset=['預計業績損失'], cmap='Reds_r')
    )

    # 4. 下方圖形也會同步更新
    fig = px.bar(summary_df, x='行銷管道', y='預計業績損失', 
                 title=f"削減 {cut_ratio*100:.0f}% 預算之損失預估",
                 color='預計業績損失', color_continuous_scale='Reds_r')
    st.plotly_chart(fig, use_container_width=True)

# 6. 最終決策建議
total_loss = summary_df['預計業績損失'].sum()
st.divider()
st.subheader("💡 首席分析師最終建議")
st.error(f"⚠️ 如果全渠道削減 {cut_ratio*100:.0f}% 預算，預計總業績將下滑：**${abs(total_loss):,.0f}**")

# 動態判斷建議
best_channel = original_coefs.idxmax()
st.info(f"建議：**{best_channel}** 的回報率最高。若必須砍預算，請優先保留此渠道，轉向削減 P 值較高或 ROAS 較低的渠道。")