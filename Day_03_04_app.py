# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:27:07 2026

@author: user
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# 設定頁面語系與標題
st.set_page_config(page_title="零售行銷決策支援系統", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 確保中文顯示

# --- 1. 載入模型與資料 (模擬環境) ---
@st.cache_resource
def load_data_and_model():
    # 這裡假設你的 CSV 已在同目錄下
    df = pd.read_csv('retail_marketing_experiment.csv')
    # 這裡加入 fillna，防止平均值計算出 NaN
    df = df.fillna(df.mean())    

    features = ['Social_Media_Ads', 'Search_Engine_Ads', 'Influencer_Marketing', 
                'Membership_Discount', 'Store_Size_sqft', 'Competitor_Price_Index', 
                'Local_Pop_Density', 'Staff_Training_Hours']
    X = df[features]
    y = df['Sales_Amount']
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)
    return df, model, features

df, model_xgb, features = load_data_and_model()

# --- 2. 側邊欄設定 (控制市場參數) ---
st.sidebar.header("📊 市場環境與成本設定")
comp_price_idx = st.sidebar.slider("競爭對手價格指數", 50, 150, int(df['Competitor_Price_Index'].mean()))
unit_cost = st.sidebar.number_input("產品單位成本", value=1200)
list_price = st.sidebar.number_input("產品預設原價", value=2000)
gross_margin_target = st.sidebar.slider("目標毛利率 (%)", 10, 80, 40) / 100

st.sidebar.markdown("---")
total_budget_limit = st.sidebar.number_input("行銷總預算上限 ($)", value=1000000)

# 準備基準資料 (以平均值為準，但替換掉側邊欄控制的變數)
base_values = df[features].mean().to_dict()
base_values['Competitor_Price_Index'] = comp_price_idx

# --- 3. 功能一：多管道預算優化 (Budget Allocation) ---
def optimize_budget(budget_limit):
    target_vars = ['Social_Media_Ads', 'Search_Engine_Ads', 'Influencer_Marketing']
    
    def objective(params):
        test_data = pd.DataFrame([base_values])
        for i, var in enumerate(target_vars):
            test_data[var] = params[i]
        pred_sales = model_xgb.predict(test_data)[0]
        profit = (pred_sales * gross_margin_target) - sum(params)
        return -profit

    constraints = {'type': 'ineq', 'fun': lambda params: budget_limit - sum(params)}
    bounds = [(0, budget_limit) for _ in target_vars]
    res = minimize(objective, [budget_limit/3]*3, bounds=bounds, constraints=constraints, method='SLSQP')
    return dict(zip(target_vars, res.x)), -res.fun

# --- 4. 功能二：折扣與動態定價分析 (Discount Optimization) ---
def analyze_discount():
    discounts = np.linspace(0, 800, 50)
    profits = []
    for d in discounts:
        test_data = pd.DataFrame([base_values])
        test_data['Membership_Discount'] = d
        pred_s = model_xgb.predict(test_data)[0]
        profits.append((list_price - d - unit_cost) * pred_s)
    return discounts, profits

# --- 5. Dashboard 佈局 ---
st.title("🚀 零售行銷利潤最大化 Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 最佳預算分配")
    best_config, max_profit = optimize_budget(total_budget_limit)
    
    # 【新增】檢查優化結果是否有效
    vals = np.array(list(best_config.values()))
    
    # 如果結果包含 NaN 或總和為 0 (代表沒投錢)
    if np.isnan(vals).any() or np.sum(vals) <= 0:
        st.warning("⚠️ 當前設定下無法獲利，優化器未找到可行解。請嘗試調高毛利率或降低成本。")
        st.metric("預計最大化毛利", "$ 0")
    else:
        st.metric("預計最大化毛利", f"${max_profit:,.0f}")
        
        # 繪製餅圖
        fig_pie, ax_pie = plt.subplots()
        labels = [f"{k}\n(${v:,.0f})" for k, v in best_config.items()]
        # 確保顏色數量正確
        colors = sns.color_palette("viridis", len(vals))
        ax_pie.pie(vals, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        st.pyplot(fig_pie)

with col2:
    st.subheader("🏷️ 折扣與動態定價分析")
    discounts, profits = analyze_discount()
    
    # 找出利潤最高點
    best_d = discounts[np.argmax(profits)]
    max_p = max(profits)
    
    st.metric("建議最優折扣", f"${best_d:,.0f}", delta=f"實質售價 ${list_price-best_d:,.0f}")
    
    # 畫利潤曲線圖
    fig_line, ax_line = plt.subplots()
    ax_line.plot(discounts, profits, lw=3, color='orange')
    ax_line.axvline(best_d, color='red', linestyle='--', label=f'最佳折扣: ${best_d:.0f}')
    ax_line.set_xlabel("會員折扣金額 ($)")
    ax_line.set_ylabel("預估總利潤 ($)")
    ax_line.legend()
    st.pyplot(fig_line)

st.markdown("---")
st.info("💡 **決策支援：** 當競爭對手價格下降時，請嘗試調整左側滑桿，觀察系統是否建議您加大『Membership_Discount』或重新分配廣告預算。")