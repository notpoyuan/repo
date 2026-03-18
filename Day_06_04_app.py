import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# ==========================================
# 1. 頁面配置與字體初始化
# ==========================================
st.set_page_config(page_title="心理圖譜導航系統", layout="wide")

def load_chinese_font():
    # 優先尋找專案資料夾內的字體
    font_path = os.path.join(os.getcwd(), 'fonts', 'msjh.ttc')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

prop = load_chinese_font()
if prop:
    plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 2. 數據加載與預處理 (Data Pipeline)
# ==========================================
@st.cache_data
def load_and_clean_data(file_path):
    # 讀取你的真實資料
    df = pd.read_csv(file_path)
    
    # 基礎清洗：移除全空行或處理缺失值 (依照你的數據特性調整)
    df_cleaned = df.dropna() 
    
    # 標準化 (PCA 對量綱非常敏感)
    scaler = StandardScaler()
    # 假設你的前 30 欄是 Q1-Q30
    df_scaled = scaler.fit_transform(df_cleaned.iloc[:, :30]) 
    
    return df_cleaned, df_scaled

# 載入資料
if os.path.exists("raw_survey.csv"):
    df_raw, df_scaled = load_and_clean_data("raw_survey.csv")
else:
    st.error("❌ 找不到 raw_survey.csv，請確認檔案位置。")
    st.stop()

# ==========================================
# 3. 核心運算：PCA & KMeans
# ==========================================
@st.cache_resource
def perform_analysis(df_scaled):
    # PCA 降維
    pca = PCA(n_components=5)
    pca_scores = pca.fit_transform(df_scaled)
    
    # K-Means 分群
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_scores[:, :3]) # 使用前三主成分分群
    
    # 建立 Results DataFrame
    col_names = ['品牌溢價忠誠度', '極致價值探索度', '數位通路依賴度', '補償性情緒消費', '獵奇衝動購買欲']
    df_pca = pd.DataFrame(pca_scores[:, :5], columns=col_names)
    df_pca['Cluster'] = clusters
    
    return pca, df_pca

pca, df_pca = perform_analysis(df_scaled)
segment_profile = df_pca.groupby('Cluster').mean()

# ==========================================
# 4. Streamlit UI 介面
# ==========================================
st.sidebar.title("📊 專案導航")
menu = st.sidebar.radio("切換分析環節", 
    ["專案概述", "數據特徵體檢", "PCA 心理維度萃取", "K-Means 客群畫像", "行銷決策輸出"])

# --- 頁面：專案概述 ---
if menu == "專案概述":
    st.title("🚀 【心理圖譜導航】客戶精準分群決策系統")
    st.markdown("""
    ### 專案動機
    在面對 **30 項消費行為特徵** 時，傳統的交叉分析難以抓出清晰的「人設」。本專案透過 **PCA 降維** 提煉心理維度，並利用 **K-Means 聚類** 實作精準分群。
    
    ### 技術棧
    * **數據工程**: Python, Pandas, Scikit-Learn
    * **降維算法**: Principal Component Analysis (PCA)
    * **聚類算法**: K-Means Clustering
    * **視覺化**: Matplotlib, Seaborn (雷達圖實作)
    """)
    st.image("https://images.unsplash.com/photo-1551288049-bbda4833effb?auto=format&fit=crop&q=80&w=1000", caption="從雜亂數據到結構化決策")

# --- 頁面：數據特徵體檢 ---
elif menu == "數據特徵體檢":
    st.title("🔍 原始數據體檢 (Exploratory Data Analysis)")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("數據維度：", df_raw.shape)
        st.write("缺失值處理：已剔除")
    with col2:
        st.write("前 5 筆資料預覽：")
        st.dataframe(df_raw.head())
    
    st.subheader("特徵相關性矩陣 (為何需要 PCA？)")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_raw.iloc[:, :15].corr(), cmap='coolwarm', annot=False) # 僅展示前15題示範
    st.pyplot(fig)
    st.caption("觀察到題目間存在高度共線性，證實了降維的必要性。")

# --- 頁面：PCA 心理維度萃取 ---
elif menu == "PCA 心理維度萃取":
    st.title("🧪 PCA 維度解析")
    
    exp_var = pca.explained_variance_ratio_ * 100
    cols = st.columns(5)
    for i, v in enumerate(exp_var):
        cols[i].metric(f"PC{i+1}", f"{v:.1f}%")

    st.markdown("---")
    st.subheader("維度定義 (基於 Loading 負荷量)")
    selected_pc = st.selectbox("選擇要檢視的維度：", df_pca.columns[:-1])
    
    # 顯示該維度正負向貢獻最高的題目
    pc_index = df_pca.columns.tolist().index(selected_pc)
    loadings = pd.Series(pca.components_[pc_index], index=df_raw.columns[:30])
    
    c1, c2 = st.columns(2)
    c1.write("🟢 正向驅動因素：")
    c1.dataframe(loadings.sort_values(ascending=False).head(5))
    c2.write("🔴 負向排斥因素：")
    c2.dataframe(loadings.sort_values(ascending=True).head(5))

# --- 頁面：K-Means 客群畫像 ---

elif menu == "K-Means 客群畫像":
    st.title("🎯 客群特徵雷達圖")

    # --- 1. 使用 Columns 縮小圖表顯示區域 ---
    # 建立三欄，比例為 1:2:1，將圖放在中間那欄
    col_left, col_mid, col_right = st.columns([1, 2, 1])

    with col_mid:
        # 稍微縮小 figsize (原本是 8,8 改為 6,6)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        categories = segment_profile.columns.tolist()
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['客群 0: 傳統務實', '客群 1: 品牌守護', '客群 2: 感性冒險', '客群 3: 數位精算']
        
        for i in range(len(segment_profile)):
            values = segment_profile.iloc[i].tolist()
            values += values[:1]
            ax.plot(angles, values, color=colors[i], linewidth=1.5, label=labels[i])
            ax.fill(angles, values, color=colors[i], alpha=0.05)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 調整標籤字體大小，避免圖縮小後字擠在一起
        plt.xticks(angles[:-1], categories, fontproperties=prop, size=9)
        plt.yticks([-1, 0, 1], ["-1", "0", "1"], color="grey", size=7)
        plt.ylim(-1.5, 2.0)
        
        # 將圖例放在圖下方，節省空間
        ax.legend(prop=prop, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize='small')
        
        st.pyplot(fig, use_container_width=True)

    # --- 2. 數據細節放在下方 ---
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("👥 各族群人數佔比")
        # 使用 Streamlit 原生 bar_chart，體積小且美觀
        st.bar_chart(df_pca['Cluster'].value_counts(normalize=True))
    with c2:
        st.subheader("📝 特徵中心點數值")
        st.dataframe(segment_profile.style.background_gradient(cmap='RdBu_r', axis=0))




# --- 頁面：行銷決策輸出 ---
elif menu == "行銷決策輸出":
    st.title("💡 策略落地建議")
    st.table(pd.DataFrame({
        "客群": ["客群 3 (數位精算)", "客群 1 (品牌守護)", "客群 2 (感性衝動)", "客群 0 (傳統務實)"],
        "核心策略": ["價格導向 / 比價推播", "價值導向 / VIP 會員制", "情感導向 / 社群美學", "通路導向 / 線下接觸"],
        "建議渠道": ["App / 官網", "EDM / 私域流量", "IG / TikTok", "SMS / 實體門市"]
    }))
    
    if st.button("📥 匯出分群名單 (CSV)"):
        df_raw['Cluster_Label'] = df_pca['Cluster']
        csv = df_raw.to_csv(index=False).encode('utf-8-sig')
        st.download_button("點此下載", data=csv, file_name="segmentation_results.csv", mime="text/csv")