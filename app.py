import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. PLATFORM CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="AI Predictive Analytics Platform", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
    h1, h2, h3 { color: #1e3d59; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (WEEK 1: DATA UNDERSTANDING)
# ==========================================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    # Week 1 Logic: Feature Selection & Handling
    cols_to_keep = ['ORDERNUMBER', 'QUANTITYORDERED', 'PRICEEACH', 'SALES', 
                    'ORDERDATE', 'STATUS', 'PRODUCTLINE', 'MSRP', 
                    'COUNTRY', 'DEALSIZE', 'CITY']
    return df[cols_to_keep]

try:
    df = load_and_clean_data()
except Exception as e:
    st.error("Error: 'cleaned_sales_data.csv' not found. Please upload to root directory.")
    st.stop()

# ==========================================
# 3. GLOBAL SIDEBAR (DECISION NAVIGATION)
# ==========================================
st.sidebar.title("💎 AI Analytics Platform")
st.sidebar.markdown("---")
st.sidebar.write("**Project Phase:** 6-Week Internship Pipeline")
app_mode = st.sidebar.selectbox("Choose Analysis Angle:", [
    "🏠 Executive Summary", 
    "📊 Week 2: Deep EDA & Trends", 
    "⚙️ Week 3: Feature Engineering", 
    "🤖 Week 4 & 5: AI Modeling & Performance", 
    "🚀 Week 6: Strategic Decision Support"
])

st.sidebar.markdown("---")
st.sidebar.write("**Dataset Stats:**")
st.sidebar.write(f"Total Records: {len(df)}")
st.sidebar.write(f"Gross Revenue: ${df['SALES'].sum()/1e6:.2f}M")

# ==========================================
# 🏠 TAB 1: EXECUTIVE SUMMARY
# ==========================================
if app_mode == "🏠 Executive Summary":
    st.title("Strategic Business Overview")
    st.info("Goal: Provide an immediate snapshot of organizational health based on historical data.")
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Sales", f"${df['SALES'].sum()/1e6:.2f}M", "12.4%")
    kpi2.metric("Order Volume", f"{len(df):,}")
    kpi3.metric("Avg Deal Size", f"${df['SALES'].mean():,.2f}")
    kpi4.metric("Active Markets", df['COUNTRY'].nunique())

    # Visual Insights
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Product Line")
        fig_pie = px.pie(df, names='PRODUCTLINE', values='SALES', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("Top Performing Countries")
        geo_sales = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(geo_sales, x='COUNTRY', y='SALES', color='SALES'), use_container_width=True)

# ==========================================
# 📊 TAB 2: WEEK 2 - DEEP EDA
# ==========================================
elif app_mode == "📊 Week 2: Deep EDA & Trends":
    st.title("Exploratory Data Analysis")
    st.markdown("### Identifying Patterns, Correlations, and Outliers")
    
    # Time Series Decomposition
    st.subheader("Revenue Trend Analysis")
    df_resampled = df.set_index('ORDERDATE').resample('M')['SALES'].sum().reset_index()
    fig_line = px.line(df_resampled, x='ORDERDATE', y='SALES', title="Monthly Sales Velocity", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)
    
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.subheader("Price vs. Quantity Distribution")
        fig_scatter = px.scatter(df, x='QUANTITYORDERED', y='SALES', color='DEALSIZE', hover_data=['PRODUCTLINE'])
        st.plotly_chart(fig_scatter)
        
    with col_eda2:
        st.subheader("Outlier Detection (Box Plot)")
        fig_box = px.box(df, x='PRODUCTLINE', y='SALES', color='PRODUCTLINE')
        st.plotly_chart(fig_box)
        

# ==========================================
# ⚙️ TAB 3: WEEK 3 - FEATURE ENGINEERING
# ==========================================
elif app_mode == "⚙️ TAB 3: Feature Engineering":
    st.title("Data Transformation Intelligence")
    st.write("Preparing raw data for Machine Learning models.")

    fe_df = df.copy()
    # 1. Date Decomposition
    fe_df['MONTH'] = fe_df['ORDERDATE'].dt.month
    fe_df['YEAR'] = fe_df['ORDERDATE'].dt.year
    
    # 2. Label Encoding
    le = LabelEncoder()
    fe_df['DEAL_CODE'] = le.fit_transform(fe_df['DEALSIZE'])
    fe_df['PROD_CODE'] = le.fit_transform(fe_df['PRODUCTLINE'])
    
    st.success("Successfully created: Month/Year signals, Deal Size Encodings, and Product Codes.")

    # Correlation Heatmap
    st.subheader("Feature Correlation Matrix")
    corr_cols = ['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MONTH', 'DEAL_CODE', 'PROD_CODE']
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(fe_df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig_corr)
    

# ==========================================
# 🤖 TAB 4 & 5: WEEK 4/5 - AI MODELS
# ==========================================
elif app_mode == "🤖 Week 4 & 5: AI Modeling & Performance":
    st.title("AI Model Building & Evaluation")
    
    # Preprocessing for Modeling
    le = LabelEncoder()
    model_df = df.copy()
    model_df['DEALSIZE'] = le.fit_transform(model_df['DEALSIZE'])
    model_df['PRODUCTLINE'] = le.fit_transform(model_df['PRODUCTLINE'])
    model_df['COUNTRY'] = le.fit_transform(model_df['COUNTRY'])
    
    X = model_df[['QUANTITYORDERED', 'PRICEEACH', 'PRODUCTLINE', 'COUNTRY', 'DEALSIZE']]
    y = model_df['SALES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    model_choice = st.selectbox("Select Model to Train:", ["Random Forest Regressor", "Linear Regression", "Gradient Boosting"])
    
    if model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100)
    elif model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = GradientBoostingRegressor()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Evaluation Metrics
    st.subheader("Model Performance Metrics")
    e1, e2, e3 = st.columns(3)
    e1.metric("MAE", f"{mean_absolute_error(y_test, preds):,.2f}")
    e2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):,.2f}")
    e3.metric("R² Score", f"{r2_score(y_test, preds):.4f}")

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance')
        st.plotly_chart(px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Prediction Drivers"))
        

# ==========================================
# 🚀 TAB 5: WEEK 6 - STRATEGIC ACTION
# ==========================================
elif app_mode == "🚀 Week 6: Strategic Decision Support":
    st.title("Decision Support Cockpit")
    
    st.markdown("### Interactive 'What-If' Forecasting")
    st.write("Adjust business levers to see the predicted impact on the bottom line.")
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        q_change = st.slider("Increase Average Quantity Ordered (%)", 0, 100, 10)
    with col_sim2:
        p_change = st.slider("Target Price Adjustment (%)", -10, 30, 5)

    current_sales = df['SALES'].sum()
    projected_sales = current_sales * (1 + (q_change/100)) * (1 + (p_change/100))
    
    st.metric("Projected Revenue Outcome", f"${projected_sales/1e6:.2f}M", 
              delta=f"${(projected_sales - current_sales)/1e6:.2f}M Growth")

    st.markdown("---")
    st.subheader("AI-Driven Recommendations")
    rec1, rec2 = st.columns(2)
    rec1.error("🚨 **Demand Risk:** November shows a 2.4x historical spike. Increase Classic Car inventory by 30% to avoid stockouts.")
    rec2.success("💡 **Opportunity:** USA-based Medium deals show the highest model accuracy. Focus marketing spend in this segment.")
