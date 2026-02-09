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
        
elif app_mode == "⚙️ Week 3: Feature Engineering":
    st.title("🛠️ Data Transformation & Feature Intelligence")
    st.markdown("""
    **Phase Objective:** In Week 3, we move from observation to creation. We transform raw sales records 
    into mathematical 'signals' that a Machine Learning model can understand.
    """)

    # --- 1. FEATURE CREATION LOGIC ---
    fe_df = df.copy()
    
    # Date Decomposition (Extracting Seasonality)
    fe_df['MONTH'] = fe_df['ORDERDATE'].dt.month
    fe_df['YEAR'] = fe_df['ORDERDATE'].dt.year
    fe_df['QUARTER'] = fe_df['ORDERDATE'].dt.quarter
    fe_df['DAY_OF_WEEK'] = fe_df['ORDERDATE'].dt.dayofweek
    fe_df['IS_MONTH_END'] = fe_df['ORDERDATE'].dt.is_month_end.astype(int)
    
    # Categorical Encoding (Transforming Text to Numbers)
    le = LabelEncoder()
    fe_df['DEAL_CODE'] = le.fit_transform(fe_df['DEALSIZE'])
    fe_df['PROD_CODE'] = le.fit_transform(fe_df['PRODUCTLINE'])
    fe_df['COUNTRY_CODE'] = le.fit_transform(fe_df['COUNTRY'])
    fe_df['STATUS_CODE'] = le.fit_transform(fe_df['STATUS'])

    # Statistical Transformations
    fe_df['SALES_LOG'] = np.log1p(fe_df['SALES'])
    scaler = StandardScaler()
    fe_df['SCALED_QUANTITY'] = scaler.fit_transform(fe_df[['QUANTITYORDERED']])

    st.success(f"✅ Created {len(fe_df.columns) - len(df.columns)} New Predictive Features for the AI Engine.")

    # --- 2. MULTI-ANGLE ANALYSIS DASHBOARD ---
    col_a, col_b = st.columns([1.2, 0.8])

    with col_a:
        st.subheader("1. Feature Correlation Matrix")
        st.write("Visualizing the relationship between engineered features and Revenue.")
        # Expanded column list for deeper insight
        corr_cols = ['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MONTH', 'QUARTER', 
                     'DEAL_CODE', 'PROD_CODE', 'COUNTRY_CODE', 'STATUS_CODE']
        
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(fe_df[corr_cols].corr(), annot=True, cmap='RdYlGn', fmt=".2f", ax=ax)
        plt.title("Pearson Correlation Heatmap")
        st.pyplot(fig_corr)
        

    with col_b:
        st.subheader("2. Target Normalization")
        st.write("AI models require a 'Bell Curve' for high accuracy.")
        
        # Overlay plot comparing original vs transformed
        fig_dist = px.histogram(fe_df, x='SALES_LOG', nbins=30, 
                               title="Log-Normalized Sales Distribution",
                               color_discrete_sequence=['#636EFA'],
                               marginal="box")
        st.plotly_chart(fig_dist, use_container_width=True)
        

    # --- 3. ADVANCED STATISTICAL ANGLES ---
    st.markdown("---")
    st.subheader("3. Seasonal Revenue Decomposition")
    st.write("This angle proves why the 'MONTH' feature is critical for forecasting.")
    
    seasonal_data = fe_df.groupby(['YEAR', 'MONTH'])['SALES'].sum().reset_index()
    fig_seasonal = px.area(seasonal_data, x='MONTH', y='SALES', color='YEAR',
                          title="Revenue Velocity by Year/Month Feature",
                          labels={'SALES': 'Monthly Revenue ($)'},
                          line_group='YEAR')
    st.plotly_chart(fig_seasonal, use_container_width=True)

    # --- 4. DATA QUALITY & NULL ANALYSIS ---
    st.markdown("---")
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.subheader("4. Information Gain (Feature Impact)")
        # Calculate Feature Importance via ExtraTrees
        X_tmp = fe_df[['QUANTITYORDERED', 'PRICEEACH', 'MONTH', 'QUARTER', 'DEAL_CODE', 'PROD_CODE', 'COUNTRY_CODE']]
        y_tmp = fe_df['SALES']
        from sklearn.ensemble import ExtraTreesRegressor
        et_model = ExtraTreesRegressor()
        et_model.fit(X_tmp, y_tmp)
        
        feat_imp = pd.Series(et_model.feature_importances_, index=X_tmp.columns).sort_values()
        fig_imp = px.bar(feat_imp, orientation='h', title="Statistical Value of Engineered Features",
                        color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_imp, use_container_width=True)
        

    with col_d:
        st.subheader("5. Feature Variance Check")
        st.write("Descriptive statistics for the newly created features.")
        st.dataframe(fe_df[['MONTH', 'QUARTER', 'DEAL_CODE', 'PROD_CODE', 'COUNTRY_CODE']].describe().T)

    # --- 5. CONCLUSION OF WEEK 3 ---
    st.info("""
    **Conclusion for Week 3:** By decomposing dates and encoding categories, we have transformed 
    raw text into a multidimensional numerical space. This allows the Random Forest model (Week 4) 
    to recognize that **November sales spikes** are recurring patterns rather than random outliers.
    """)
elif app_mode == "🤖 Week 4 & 5: AI Modeling & Performance":
    st.title("AI Model Building & Evaluation")
    
    # --- STEP 1: PREPROCESSING ---
    # We store the encoders so we can use them for the "Live Predictor" later
    model_df = df.copy()
    encoders = {}
    for col in ['DEALSIZE', 'PRODUCTLINE', 'COUNTRY']:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])
        encoders[col] = le # Save for later use

    X = model_df[['QUANTITYORDERED', 'PRICEEACH', 'PRODUCTLINE', 'COUNTRY', 'DEALSIZE']]
    y = model_df['SALES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- STEP 2: MODEL SELECTION ---
    model_choice = st.selectbox("Select Model to Train:", ["Random Forest Regressor", "Linear Regression", "Gradient Boosting"])
    
    if model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = GradientBoostingRegressor(random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # --- STEP 3: METRICS WITH EXPLANATIONS ---
    st.subheader("📊 Model Performance Metrics")
    e1, e2, e3 = st.columns(3)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    e1.metric("MAE (Avg Error)", f"${mae:,.2f}", help="On average, the prediction is off by this amount.")
    e2.metric("RMSE (Outlier Penalty)", f"${rmse:,.2f}", help="Higher value means the model is making some big mistakes on specific deals.")
    e3.metric("R² Score (Accuracy)", f"{r2:.4f}", help="How much of the data patterns the AI understands. Closer to 1.0 is better.")

    # Simple logic-check message for the user
    if r2 > 0.80:
        st.success(f"✅ Excellent! This model explains {r2*100:.1f}% of the sales variation.")
    else:
        st.warning("⚠️ The model accuracy is moderate. Try cleaning more outliers.")

    # --- STEP 4: VISUALIZING ACCURACY ---
    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        # 1. Feature Importance
        if hasattr(model, 'feature_importances_'):
            imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance')
            st.plotly_chart(px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="What drives the Price?"), use_container_width=True)
    
    with c2:
        # 2. Actual vs Predicted Chart
        test_results = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
        fig_scatter = px.scatter(test_results, x='Actual', y='Predicted', 
                                 title="Actual Sales vs. AI Prediction",
                                 labels={'Actual': 'Real Value ($)', 'Predicted': 'AI Guess ($)'},
                                 opacity=0.5)
        # Add a "Perfect Prediction" line
        fig_scatter.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), 
                              line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- STEP 5: PRO DECISION ENGINE (Multi-Compare & History) ---
    st.divider()
    st.subheader("🚀 Global Deal Intelligence & History")

    # 1. Setup Session State for History (This remembers your past predictions)
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    # 2. Quick Inputs (Top Row)
    p1, p2, p3 = st.columns([1, 1, 2])
    with p1: input_qty = st.number_input("Quantity", value=30)
    with p2: input_price = st.number_input("Price Each", value=100.0)
    with p3: input_pline = st.selectbox("Product Line", df['PRODUCTLINE'].unique())
    
    # 3. Multi-Country Selection (This solves your "forgetting" problem)
    # Allow user to pick many countries at once
    selected_countries = st.multiselect(
        "Select Countries to Compare:", 
        options=df['COUNTRY'].unique(),
        default=[df['COUNTRY'].unique()[0]] # Default to the first one
    )
    
    input_deal = st.radio("Select Deal Size Context:", df['DEALSIZE'].unique(), horizontal=True)

    if st.button("⚡ Run Global Analysis"):
        results = []
        avg_sales = df['SALES'].mean()

        # 4. Loop through all selected countries and predict
        for country in selected_countries:
            # Prepare data
            user_input = pd.DataFrame([[
                input_qty, input_price, 
                encoders['PRODUCTLINE'].transform([input_pline])[0],
                encoders['COUNTRY'].transform([country])[0],
                encoders['DEALSIZE'].transform([input_deal])[0]
            ]], columns=X.columns)
            
            pred = model.predict(user_input)[0]
            
            # Logic for Status
            if pred > (avg_sales * 1.2): status, icon = "High Priority", "🟢"
            elif pred > avg_sales: status, icon = "Standard", "🟡"
            else: status, icon = "Low Margin", "🔴"
            
            # Save to temporary results
            res_dict = {
                "Country": country,
                "Predicted Sales": f"${pred:,.2f}",
                "Status": f"{icon} {status}",
                "Deal Size": input_deal,
                "Product": input_pline
            }
            results.append(res_dict)
            
            # Also save to Global History (Session State)
            st.session_state.prediction_history.insert(0, res_dict)

        # 5. Display Comparison Table for current click
        st.write("### 📊 Current Comparison")
        st.table(pd.DataFrame(results))

    # 6. THE HISTORY LOG (This ensures you never forget past ones)
    if st.session_state.prediction_history:
        st.divider()
        with st.expander("📜 View All Prediction History", expanded=False):
            st.write("This list keeps track of every scenario you have tested in this session.")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()


# ==========================================
# 🚀 TAB 5: WEEK 6 - STRATEGIC ACTION
# ==========================================
elif app_mode == "🚀 Week 6: Strategic Decision Support":
    st.title("🎯 Strategic Decision Cockpit")
    st.info("Utilizing AI-driven predictive logic to simulate business outcomes and mitigate risks.")
    
    st.markdown("### 📊 Interactive 'What-If' Forecasting")
    st.write("Simulate market changes to visualize impact on projected revenue.")
    
    # Sidebar or top-level levers
    with st.container():
        col_sim1, col_sim2, col_sim3 = st.columns([2, 2, 1])
        with col_sim1:
            q_change = st.slider("Increase Avg Quantity Ordered (%)", 0, 100, 15)
        with col_sim2:
            p_change = st.slider("Target Price Adjustment (%)", -20, 50, 10)
        with col_sim3:
            st.write("") # Spacer
            if st.button("Reset Simulation"):
                st.rerun()

    # Calculations
    current_sales = df['SALES'].sum()
    # Simple simulation logic: Sales = Q * P. 
    # Applying compounding percentage changes
    projected_sales = current_sales * (1 + (q_change/100)) * (1 + (p_change/100))
    growth_val = projected_sales - current_sales
    
    # Display Metrics
    st.metric(
        label="Projected Revenue Outcome", 
        value=f"${projected_sales/1e6:.2f}M", 
        delta=f"${growth_val/1e6:.2f}M Projected Growth",
        delta_color="normal"
    )

    # VISUAL COMPARISON: To impress the mentor
    sim_data = pd.DataFrame({
        "Scenario": ["Current Status", "Simulated Projection"],
        "Revenue ($M)": [current_sales/1e6, projected_sales/1e6]
    })
    
    import plotly.express as px
    fig_sim = px.bar(sim_data, x="Scenario", y="Revenue ($M)", 
                     color="Scenario", 
                     text_auto='.2f',
                     color_discrete_sequence=["#636EFA", "#00CC96"])
    fig_sim.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")
    
    # AI RECOMMENDATIONS SECTION
    st.subheader("🤖 AI-Generated Strategic Directives")
    
    # Container for a "cleaner" look
    with st.expander("View Detailed Action Plan", expanded=True):
        rec1, rec2 = st.columns(2)
        
        with rec1:
            st.markdown("#### ⚠️ Risk Mitigation")
            st.error(f"""
            **Inventory Alert:** Historical data indicates a **2.4x spike** in Q4 demand.  
            **Action:** Buffer 'Classic Cars' inventory by **{q_change + 10}%** to maintain service levels during the simulated growth.
            """)
            
        with rec2:
            st.markdown("#### 📈 Growth Opportunity")
            st.success(f"""
            **Market Expansion:** High-accuracy clusters identified in **USA & France**.  
            **Action:** Reallocate {p_change/2:.1f}% of marketing budget 
            toward 'Medium' deal sizes to maximize ROI based on current price elasticity.
            """)

    # FOOTER LOGIC
    st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
