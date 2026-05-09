import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, r2_score, mean_squared_error
import io

# Set page config
st.set_page_config(
    page_title="Clothes Inventory & Customer Behaviour Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper Function: Preprocess Data
@st.cache_data
def preprocess_data(df):
    df_clean = df.copy()
    
    # Handle missing values
    num_cols = df_clean.select_dtypes(include=np.number).columns
    cat_cols = df_clean.select_dtypes(exclude=np.number).columns
    
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            
    # Encode categorical variables
    encoders = {}
    df_encoded = df_clean.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        
    return df_clean, df_encoded, encoders

def main():
    st.title("🛍️ Clothes Inventory & Customer Behaviour Dashboard")
    st.markdown("Analyze customer behaviour, predict inventory risk, and get actionable recommendations.")

    # Sidebar
    st.sidebar.header("Navigation")
    menu = st.sidebar.radio("Go to:", [
        "1. AWS Architecture", 
        "2. Dataset Overview", 
        "3. Visualizations", 
        "4. ML: Classification (Risk)", 
        "5. ML: Regression (Sales)", 
        "6. ML: Customer Segmentation",
        "7. Prescriptive Analytics"
    ])

    st.sidebar.markdown("---")
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload Cleaned CSV", type=['csv'])

    if menu == "1. AWS Architecture":
        st.header("☁️ Conceptual AWS Integration Pipeline")
        st.markdown("""
        This dashboard conceptually represents the final stage of an AWS-based Machine Learning Pipeline.
        """)
        
        # Draw a flowchart
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = ["AWS S3 (Raw Data)", "AWS Glue (Data Cleaning)", "AWS SageMaker (ML Models)", "AWS RDS/Redshift (Processed Data)", "Streamlit / QuickSight (Dashboard)"],
              color = ["#FF9900", "#FF9900", "#FF9900", "#FF9900", "#00A1C9"]
            ),
            link = dict(
              source = [0, 1, 2, 1], # indices correspond to labels
              target = [1, 2, 4, 3],
              value = [10, 10, 8, 2]
            )
        )])
        fig.update_layout(title_text="Data Pipeline Architecture", font_size=12, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Pipeline Flow:**
        1. **Storage:** Raw CSV datasets are uploaded and stored securely in **AWS S3**.
        2. **Processing:** **AWS Glue** performs ETL (Extract, Transform, Load) to handle missing values and encode categorical features.
        3. **Machine Learning:** Models (Random Forest, Linear Regression, K-Means) are trained and deployed using **AWS SageMaker**.
        4. **Visualization:** The predictions and prescriptive analytics are served through an interactive dashboard (like this one) hosted on **EC2** or **App Runner**.
        """)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df_clean, df_encoded, encoders = preprocess_data(df)
            
            # Identify columns heuristically if not strictly named
            target_col_classification = st.sidebar.selectbox("Select Classification Target (e.g. Risk_Level)", df.columns, index=len(df.columns)-1 if "Risk" not in str(df.columns) else [i for i, col in enumerate(df.columns) if "Risk" in col][0])
            target_col_regression = st.sidebar.selectbox("Select Regression Target (e.g. Sales, Price)", df.columns, index=0)
            
            if menu == "2. Dataset Overview":
                st.header("📊 Dataset Overview")
                st.write("First 10 rows of the dataset:")
                st.dataframe(df.head(10))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Data Summary")
                    st.write(df.describe())
                with col2:
                    st.subheader("Missing Values Check")
                    missing = df.isnull().sum()
                    st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values found! 🎉")
                    
                st.subheader("Correlation Heatmap")
                fig = px.imshow(df_encoded.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
                st.plotly_chart(fig, use_container_width=True)

            elif menu == "3. Visualizations":
                st.header("📈 Data Visualizations")
                
                cat_cols = df.select_dtypes(exclude=np.number).columns
                num_cols = df.select_dtypes(include=np.number).columns
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Category-wise Distribution")
                    if len(cat_cols) > 0:
                        selected_cat = st.selectbox("Select Categorical Column", cat_cols)
                        fig = px.pie(df, names=selected_cat, title=f"Distribution of {selected_cat}", hole=0.4)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No categorical columns found for Pie Chart.")
                
                with col2:
                    st.subheader("Numerical Distribution")
                    if len(num_cols) > 0:
                        selected_num = st.selectbox("Select Numerical Column (e.g. Price)", num_cols)
                        fig = px.histogram(df, x=selected_num, title=f"Distribution of {selected_num}", marginal="box", nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numerical columns found.")
                        
                st.subheader("Feature Relationships (Customer Buying Behaviour)")
                if len(num_cols) >= 2:
                    x_axis = st.selectbox("X-Axis", num_cols, index=0)
                    y_axis = st.selectbox("Y-Axis", num_cols, index=1 if len(num_cols) > 1 else 0)
                    color_col = st.selectbox("Color By", df.columns, index=0)
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis} by {color_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif menu == "4. ML: Classification (Risk)":
                st.header("🛡️ Classification Model: Predict Risk Level")
                st.markdown("Predict the inventory risk (High / Medium / Low) to manage stock effectively.")
                
                clf_model_choice = st.radio("Select Model", ["Random Forest", "Decision Tree"])
                
                X = df_encoded.drop(columns=[target_col_classification])
                y = df_encoded[target_col_classification]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if clf_model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = DecisionTreeClassifier(random_state=42)
                    
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Model Selected", clf_model_choice)
                col2.metric("Accuracy Score", f"{acc:.2%}")
                col3.metric("Test Set Size", len(y_test))
                
                st.subheader("Model Evaluation")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', 
                                     labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                with c2:
                    st.markdown("**Classification Report**")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    df_report = pd.DataFrame(report).transpose()
                    st.dataframe(df_report.style.background_gradient(cmap='Greens'))
                
                st.subheader("Feature Importance")
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)
                    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Which features drive risk?")
                    st.plotly_chart(fig_imp, use_container_width=True)

            elif menu == "5. ML: Regression (Sales)":
                st.header("📈 Regression Model: Predict Sales Trends")
                st.markdown("Use Linear Regression to forecast sales or demand based on other features.")
                
                X = df_encoded.drop(columns=[target_col_regression])
                y = df_encoded[target_col_regression]
                
                if not pd.api.types.is_numeric_dtype(y):
                    st.error(f"Selected target column '{target_col_regression}' must be numeric for Regression.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    reg_model = LinearRegression()
                    reg_model.fit(X_train, y_train)
                    y_pred = reg_model.predict(X_test)
                    
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("R² Score (Goodness of Fit)", f"{r2:.4f}")
                    col2.metric("Mean Squared Error", f"{mse:.2f}")
                    
                    st.subheader("Actual vs Predicted")
                    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                    fig = px.scatter(results_df, x='Actual', y='Predicted', title="Actual vs Predicted Values", trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Regression Coefficients")
                    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': reg_model.coef_}).sort_values(by='Coefficient', ascending=True)
                    fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', title="Feature Impact on Target")
                    st.plotly_chart(fig_coef, use_container_width=True)

            elif menu == "6. ML: Customer Segmentation":
                st.header("🎯 Customer Segmentation (K-Means)")
                st.markdown("Group similar customer purchasing patterns to target them effectively.")
                
                num_cols = df_encoded.select_dtypes(include=np.number).columns.tolist()
                selected_features = st.multiselect("Select Features for Clustering", num_cols, default=num_cols[:2] if len(num_cols)>=2 else num_cols)
                
                if len(selected_features) >= 2:
                    k = st.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3)
                    
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    clusters = kmeans.fit_predict(df_encoded[selected_features])
                    
                    df_clustered = df.copy()
                    df_clustered['Cluster'] = clusters
                    
                    st.subheader("Cluster Visualization")
                    fig = px.scatter(df_clustered, x=selected_features[0], y=selected_features[1], color=df_clustered['Cluster'].astype(str), title=f"K-Means Clusters ({selected_features[0]} vs {selected_features[1]})")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least 2 numerical features for clustering visualization.")

            elif menu == "7. Prescriptive Analytics":
                st.header("💡 Prescriptive Analytics & Recommendations")
                st.markdown("Actionable insights based on Risk Prediction.")
                
                if target_col_classification in df.columns:
                    st.subheader("Current Inventory Risk Status")
                    risk_counts = df[target_col_classification].value_counts().reset_index()
                    risk_counts.columns = ['Risk Level', 'Count']
                    fig = px.bar(risk_counts, x='Risk Level', y='Count', color='Risk Level', title="Risk Level Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Recommended Actions")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.error("🔴 **High Risk (Overstock / Low Demand)**")
                        st.markdown("""
                        - **Action:** Apply heavy discounts.
                        - **Marketing:** Run flash sales / targeted campaigns.
                        - **Inventory:** Halt reordering immediately.
                        """)
                    with col2:
                        st.warning("🟡 **Medium Risk (Stable)**")
                        st.markdown("""
                        - **Action:** Bundle with top-selling items.
                        - **Marketing:** Standard promotional emails.
                        - **Inventory:** Monitor closely before reordering.
                        """)
                    with col3:
                        st.success("🟢 **Low Risk (High Demand / Fast Selling)**")
                        st.markdown("""
                        - **Action:** Maintain regular price, avoid discounts.
                        - **Marketing:** Highlight as 'Bestseller'.
                        - **Inventory:** Expedite reorder to prevent stockouts.
                        """)
                else:
                    st.warning("Please specify the Risk Level column in the sidebar to view specific recommendations.")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        if menu != "1. AWS Architecture":
            st.info("👋 Welcome! Please upload your dataset CSV from the sidebar to begin analysis.")
            
            # Show a demo option or dummy dashboard picture
            st.markdown("### What to expect:")
            st.markdown("- **Robust ML Models**: Random Forest, Linear Regression, K-Means.")
            st.markdown("- **Interactive Visuals**: Understand sales and behavior via Plotly graphs.")
            st.markdown("- **Actionable Insights**: Get prescriptive recommendations to optimize inventory.")

if __name__ == "__main__":
    main()
