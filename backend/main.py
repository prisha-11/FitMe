from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score
import os

app = FastAPI(title="FitMe Startup API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model state
DATA_PATH = "../clothes_inventory.csv"
df_global = None
df_encoded_global = None
encoders_global = {}
rf_model = None
lr_model = None

def load_and_preprocess():
    global df_global, df_encoded_global, encoders_global, rf_model, lr_model
    if not os.path.exists(DATA_PATH):
        return False
        
    # Reset models so we don't carry over old models to new datasets
    rf_model = None
    lr_model = None
    encoders_global = {}
        
    df = pd.read_csv(DATA_PATH)
    df_clean = df.copy()
    
    # Preprocessing
    num_cols = df_clean.select_dtypes(include=np.number).columns
    cat_cols = df_clean.select_dtypes(exclude=np.number).columns
    
    for col in num_cols:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    for col in cat_cols:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
    df_encoded = df_clean.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders_global[col] = le
        
    df_global = df_clean
    df_encoded_global = df_encoded
    
    # Train Models
    # Try to find target columns heuristically if exact names don't exist
    cols_lower = {c.lower(): c for c in df_encoded.columns}
    
    risk_col = cols_lower.get('risk_level', cols_lower.get('risk', None))
    sales_col = cols_lower.get('sales_volume', cols_lower.get('sales', None))
    
    # 1. Classification (Risk_Level)
    if risk_col:
        X_clf = df_encoded.drop(columns=[risk_col, 'Product_ID'], errors='ignore')
        y_clf = df_encoded[risk_col]
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(X_clf, y_clf)
        
    # 2. Regression (Sales_Volume)
    if sales_col:
        X_reg = df_encoded.drop(columns=[sales_col, 'Product_ID', risk_col] if risk_col else [sales_col, 'Product_ID'], errors='ignore')
        y_reg = df_encoded[sales_col]
        lr_model = LinearRegression()
        lr_model.fit(X_reg, y_reg)
        
    return True

@app.on_event("startup")
def startup_event():
    load_and_preprocess()

@app.get("/api/dashboard-stats")
def get_dashboard_stats():
    if df_global is None:
        raise HTTPException(status_code=404, detail="Data not loaded")
    
    total_products = len(df_global)
    avg_price = float(df_global['Price'].mean())
    total_sales = int(df_global['Sales_Volume'].sum())
    high_risk_count = int(len(df_global[df_global['Risk_Level'] == 'High']))
    
    return {
        "total_products": total_products,
        "avg_price": round(avg_price, 2),
        "total_sales": total_sales,
        "high_risk_count": high_risk_count
    }

@app.get("/api/charts/category-sales")
def get_category_sales():
    if df_global is None:
        return []
    cat_sales = df_global.groupby("Category")["Sales_Volume"].sum().reset_index()
    return {
        "labels": cat_sales["Category"].tolist(),
        "series": cat_sales["Sales_Volume"].tolist()
    }

@app.get("/api/charts/risk-distribution")
def get_risk_distribution():
    if df_global is None:
        return []
    risk_counts = df_global["Risk_Level"].value_counts().reset_index()
    return {
        "labels": risk_counts["Risk_Level"].tolist(),
        "series": risk_counts["count"].tolist()
    }

@app.get("/api/charts/price-vs-sales")
def get_price_vs_sales():
    if df_global is None:
        return []
    # Sample 100 points for frontend performance
    sample = df_global.sample(n=min(100, len(df_global)))
    data = []
    for _, row in sample.iterrows():
        data.append({"x": row["Price"], "y": row["Sales_Volume"], "category": row["Category"]})
    return {"data": data}

@app.get("/api/models/evaluation")
def get_model_evaluation():
    if df_global is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
        
    cols_lower = {c.lower(): c for c in df_encoded_global.columns}
    risk_col = cols_lower.get('risk_level', cols_lower.get('risk', None))
    sales_col = cols_lower.get('sales_volume', cols_lower.get('sales', None))
    
    acc = 0.0
    r2 = 0.0
    
    # Re-evaluate on train set for simplicity in this demo endpoint
    if rf_model and risk_col:
        X_clf = df_encoded_global.drop(columns=[risk_col, 'Product_ID'], errors='ignore')
        y_clf = df_encoded_global[risk_col]
        acc = accuracy_score(y_clf, rf_model.predict(X_clf))
        
    if lr_model and sales_col:
        X_reg = df_encoded_global.drop(columns=[sales_col, 'Product_ID', risk_col] if risk_col else [sales_col, 'Product_ID'], errors='ignore')
        y_reg = df_encoded_global[sales_col]
        r2 = r2_score(y_reg, lr_model.predict(X_reg))
    
    return {
        "classification_accuracy": round(acc * 100, 1) if acc > 0 else "--",
        "regression_r2": round(r2, 3) if r2 != 0.0 else "--",
        "recommendation": "Models are ready." if rf_model and lr_model else "Some models could not be trained due to missing target columns (Risk/Sales)."
    }

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global DATA_PATH
    content = await file.read()
    
    # Save the new dataset
    DATA_PATH = f"../{file.filename}"
    with open(DATA_PATH, "wb") as f:
        f.write(content)
        
    # Re-run preprocessing and training
    success = load_and_preprocess()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.")
        
    return {"message": "File uploaded and models retrained successfully!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
