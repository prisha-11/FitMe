import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000

# Generate features
product_ids = [f"PRD{str(i).zfill(4)}" for i in range(n_samples)]
categories = np.random.choice(['T-Shirt', 'Jeans', 'Jacket', 'Sweater', 'Dress', 'Sneakers'], n_samples)
prices = np.random.uniform(15.0, 150.0, n_samples).round(2)
stock_levels = np.random.randint(10, 500, n_samples)
sales_volumes = np.random.randint(5, 300, n_samples)

# Generate some missing values to test handling
prices[np.random.choice(n_samples, 50, replace=False)] = np.nan
categories[np.random.choice(n_samples, 30, replace=False)] = np.nan

# Customer Behaviour features
customer_ages = np.random.randint(18, 65, n_samples)
customer_genders = np.random.choice(['Male', 'Female', 'Other'], n_samples)

# Define Risk Level based on Sales/Stock ratio (Low risk if high sales & low stock, High risk if low sales & high stock)
risk_levels = []
for sales, stock in zip(sales_volumes, stock_levels):
    ratio = sales / stock
    if ratio > 0.8:
        risk_levels.append('Low')
    elif ratio > 0.4:
        risk_levels.append('Medium')
    else:
        risk_levels.append('High')

# Create DataFrame
df = pd.DataFrame({
    'Product_ID': product_ids,
    'Category': categories,
    'Price': prices,
    'Stock_Level': stock_levels,
    'Sales_Volume': sales_volumes,
    'Customer_Age': customer_ages,
    'Customer_Gender': customer_genders,
    'Risk_Level': risk_levels
})

# Save to CSV
df.to_csv('clothes_inventory.csv', index=False)
print("Mock dataset generated successfully as 'clothes_inventory.csv'!")
