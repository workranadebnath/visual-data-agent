import sqlite3
import pandas as pd

# 1. Define the Complex Data
employees_data = {
    'EmployeeID': [101, 102, 103, 104],
    'Name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown', 'Diana Prince'],
    'Department': ['Enterprise Sales', 'SMB Sales', 'Enterprise Sales', 'SMB Sales'],
    'BaseSalary': [120000, 85000, 115000, 90000]
}

customers_data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'CompanyName': ['TechCorp', 'HealthPlus', 'FinServe', 'EduTech', 'GlobalLogistics'],
    'Industry': ['Technology', 'Healthcare', 'Finance', 'Education', 'Logistics'],
    'Region': ['North America', 'Europe', 'North America', 'Europe', 'Asia']
}

products_data = {
    'ProductID': [201, 202, 203],
    'ProductName': ['Data Warehouse Pro', 'Analytics Dashboard', 'Predictive AI Module'],
    'Category': ['Storage', 'Visualization', 'Machine Learning'],
    'Price': [50000, 15000, 85000],
    'CostToDeliver': [10000, 2000, 15000] 
}

sales_data = {
    'SaleID': [1001, 1002, 1003, 1004, 1005, 1006],
    'SaleDate': ['2026-01-15', '2026-02-20', '2026-03-05', '2026-03-12', '2026-04-01', '2026-04-08'],
    'EmployeeID': [101, 102, 101, 104, 103, 102],
    'CustomerID': [1, 2, 3, 4, 5, 1],
    'ProductID': [201, 202, 203, 202, 203, 201],
    'Quantity': [1, 5, 2, 10, 1, 2]
}

# 2. Convert to DataFrames
df_employees = pd.DataFrame(employees_data)
df_customers = pd.DataFrame(customers_data)
df_products = pd.DataFrame(products_data)
df_sales = pd.DataFrame(sales_data)

# 3. Write to SQLite Database
conn = sqlite3.connect('enterprise_data.db')

df_employees.to_sql('employees', conn, if_exists='replace', index=False)
df_customers.to_sql('customers', conn, if_exists='replace', index=False)
df_products.to_sql('products', conn, if_exists='replace', index=False)
df_sales.to_sql('sales', conn, if_exists='replace', index=False)

conn.close()
print("Success: 'enterprise_data.db' created with 4 relational tables!")