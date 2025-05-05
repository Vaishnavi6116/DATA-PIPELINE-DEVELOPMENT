# DATA-PIPELINE-DEVELOPMENT
# Examine Data Structure
display(df.head())
print(df.shape)
display(df.info())

# Identify Missing Values
print(df.isnull().sum())

# Examine Key Variable Distributions
key_variables = ['Sales', 'Profit', 'Quantity', 'Order Date']
display(df[key_variables].describe())

import matplotlib.pyplot as plt
plt.figure(figsize=(16, 6))
plt.subplot(1, 4, 1)
plt.hist(df['Sales'], bins=20, color='skyblue', edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')

plt.subplot(1, 4, 2)
plt.hist(df['Profit'], bins=20, color='lightcoral', edgecolor='black')
plt.title('Profit Distribution')
plt.xlabel('Profit')
plt.ylabel('Frequency')

plt.subplot(1, 4, 3)
plt.hist(df['Quantity'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Quantity Distribution')
plt.xlabel('Quantity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Initial Data Type Considerations
# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
print(df['Order Date'].dtype)
import pandas as pd
import matplotlib.pyplot as plt

# 1. Check for inconsistencies in categorical columns
categorical_cols = ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category']
for col in categorical_cols:
    print(f"Unique values for {col}: {df[col].unique()}")

# 2. Investigate outliers in 'Sales', 'Profit', and 'Quantity'
numerical_cols = ['Sales', 'Profit', 'Quantity']
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    plt.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Handle outliers (capping)
def cap_outliers(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in numerical_cols:
  df[col] = cap_outliers(df[col])

plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    plt.boxplot(df[col])
    plt.title(f'Boxplot of {col} (After Capping)')
plt.tight_layout()
plt.show()

# 3. Review 'Order Date' and 'Ship Date'
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
print(f"Number of rows where Ship Date is before Order Date: {(df['Ship Date'] < df['Order Date']).sum()}")

# 4. Verify data types and correct if needed
display(df.info())
import pandas as pd
import matplotlib.pyplot as plt

# 1. Key Statistics
print("Key Statistics:")
display(df[['Sales', 'Profit', 'Quantity']].describe())

# Segment by Category and calculate key statistics
print("\nKey Statistics by Category:")
display(df.groupby('Category')[['Sales', 'Profit', 'Quantity']].agg(['mean', 'median', 'std']))

# 2. Sales Trends
# Convert Order Date to datetime if not already done
if not pd.api.types.is_datetime64_any_dtype(df['Order Date']):
    df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Order Month'] = df['Order Date'].dt.to_period('M')

monthly_sales = df.groupby('Order Month')['Sales'].sum()
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index.astype(str), monthly_sales.values)
plt.xlabel('Order Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Relationships between variables
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['Sales'], df['Profit'], alpha=0.5)
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.title('Sales vs. Profit')

plt.subplot(1, 2, 2)
plt.scatter(df['Sales'], df['Quantity'], alpha=0.5)
plt.xlabel('Sales')
plt.ylabel('Quantity')
plt.title('Sales vs. Quantity')
plt.tight_layout()
plt.show()

print(df[['Sales', 'Profit', 'Quantity']].corr())


# 4. Top Performers
# Top Products by Sales
top_products_sales = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Sales:")
display(top_products_sales)

# Top Customers by Spending
top_customers = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Customers by Spending:")
display(top_customers)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Sales Trend Visualization
plt.figure(figsize=(14, 6))
monthly_sales = df.groupby('Order Month')['Sales'].sum()
plt.plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o', linestyle='-')
plt.xlabel('Order Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# 2. Sales by Category/Sub-Category
plt.figure(figsize=(14, 6))
category_sales = df.groupby('Category')['Sales'].sum()
category_sales.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.title('Total Sales by Category')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 6))
subcategory_sales = df.groupby('Sub-Category')['Sales'].sum()
subcategory_sales.plot(kind='bar')
plt.xlabel('Sub-Category')
plt.ylabel('Total Sales')
plt.title('Total Sales by Sub-Category')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 3. Profit Margins by Region
plt.figure(figsize=(12, 6))
region_profit = df.groupby('Region')['Profit'].mean()
region_profit.plot(kind='bar', color='orange')
plt.xlabel('Region')
plt.ylabel('Average Profit')
plt.title('Average Profit Margin by Region')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 4. Customer Segmentation
plt.figure(figsize=(10, 6))
plt.scatter(df['Sales'], df['Quantity'], c=df['Profit'], cmap='viridis', alpha=0.6)
plt.xlabel('Total Spending (Sales)')
plt.ylabel('Number of Orders (Quantity)')
plt.title('Customer Segmentation')
plt.colorbar(label='Profit')
plt.tight_layout()
plt.show()


# 5. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[['Sales', 'Quantity', 'Discount', 'Profit']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np

# 1. Calculate Total Revenue
df['TotalRevenue'] = df['Sales'] * df['Quantity']

# 2. Calculate Profit Margin
df['ProfitMargin'] = np.where(df['TotalRevenue'] != 0, df['Profit'] / df['TotalRevenue'], 0)

# 3. Extract Order Month
df['OrderMonth'] = pd.to_datetime(df['Order Date']).dt.month

# 4. Calculate Shipping Time
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['ShippingTime'] = (df['Ship Date'] - df['Order Date']).dt.days
df['ShippingTime'] = df['ShippingTime'].clip(lower=0) # Ensure non-negative shipping time

display(df.head())
