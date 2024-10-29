import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Load the Data
df = pd.read_csv('C:/Users/amani/MachineL/airbnb_hw.csv')  # Update this with your correct file path

# Step 1: Data Cleaning
print("Columns in DataFrame:")
print(df.columns)

print("\nInitial DataFrame Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with critical missing values
df.dropna(subset=['Price', 'Review Scores Rating'], inplace=True)

# Fill missing values for numeric columns with mean
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].mean())

# Clean the Price column by removing commas and converting to float
df['Price'] = df['Price'].str.replace(',', '').astype(float)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Handle other categorical variables
for col in ['Property Type', 'Room Type', 'Neighbourhood']:
    if col in df.columns:
        df[col] = df[col].str.strip()

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing Values Heatmap')
plt.show()

# Step 2: EDA
# Visualizing the Price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=30)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Boxplot for Price vs. Property Type
plt.figure(figsize=(12, 6))
sns.boxplot(x='Property Type', y='Price', data=df)
plt.title('Price by Property Type')
plt.xticks(rotation=45)
plt.show()

# Step 3: Transforming Categorical Variables
df = pd.get_dummies(df, drop_first=True)

# Step 4: Train-Test Split
X = df.drop(columns=['Price'])  # Features
y = df['Price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Building Simple Linear Model
simple_model = LinearRegression()
simple_model.fit(X_train, y_train)

# Predictions and RMSE for simple model
y_train_pred = simple_model.predict(X_train)
y_test_pred = simple_model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f'Simple Model Train RMSE: {rmse_train}')
print(f'Simple Model Test RMSE: {rmse_test}')

# Step 6: Building a More Complex Model
poly = PolynomialFeatures(degree=2)  # Adjust degree as needed
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

complex_model = LinearRegression()
complex_model.fit(X_train_poly, y_train)

# Predictions and RMSE for complex model
y_train_poly_pred = complex_model.predict(X_train_poly)
y_test_poly_pred = complex_model.predict(X_test_poly)

rmse_train_poly = np.sqrt(mean_squared_error(y_train, y_train_poly_pred))
rmse_test_poly = np.sqrt(mean_squared_error(y_test, y_test_poly_pred))

print(f'Complex Model Train RMSE: {rmse_train_poly}')
print(f'Complex Model Test RMSE: {rmse_test_poly}')

# Step 7: Lasso Regularization
lasso_model = Lasso(alpha=0.1)  # Adjust alpha as needed
lasso_model.fit(X_train, y_train)

# Predictions and RMSE for Lasso model
lasso_y_train_pred = lasso_model.predict(X_train)
lasso_y_test_pred = lasso_model.predict(X_test)

lasso_rmse_train = np.sqrt(mean_squared_error(y_train, lasso_y_train_pred))
lasso_rmse_test = np.sqrt(mean_squared_error(y_test, lasso_y_test_pred))

print(f'Lasso Model Train RMSE: {lasso_rmse_train}')
print(f'Lasso Model Test RMSE: {lasso_rmse_test}')

# Selected features by Lasso
selected_features = X.columns[lasso_model.coef_ != 0]
print("Selected features by Lasso:", selected_features)

# Step 8: Summary of Results
print("\nSummary of RMSE Results:")
print(f"Simple Model RMSE: Train - {rmse_train}, Test - {rmse_test}")
print(f"Complex Model RMSE: Train - {rmse_train_poly}, Test - {rmse_test_poly}")
print(f"Lasso Model RMSE: Train - {lasso_rmse_train}, Test - {lasso_rmse_test}")

# Step 9: Overfitting and Underfitting Observations
if rmse_train_poly < rmse_train and rmse_test_poly > rmse_test:
    print("The complex model shows signs of overfitting.")
elif rmse_train_poly > rmse_train and rmse_test_poly < rmse_test:
    print("The complex model shows signs of underfitting.")
else:
    print("The models seem to fit well without evident overfitting or underfitting.")

#IusedStackOverflowandChatGPT to help sort through this code 