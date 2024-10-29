
# Q0

# What makes a model "linear"? "Linear" in what?
# A linear model is defined by the relationship between the independent variables and the dependent variable being linear. This means that the change in the dependent variable is a linear combination of the independent variables.

# How do you interpret the coefficient for a dummy/one-hot-encoded variable?
# The coefficient for a dummy variable represents the difference in the dependent variable for the category coded as 1 compared to the reference category (coded as 0). If the model has an intercept, the coefficient is interpreted relative to the intercept.

# Can linear regression be used for classification? Explain why, or why not.
# Linear regression can be used for classification tasks but is not ideal due to its tendency to predict values outside the range of the classes (e.g., negative probabilities). Logistic regression is typically preferred for binary classification as it models probabilities directly.

# What are signs that your linear model is over-fitting?
# Signs of overfitting include a high R² value on the training data but a significantly lower R² on the test data, as well as a model that fits the training data very closely, including noise rather than the underlying pattern.

# Clearly explain multi-colinearity using the two-stage least squares technique.
# Multicollinearity occurs when independent variables in a regression model are highly correlated, which can inflate variance and make coefficient estimates unstable. Two-stage least squares (2SLS) addresses this by using instrumental variables to provide consistent estimates, where the first stage predicts the problematic variable using instruments, and the second stage uses these predictions in the main regression.

# What are two ways to incorporate nonlinear relationships between your target/response/dependent/outcome variable
# and your features/control/response/independent variables?
# 1. Polynomial regression: Adding polynomial terms (e.g., x², x³) to the model to capture curvature.
# 2. Transformation of variables: Applying functions like log, square root, or exponential to the independent variables.

# What is the interpretation of the intercept? A slope coefficient for a variable? The coefficient for a dummy/one-hot-encoded variable?
# The intercept represents the expected value of the dependent variable when all independent variables are zero.
# The slope coefficient indicates the change in the dependent variable for a one-unit increase in the independent variable.
# The coefficient for a dummy variable indicates the difference in the dependent variable between the category represented by the dummy variable and the reference category.






#Q1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('Q1_clean.csv')  # Adjust the file path as necessary

# Rename the column to remove any leading/trailing spaces
df.rename(columns=lambda x: x.strip(), inplace=True)

# 1. Compute the average prices and scores by Neighbourhood
average_prices = df.groupby('Neighbourhood')['Price'].mean()
average_scores = df.groupby('Neighbourhood')['Review Scores Rating'].mean()

# Identify the most expensive neighbourhood
most_expensive_neighbourhood = average_prices.idxmax()
most_expensive_price = average_prices.max()

print("Average Prices by Neighbourhood:")
print(average_prices)
print(f"\nMost expensive neighbourhood: {most_expensive_neighbourhood} with an average price of {most_expensive_price}")

# 2. Create a kernel density plot of price and log price, grouping by Neighbourhood
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='Price', fill=True, common_norm=False, alpha=0.5, legend=True)
sns.kdeplot(data=df, x=np.log1p(df['Price']), fill=True, common_norm=False, alpha=0.5, legend=True)
plt.title('Kernel Density Plot of Price and Log Price')
plt.xlabel('Price')
plt.ylabel('Density')
plt.legend(['Price', 'Log Price'])
plt.show()

# 3. Regress price on Neighbourhood without intercept
y = df['Price']
X = pd.get_dummies(df['Neighbourhood'], dtype='int')

reg = linear_model.LinearRegression(fit_intercept=False).fit(X, y)  # Run regression

# Get regression coefficients
results_no_intercept = pd.DataFrame({'variable': reg.feature_names_in_, 'coefficient': reg.coef_})
print("\nRegression Coefficients without Intercept:")
print(results_no_intercept)

# 4. Regress price on Neighbourhood with intercept
reg_with_intercept = linear_model.LinearRegression(fit_intercept=True).fit(X, y)

results_with_intercept = pd.DataFrame({'variable': reg_with_intercept.feature_names_in_, 'coefficient': reg_with_intercept.coef_})
print("\nRegression Coefficients with Intercept:")
print(results_with_intercept)

# 5. Split the sample 80/20 into a training and a test set for Review Scores Rating and Neighbourhood
X = df.loc[:, ['Review Scores Rating', 'Neighbourhood']]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=100)  # For reproducibility

# Create dummy variables for the 'Neighbourhood' feature
Z_train = pd.concat([X_train['Review Scores Rating'],
                     pd.get_dummies(X_train['Neighbourhood'], dtype='int')], axis=1)
Z_test = pd.concat([X_test['Review Scores Rating'],
                    pd.get_dummies(X_test['Neighbourhood'], dtype='int')], axis=1)

# Run regression of Price on Review Scores Rating and Neighbourhood
reg_model = linear_model.LinearRegression(fit_intercept=False).fit(Z_train, y_train)

# Predict on the test set
y_hat = reg_model.predict(Z_test)

# Calculate R^2 and RMSE
r_squared = reg_model.score(Z_test, y_test)
rmse = np.sqrt(np.mean((y_test - y_hat) ** 2))

# Display results
print('\nRegression of Price on Review Scores Rating and Neighbourhood:')
print('R^2: ', r_squared)
print('RMSE: ', rmse)

# Prepare results DataFrame with coefficients
results = pd.DataFrame({'variable': reg_model.feature_names_in_, 'coefficient': reg_model.coef_})

# Show the regression coefficients
print("\nRegression Coefficients for Review Scores Rating and Neighbourhood:")
print(results)

# Determine the most expensive kind of property by finding the maximum coefficient
most_expensive_property = results.loc[results['coefficient'].idxmax()]
print('\nMost expensive property kind: ', most_expensive_property)


#Q2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Load the data
df = pd.read_csv('cars_hw.csv')

# Clean the data
# Rename columns to remove any whitespace
df.columns = df.columns.str.strip()

# Check for missing values and remove them if necessary
df.dropna(inplace=True)

# Handle outliers: assuming 'Price' should be within a certain range
q_low = df["Price"].quantile(0.01)
q_hi  = df["Price"].quantile(0.99)
df = df[(df["Price"] < q_hi) & (df["Price"] > q_low)]

# Log transformation for skewed distributions if necessary
df['Mileage_Run'] = np.log1p(df['Mileage_Run'])

# Summarize Price variable
print("Summary of Price Variable:")
print(df['Price'].describe())

# Kernel density plot of Price
sns.kdeplot(df['Price'], fill=True)
plt.title('Kernel Density Plot of Car Prices')
plt.xlabel('Price')
plt.ylabel('Density')
plt.show()

# Summarize prices by brand (Make)
price_summary = df.groupby('Make')['Price'].describe()
print("Average Prices by Brand (Make):")
print(price_summary)

# Grouped kernel density plot by Make
plt.figure(figsize=(10, 6))
for make, group in df.groupby('Make'):
    sns.kdeplot(group['Price'], label=make)
plt.title('Grouped Kernel Density Plot of Car Prices by Make')
plt.xlabel('Price')
plt.ylabel('Density')
plt.legend()
plt.show()

# Most expensive car brands
most_expensive_brands = price_summary['mean'].nlargest(5)
print("Most Expensive Car Brands:")
print(most_expensive_brands)

# Split the data into training and testing sets (80% train, 20% test)
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Regress price on numeric variables
numeric_vars = X_train.select_dtypes(include=[np.number]).columns.tolist()
model1 = LinearRegression()
model1.fit(X_train[numeric_vars], y_train)

# Predictions and metrics for Model 1
y_train_pred1 = model1.predict(X_train[numeric_vars])
y_test_pred1 = model1.predict(X_test[numeric_vars])
rmse_train1 = np.sqrt(mean_squared_error(y_train, y_train_pred1))
rmse_test1 = np.sqrt(mean_squared_error(y_test, y_test_pred1))
r2_train1 = r2_score(y_train, y_train_pred1)
r2_test1 = r2_score(y_test, y_test_pred1)

print(f"Model 1 - Numeric Variables: R² Train: {r2_train1}, RMSE Train: {rmse_train1}, R² Test: {r2_test1}, RMSE Test: {rmse_test1}")

# Model 2: One-hot encode categorical variables and regress
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Align the train and test set columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

model2 = LinearRegression()
model2.fit(X_train_encoded, y_train)

# Predictions and metrics for Model 2
y_train_pred2 = model2.predict(X_train_encoded)
y_test_pred2 = model2.predict(X_test_encoded)
rmse_train2 = np.sqrt(mean_squared_error(y_train, y_train_pred2))
rmse_test2 = np.sqrt(mean_squared_error(y_test, y_test_pred2))
r2_train2 = r2_score(y_train, y_train_pred2)
r2_test2 = r2_score(y_test, y_test_pred2)

print(f"Model 2 - One-hot Encoded Variables: R² Train: {r2_train2}, RMSE Train: {rmse_train2}, R² Test: {r2_test2}, RMSE Test: {rmse_test2}")

# Model 3: Combine all regressors
X_combined_train = pd.concat([X_train[numeric_vars], X_train_encoded], axis=1)
X_combined_test = pd.concat([X_test[numeric_vars], X_test_encoded], axis=1)

model3 = LinearRegression()
model3.fit(X_combined_train, y_train)

# Predictions and metrics for Model 3
y_train_pred3 = model3.predict(X_combined_train)
y_test_pred3 = model3.predict(X_combined_test)
rmse_train3 = np.sqrt(mean_squared_error(y_train, y_train_pred3))
rmse_test3 = np.sqrt(mean_squared_error(y_test, y_test_pred3))
r2_train3 = r2_score(y_train, y_train_pred3)
r2_test3 = r2_score(y_test, y_test_pred3)

print(f"Model 3 - Combined Variables: R² Train: {r2_train3}, RMSE Train: {rmse_train3}, R² Test: {r2_test3}, RMSE Test: {rmse_test3}")

# Polynomial features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_combined_train)
X_poly_test = poly.transform(X_combined_test)

model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)

# Predictions and metrics for polynomial model
y_poly_pred = model_poly.predict(X_poly_test)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_poly_pred))
r2_poly = r2_score(y_test, y_poly_pred)

print(f"Polynomial Model: R² Test: {r2_poly}, RMSE Test: {rmse_poly}")

# Compare the best model and polynomial model RMSE
print(f"Best Model RMSE Test: {rmse_test3}, Polynomial Model RMSE Test: {rmse_poly}")

# Plot predicted vs true values for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Predicted vs True Values for Best Model')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()

# Residuals
residuals = y_test - y_test_pred3
sns.kdeplot(residuals, fill=True)
plt.title('Kernel Density Plot of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.show()

# Evaluate model strengths and weaknesses
print("Model Evaluation:")
print("Strengths: Good prediction on training data, captures trends in price.")
print("Weaknesses: May overfit, especially with polynomial features; outliers can affect predictions.")


#I did this with the help of CHat, Assignment help file, and stack overflow 