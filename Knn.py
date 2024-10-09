# Q0: Differences between regression and classification:
# Regression is used for predicting continuous outcomes (e.g., predicting sales), while classification is used to predict categorical outcomes (e.g., predicting if an email is spam or not).
# A confusion table (confusion matrix) is a tool used in classification to summarize the performance of a model by showing the true positives, true negatives, false positives, and false negatives. It helps us understand where the model makes mistakes.
# SSE (Sum of Squared Errors) is a measure of how well a regression model fits the data. It quantifies the total difference between predicted and actual values.
# Overfitting occurs when a model is too complex and learns the noise in the training data, which reduces its performance on new data. Underfitting occurs when a model is too simple to capture the underlying patterns.
# Splitting data into training and testing sets improves model performance by ensuring that the model is evaluated on unseen data, avoiding overfitting. K is chosen by evaluating SSE or accuracy on the test set to find the optimal model.
# In classification, reporting a class label is simple and interpretable, but it doesn't convey uncertainty. Reporting a probability distribution over class labels provides more information but can be harder to interpret for non-experts.

# Load the data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('cars_env.csv')

# Rename relevant columns to simpler names
data.rename(columns={
    'baseline mpg': 'mpg',
    'baseline price': 'price',
    'baseline sales': 'sales',
    'footprint': 'footprint'
}, inplace=True)

# Check if renaming worked
print("Renamed Data Columns:", data.columns)

# Exploratory Data Analysis (EDA)
print(data[['footprint', 'mpg', 'price', 'sales']].describe())

# Histograms and Kernel Density Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(data['footprint'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Footprint Distribution')

sns.histplot(data['mpg'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('MPG Distribution')

sns.histplot(data['price'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Price Distribution')

sns.histplot(data['sales'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Sales Distribution')

plt.tight_layout()
plt.show()

# Scatterplots
sns.pairplot(data[['footprint', 'mpg', 'price', 'sales']])
plt.show()

# Normalizing footprint, mpg, and price
scaler = MinMaxScaler()
data[['footprint', 'mpg', 'price']] = scaler.fit_transform(data[['footprint', 'mpg', 'price']])

# Split the data into train (70%) and test (30%)
X = data[['footprint', 'mpg', 'price']]
y = data['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Finding the best k value for KNN Regression
errors = []
k_values = range(2, 151)

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    sse = mean_squared_error(y_test, y_pred) * len(y_test)  # SSE = MSE * number of samples
    errors.append(sse)

# Find the k with the lowest SSE
best_k = k_values[errors.index(min(errors))]
best_sse = min(errors)

print(f"The best k is {best_k} with an SSE of {best_sse}.")

# Plotting SSE for different k values
plt.plot(k_values, errors)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('SSE vs. k for KNN Regression')
plt.show()

# Additional analysis: pairwise variable combinations can be implemented similarly



#I used open AI and Stack overflow and some of the assignment solutions to help with this