import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import seaborn as sns

# Q1: Tokenizing and Text Cleaning
print("---- Q1: Tokenizing and Text Cleaning ----")
df = pd.read_parquet('https://data434.s3.us-east-2.amazonaws.com/Phishing_Email.parquet')

# Tokenize Email Text
df['tokens'] = df['Email Text'].str.split()
print(df[['Email Text', 'tokens']].head())  # Display original text and tokens

# Summary of cleaning tokens
summary = """
To clean the tokens for prediction, the following steps could be taken:
1. Convert text to lowercase to normalize case.
2. Remove punctuation and special characters.
3. Remove stop words (e.g., "the", "and").
4. Stem or lemmatize words to reduce inflectional forms (e.g., 'running' -> 'run').
5. Use token frequency as features for predictive modeling.
A predictive algorithm like logistic regression or a tree-based model can then use the cleaned tokens to predict phishing emails.
"""
print(summary)

# Q2: Token Frequency and Histogram
print("\n---- Q2: Token Frequency and Histogram ----")
with open('all_tokens.pickle', 'rb') as file:
    all_tokens = pickle.load(file)

# Count token occurrences
token_count = Counter(all_tokens)
token_freq = token_count.most_common()

# Plot histogram
frequencies = [count for _, count in token_freq]
plt.hist(frequencies, bins=50, edgecolor='black')
plt.title('Histogram of Token Frequencies')
plt.xlabel('Token Occurrence Frequency')
plt.ylabel('Number of Tokens')
plt.yscale('log')  # Optional: Use log scale for clarity
plt.show()

# Observation
observation = """
Most tokens occur infrequently, while a small number of tokens appear very frequently. 
This pattern is typical of natural language data, where common words (e.g., "the", "and") dominate.
"""
print(observation)

# Q3: Regression with One-Hot Encoded Tokens
print("\n---- Q3: Regression with One-Hot Encoded Tokens ----")
df_clean = pd.read_parquet('Phishing_clean.parquet')

# Train-test split
X = df_clean.drop(columns=['Email Type'])
y = df_clean['Email Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate performance
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
print(f"Training R^2: {train_r2}")
print(f"Test R^2: {test_r2}")

# Identify most influential words
coefficients = pd.Series(model.coef_, index=X.columns)
top_positive = coefficients.nlargest(10)
top_negative = coefficients.nsmallest(10)
print("Top Positive Influences:", top_positive)
print("Top Negative Influences:", top_negative)

# Q4: PCA and Scatter Plot
print("\n---- Q4: PCA and Scatter Plot ----")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a scatter plot
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Email Type'] = y
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Email Type', palette='coolwarm')
plt.title('PCA Scatter Plot of Emails')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Observation
observation_pca = """
The phishing emails and safe emails may form distinct clusters along certain axes, 
indicating that PCA captures variance that separates them. However, some overlap is likely.
"""
print(observation_pca)

# Q5: Regression with PCA Components
print("\n---- Q5: Regression with PCA Components ----")
pca = PCA(n_components=2610)
X_pca_2610 = pca.fit_transform(X)

# Train-test split
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_2610, y, test_size=0.2, random_state=42)

# Linear regression
model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train_pca)

# Evaluate performance
train_r2_pca = model_pca.score(X_train_pca, y_train_pca)
test_r2_pca = model_pca.score(X_test_pca, y_test_pca)
print(f"Training R^2 with PCA: {train_r2_pca}")
print(f"Test R^2 with PCA: {test_r2_pca}")

# Q6: PCA Advantage
print("\n---- Q6: PCA Advantage ----")
explanation = """
Using PCA reduces the dimensionality of the data, mitigating overfitting and computational overhead. 
By projecting the data into a smaller number of components that capture the most variance, 
PCA emphasizes meaningful patterns and relationships, improving generalization.
"""
print(explanation)


#I used assignment solutions and OpenAI to help complete assignment 