import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('college_completion.csv')

# Q1: Exploring the dataset
# Dimensions of the data
print("Dimensions of the data:", df.shape)
print("Number of observations:", df.shape[0])
print("Variables included:", df.columns.tolist())
print(df.head())

# Cross-tabulate control and level
control_level_crosstab = pd.crosstab(df['control'], df['level'])
print(control_level_crosstab)

# Histogram, kernel density plot, boxplot, and statistical description for grad_100_value
plt.figure(figsize=(12, 8))

# Histogram
plt.subplot(2, 2, 1)
sns.histplot(df['grad_100_value'], bins=30, kde=True)
plt.title('Histogram of Graduation Rates (grad_100_value)')

# Kernel density plot
plt.subplot(2, 2, 2)
sns.kdeplot(df['grad_100_value'], fill=True)
plt.title('Kernel Density Plot of Graduation Rates (grad_100_value)')

# Boxplot
plt.subplot(2, 2, 3)
sns.boxplot(x=df['grad_100_value'])
plt.title('Boxplot of Graduation Rates (grad_100_value)')

# Statistical description
print("Statistical description of grad_100_value:")
print(df['grad_100_value'].describe())

plt.tight_layout()
plt.show()

# Grouped kernel density plot by control and by level
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='grad_100_value', hue='control', fill=True, common_norm=False, alpha=0.5)
plt.title('Kernel Density Plot of Graduation Rates by Control')
plt.show()

sns.kdeplot(data=df, x='grad_100_value', hue='level', fill=True, common_norm=False, alpha=0.5)
plt.title('Kernel Density Plot of Graduation Rates by Level')
plt.show()

# Grouped calculations of statistical descriptions of grad_100_value by level and control
grouped_desc = df.groupby(['level', 'control'])['grad_100_value'].describe()
print(grouped_desc)

# Identifying institutions with the best graduation rates
best_grads = df.loc[df['grad_100_value'] >= 90]  # Example threshold for best graduation rates
print("Institutions with graduation rates of 90% or higher:")
print(best_grads[['control', 'level', 'grad_100_value']])

# Create new variable levelXcontrol
df['levelXcontrol'] = df['level'] + ', ' + df['control']

# Grouped kernel density plot for levelXcontrol
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='grad_100_value', hue='levelXcontrol', fill=True, common_norm=False, alpha=0.5)
plt.title('Kernel Density Plot of Graduation Rates by Level and Control')
plt.show()

# Kernel density plot of aid_value
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='aid_value', fill=True)
plt.title('Kernel Density Plot of Aid Value')
plt.show()

# Grouped calculations of statistical descriptions of aid_value by level and control
aid_value_desc = df.groupby(['level', 'control'])['aid_value'].describe()
print(aid_value_desc)

# Scatterplot of grad_100_value by aid_value
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='aid_value', y='grad_100_value')
plt.title('Scatterplot of Graduation Rates by Aid Value')
plt.xlabel('Aid Value')
plt.ylabel('Graduation Rates (grad_100_value)')
plt.show()

# Grouping by level
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='aid_value', y='grad_100_value', hue='level')
plt.title('Scatterplot of Graduation Rates by Aid Value (Grouped by Level)')
plt.xlabel('Aid Value')
plt.ylabel('Graduation Rates (grad_100_value)')
plt.show()

# Grouping by control
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='aid_value', y='grad_100_value', hue='control')
plt.title('Scatterplot of Graduation Rates by Aid Value (Grouped by Control)')
plt.xlabel('Aid Value')
plt.ylabel('Graduation Rates (grad_100_value)')
plt.show()


#question 4
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Medicare Monthly Enrollment Data_Sept2024.csv')  # Update with the correct file path

# Data Cleaning
# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Drop rows with missing values if necessary (comment this line if you prefer filling missing values)
df = df.dropna()  # Alternatively, use df.fillna(method='ffill') to fill missing values

# Exploratory Data Analysis (EDA)
# Display the first few rows of the dataframe
print("\nFirst 5 rows of the dataset:\n", df.head())

# Get summary statistics
print("\nSummary statistics:\n", df.describe())

# Check the data types
print("\nData types:\n", df.dtypes)

# 1. Total Number of Beneficiaries in Alabama for 2013
total_beneficiaries = df['TOT_BENES'].sum()
print(f'\nTotal number of beneficiaries in Alabama for 2013: {total_beneficiaries}')

# 2. Percentage of Beneficiaries Aged 65 and Older
beneficiaries_65_plus = (
    df['AGE_65_TO_69_BENES'].sum() +
    df['AGE_70_TO_74_BENES'].sum() +
    df['AGE_75_TO_79_BENES'].sum() +
    df['AGE_80_TO_84_BENES'].sum() +
    df['AGE_85_TO_89_BENES'].sum() +
    df['AGE_90_TO_94_BENES'].sum() +
    df['AGE_GT_94_BENES'].sum()
)
percentage_65_plus = (beneficiaries_65_plus / total_beneficiaries) * 100
print(f'Percentage of beneficiaries aged 65 and older: {percentage_65_plus:.2f}%')

# 3. Racial Demographics
race_counts = {
    'White': df['WHITE_TOT_BENES'].sum(),
    'Black': df['BLACK_TOT_BENES'].sum(),
    'Asian/Pacific Islander': df['API_TOT_BENES'].sum(),
    'Hispanic': df['HSPNC_TOT_BENES'].sum(),
    'Native American': df['NATIND_TOT_BENES'].sum(),
    'Other': df['OTHR_TOT_BENES'].sum()
}
print('\nRacial Demographics:')
for race, count in race_counts.items():
    print(f'{race}: {count}')

# Plot the racial demographics
plt.figure(figsize=(10, 6))
plt.bar(race_counts.keys(), race_counts.values(), color='skyblue')
plt.title('Racial Demographics of Medicare Beneficiaries in Alabama (2013)')
plt.xlabel('Race')
plt.ylabel('Number of Beneficiaries')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. County with the Highest Number of Disabled Beneficiaries
# Assuming 'DSBLD_TOT_BENES' column represents the total disabled beneficiaries
df['TOTAL_DISABLED_BENES'] = df['DSBLD_TOT_BENES']
county_with_most_disabled = df.loc[df['TOTAL_DISABLED_BENES'].idxmax(), 'BENE_COUNTY_DESC']
disabled_count = df['TOTAL_DISABLED_BENES'].max()

print(f'\nCounty with the highest number of disabled beneficiaries: {county_with_most_disabled} with {disabled_count} beneficiaries')



#I used open AI and stack overflow to help me write my code