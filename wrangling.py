import pandas as pd
import numpy as np

# Load the CSV files
airbnb_df = pd.read_csv('airbnb_hw.csv')
sharks_df = pd.read_csv('sharks.csv', low_memory=False)  # Set low_memory=False to handle mixed types
pretrial_df = pd.read_csv('pretrial_data.csv', low_memory=False)  # Set low_memory=False


# Q2: Clean the Price variable in airbnb_df
def clean_price(price_str):
    """Cleans the price string by removing commas and converting to float."""
    if isinstance(price_str, str):
        try:
            price_str = price_str.replace(',', '')  # Remove commas
            return float(price_str)
        except ValueError:
            return np.nan
    return np.nan


# Apply the cleaning function
airbnb_df['Price'] = airbnb_df['Price'].apply(clean_price)

# Calculate and print the number of missing values in the Price column
missing_price_values = airbnb_df['Price'].isna().sum()
print(f"Missing values in Price variable: {missing_price_values}")


# Clean the Type variable in sharks_df
def clean_type(type_str):
    """Standardizes the type string by converting to lowercase and removing extra spaces."""
    if isinstance(type_str, str):
        return type_str.strip().lower()  # Convert to lowercase and remove extra spaces
    return np.nan


sharks_df['Type'] = sharks_df['Type'].apply(clean_type)

# Validate and print unique values in the cleaned Type column to check the cleaning
print(f"Unique values in the 'Type' column after cleaning: {sharks_df['Type'].unique()}")

# Check columns in the DataFrame
print(f'Columns in pretrial_df: {pretrial_df.columns}')


# Define a function to classify pretrial release based on PretrialReleaseType1
# Update the classify_pretrial_release function
def classify_pretrial_release(release_type):
    if pd.isna(release_type):
        return 'Not Released'
    elif release_type == 1:
        return 'Released on Personal Recognizance/Unsecured Bond without Supervision'
    elif release_type == 2:
        return 'Released on Summons without Supervision'
    elif release_type == 3:
        return 'Released on Secured Bond without Supervision'
    elif release_type == 4:
        return 'Released on Personal Recognizance/Unsecured Bond with Supervision'
    elif release_type == 5:
        return 'Released on Summons with Supervision'
    elif release_type == 6:
        return 'Released on Secured Bond with Supervision'
    elif release_type == 0:
        return 'Not Released Pretrial'
    elif release_type == 9:
        return 'Cannot Determine Bond Type'
    else:
        return 'Unknown'


# Apply the function to create a new column for pretrial release indicator
pretrial_df['Pretrial_Release_Indicator'] = pretrial_df['PretrialReleaseType1'].apply(classify_pretrial_release)

# Check for missing values in the 'ImposedSentenceAllChargeInContactEvent' variable
print(
    f'Missing values in "ImposedSentenceAllChargeInContactEvent" after cleaning: {pretrial_df["ImposedSentenceAllChargeInContactEvent"].isna().sum()}')

# Display the results
print(pretrial_df[['PretrialReleaseType1', 'Pretrial_Release_Indicator']].head())

# Save the updated DataFrame to a new CSV if needed
pretrial_df.to_csv('updated_pretrial_data.csv', index=False)

# Clean the 'ImposedSentenceAllChargeInContactEvent' variable
sentence_column_name = 'ImposedSentenceAllChargeInContactEvent'
if sentence_column_name in pretrial_df.columns:
    # Replace empty strings with NaN
    pretrial_df[sentence_column_name] = pretrial_df[sentence_column_name].replace('', np.nan)

    # Use 'SentenceTypeAllChargesAtConvictionInContactEvent' to infer missing values
    conviction_column_name = 'SentenceTypeAllChargesAtConvictionInContactEvent'
    if conviction_column_name in pretrial_df.columns:
        pretrial_df.loc[pretrial_df[conviction_column_name] == 'no sentence', sentence_column_name] = np.nan
else:
    print(f"The column '{sentence_column_name}' does not exist in the DataFrame.")

# Validate and print the number of missing values in the cleaned columns
if sentence_column_name in pretrial_df.columns:
    missing_sentence_values = pretrial_df[sentence_column_name].isna().sum()
    print(f"Missing values in '{sentence_column_name}' after cleaning: {missing_sentence_values}")

# Q3: The most recent U.S. Census gathered data on race through a series of self-reported surveys
# distributed to households across the country. Respondents were given the option to select
# from a predefined list of racial categories, including options like Black, White, Asian, Latino,
# and Indigenous. Additionally, the Census allowed individuals to specify their own race in
# an open-ended response. This approach aimed to capture the diversity of the U.S. population,
# reflecting both established categories and unique self-identifications.

# Data on race is collected for several important reasons. Firstly, it helps inform government
# policies and funding decisions, as demographic data is crucial for allocating resources
# effectively. Moreover, these data play a significant role in political representation, helping
# ensure that various racial groups are adequately represented in legislative bodies. The quality
# of this data is paramount; inaccuracies or biases in reporting can lead to misinformed policies
# that disproportionately affect certain populations.

# In evaluating the Census methodology, there were commendable aspects, such as the inclusion
# of an open-ended option for race, which allows for greater self-identification. However,
# critics might argue that the limited number of categories could alienate individuals who do not
# see themselves reflected in the options provided. Future large-scale surveys should consider
# expanding these categories or using more inclusive language to better capture the diversity of
# the population. Additionally, best practices from the Census, such as outreach to underrepresented
# communities, could be adopted more widely to improve data richness.

# Regarding the collection of data on sex and gender, the Census employed a similar strategy.
# Respondents were asked to select their sex from binary options, with some versions also including
# a question on gender identity. While this approach reflects traditional categories, it falls
# short by not adequately representing non-binary or gender non-conforming individuals.
# A constructive criticism would suggest the inclusion of more options to capture the full spectrum
# of gender identities, allowing for better representation in the data.

# When cleaning data related to protected characteristics like sex, gender, and race, concerns arise
# about the potential for bias. Missing values can lead to inaccurate representations, and
# improper handling of these missing values could result in the erasure of certain identities
# or experiences. Good practices would include carefully considering how to impute values and ensuring
# transparency about any assumptions made during the process, while bad practices could involve
# over-simplification or the perpetuation of existing biases.

# The invention of an algorithm to impute values for protected characteristics poses significant
# ethical concerns. Issues such as accuracy, bias, and representation must be critically examined.
# Algorithms might inadvertently reinforce stereotypes or fail to account for the complexities
# of identity. There is a risk of oversimplification, leading to the imputation of values that
# do not truly reflect the individual’s identity. It is crucial to approach such technologies
# with caution and involve diverse stakeholder perspectives to ensure ethical practices are upheld.

#I used stack overflow and the help of the professor hints to complete this assignment