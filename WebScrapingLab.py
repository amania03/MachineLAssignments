import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup as soup
import re

# Step 1: Define the URL and headers
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
}
url = 'https://charlottesville.craigslist.org/search/msa?purveyor=owner'  # URL for musical instruments
raw = requests.get(url, headers=header)  # Get page

# Step 2: Parse the webpage
bsObj = soup(raw.content, 'html.parser')  # Parse the HTML
listings = bsObj.find_all(class_="cl-static-search-result")  # Find all listings

# Step 3: Data extraction
brands = ['guitar', 'piano', 'drum', 'violin', 'flute', 'trumpet', 'saxophone', 'banjo', 'ukulele']

data = []  # List to store data
for listing in listings:
    title = listing.find('div', class_='title').get_text().lower()  # Extract title
    price = listing.find('div', class_='price').get_text()  # Extract price
    link = listing.find(href=True)['href']  # Extract link

    # Get brand from title
    words = title.split()
    hits = [word for word in words if word in brands]  # Check for brands
    brand = hits[0] if hits else 'missing'  # Default to 'missing' if no brand found

    # Use regex to find year in title (assumed format)
    regex_search = re.search(r'\b(20[0-9]{2}|19[0-9]{2})\b', title)  # Match 1900s or 2000s
    year = regex_search.group(0) if regex_search else np.nan  # Use found year or NaN

    # Append the data
    data.append({'title': title, 'price': price, 'year': year, 'link': link, 'brand': brand})

# Step 4: Wrangle the data
df = pd.DataFrame.from_dict(data)
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)  # Clean price
df['year'] = pd.to_numeric(df['year'], errors='coerce')  # Convert year to numeric
df['age'] = 2024 - df['year']  # Calculate age based on current year
df.to_csv('craigslist_cville_musical_instruments.csv', index=False)  # Save data
print(df.head())  # Print first few rows

# Step 5: Exploratory Data Analysis (EDA)
print(df.describe())  # Summary statistics
df['price'].hist(grid=False)
plt.title('Price Distribution of Musical Instruments')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.show()

# EDA for age
df['age'].hist(grid=False)
plt.title('Age Distribution of Musical Instruments')
plt.xlabel('Age (Years)')
plt.ylabel('Frequency')
plt.show()

# Price by brand
price_brand = df.groupby('brand')['price'].describe()
print(price_brand)  # Summary statistics for price by brand

# Visualization of average price by brand
avg_price = df.groupby('brand')['price'].mean().sort_values()
avg_price.plot(kind='barh')
plt.title('Average Price of Musical Instruments by Brand')
plt.xlabel('Average Price ($)')
plt.ylabel('Brand')
plt.show()

#I used class notes, open AI and stack overflow to help with this lab