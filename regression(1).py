import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np


# Load Excel file
df = pd.read_excel("resources/crime_data_with_issues.xlsx")
print(df.head())

import os

file_path = "resources/crime_data_with_issues.xlsx"

if os.path.exists(file_path):
    print("File found!")
else:
    print("File NOT found. Check the path.")

#count duplicate
duplicate_count = df.duplicated().sum()

#display duplicate rows
duplicate_rows = df[df.duplicated()]

#remove duplicates
df_no_duplicates = df.drop_duplicates()

df = df_no_duplicates

original_missing = df.isna().copy()  

# Check for missing values in each column
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Encode categorical values to numeric (Barangay)
encoder = LabelEncoder()
df['Barangay'] = encoder.fit_transform(df['Barangay'])
df['Barangay'] = df['Barangay'].replace(3, np.nan)

df['Suspect Gender'] = encoder.fit_transform(df['Suspect Gender'])
df['Province'] = encoder.fit_transform(df['Province'])
df['Municipality'] = encoder.fit_transform(df['Municipality'])
df['Crime ID'] = encoder.fit_transform(df['Crime ID'])
df['Date'] = encoder.fit_transform(df['Date'])

df['Day of the Week'] = encoder.fit_transform(df['Day of the Week'])
df['Day of the Week'] = df['Day of the Week'].replace(7, np.nan)

df['Arrest Made'] = encoder.fit_transform(df['Arrest Made'])
df['Arrest Made'] = df['Arrest Made'].replace(2, np.nan)

df['Crime Type'] = encoder.fit_transform(df['Crime Type'])
df['Street'] = encoder.fit_transform(df['Street'])
df['Victim Gender'] = encoder.fit_transform(df['Victim Gender'])
df['Outcome'] = encoder.fit_transform(df['Outcome'])

df['Weather'] = encoder.fit_transform(df['Weather'])
df['Weather'] = df['Weather'].replace(5, np.nan)


# Fill missing values with the column mean
df.loc[:, 'Victim Age'] = df['Victim Age'].fillna(round(df['Victim Age'].mean()))
df.loc[:, 'Day of the Week'] = df['Day of the Week'].fillna(round(df['Day of the Week'].mean()))
df.loc[:, 'Barangay'] = df['Barangay'].fillna(round(df['Barangay'].mean()))
df.loc[:, 'Arrest Made'] = df['Arrest Made'].fillna(round(df['Arrest Made'].mean()))
df.loc[:, 'Suspect Age'] = df['Suspect Age'].fillna(round(df['Suspect Age'].mean()))

# Check for missing values after filling
missing_values_after = df.isnull().sum()
print("Missing values per column (After filling):\n", missing_values_after)

# Select features for prediction
features = ['Crime ID', 'Date', 'Day of the Week', 'Province', 'Municipality', 
            'Barangay', 'Street', 'Victim Age', 'Victim Gender', 'Arrest Made',
            'Suspect Age', 'Suspect Gender', 'Outcome']
df_features = df[features].copy()
target = df['Weather']

# Separate data into known and missing
df_known = df.dropna(subset=['Weather']) 
df_missing = df[df['Weather'].isnull()]

# Define X (features) and y (target)
X_train = df_known[features]
y_train = df_known['Weather']
X_missing = df_missing[features]

# Train a Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing values
predicted_values = model.predict(X_missing)

# Round to ensure valid categorical values
df.loc[df['Weather'].isnull(), 'Weather'] = np.round(predicted_values)

# Convert 'Barangay' back to integer type
df['Weather'] = df['Weather'].astype(int)

# Display missing values count after imputation
missing_data_counts = df.isnull().sum()
print("Missing Values After Imputation:\n", missing_data_counts)

# Display updated DataFrame
print(df.head())

# Revert only those originally missing values back to NaN
df.loc[original_missing['Victim Age'], 'Victim Age'] = np.nan
df.loc[original_missing['Day of the Week'], 'Day of the Week'] = np.nan
df.loc[original_missing['Barangay'], 'Barangay'] = np.nan
df.loc[original_missing['Arrest Made'], 'Arrest Made'] = np.nan
df.loc[original_missing['Suspect Age'], 'Suspect Age'] = np.nan

# Verify missing values after reverting
missing_values_reverted = df.isnull().sum()
print("Missing values per column (After reverting imputed values):\n", missing_values_reverted)

########################################################################################################
# suspect age part eyy
# Fill missing values with the column mean
df.loc[:, 'Victim Age'] = df['Victim Age'].fillna(round(df['Victim Age'].mean()))
df.loc[:, 'Day of the Week'] = df['Day of the Week'].fillna(round(df['Day of the Week'].mean()))
df.loc[:, 'Barangay'] = df['Barangay'].fillna(round(df['Barangay'].mean()))
df.loc[:, 'Arrest Made'] = df['Arrest Made'].fillna(round(df['Arrest Made'].mean()))

# Check for missing values after filling
missing_values_after = df.isnull().sum()
print("Missing values per column (After filling):\n", missing_values_after)

# Select features for prediction
features = ['Crime ID', 'Date', 'Day of the Week', 'Province', 'Municipality', 
            'Barangay', 'Street', 'Victim Age', 'Victim Gender', 'Arrest Made',
            'Weather', 'Suspect Gender', 'Outcome']
df_features = df[features].copy()
target = df['Suspect Age']

# Separate data into known and missing
df_known = df.dropna(subset=['Suspect Age']) 
df_missing = df[df['Suspect Age'].isnull()]

# Define X (features) and y (target)
X_train = df_known[features]
y_train = df_known['Suspect Age']
X_missing = df_missing[features]

# Train a Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing values
predicted_values = model.predict(X_missing)

# Round to ensure valid categorical values
df.loc[df['Suspect Age'].isnull(), 'Suspect Age'] = np.round(predicted_values)

# Convert 'Barangay' back to integer type
df['Suspect Age'] = df['Suspect Age'].astype(int)

# Display missing values count after imputation
missing_data_counts = df.isnull().sum()
print("Missing Values After Imputation:\n", missing_data_counts)

# Revert only those originally missing values back to NaN
df.loc[original_missing['Victim Age'], 'Victim Age'] = np.nan
df.loc[original_missing['Day of the Week'], 'Day of the Week'] = np.nan
df.loc[original_missing['Barangay'], 'Barangay'] = np.nan
df.loc[original_missing['Arrest Made'], 'Arrest Made'] = np.nan

#######################################################################################3

