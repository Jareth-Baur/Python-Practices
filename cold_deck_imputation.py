# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:39:21 2025

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load the dataset
df = pd.read_excel('crime_data_with_issues.xlsx')

# Remove duplicate rows
df = df.drop_duplicates()

# Identify columns with missing data
missing_data_columns = df.columns[df.isnull().any()]
missing_data_counts = df.isnull().sum()

# Encode categorical values to numeric
encoder = LabelEncoder()
df['Barangay'] = encoder.fit_transform(df['Barangay'])
df['Suspect Gender'] = encoder.fit_transform(df['Suspect Gender'])
df['Province'] = encoder.fit_transform(df['Province'])
df['Municipality'] = encoder.fit_transform(df['Municipality'])

# Replace all 3s with NaN (Missing)
df.replace(3, np.nan, inplace=True)

# Count missing values after replacement
missing_data_counts = df.isnull().sum()

# 🔹 Create Dummy Historical Data
historical_data = {
    "Municipality": [0, 1, 2, 3, 4, 5],
    "Barangay": [10, 15, 20, 25, 30, 35]
}
reference_df = pd.DataFrame(historical_data)

# 🔹 Cold Deck Imputation Function
def cold_deck_imputation(df, target_column, reference_df, reference_column):
    """
    Imputes missing values using predefined historical/external data.
    
    df: DataFrame with missing values
    target_column: Column with missing values
    reference_df: External or historical dataset
    reference_column: Corresponding column in the reference dataset
    """
    df_copy = df.copy()
    
    # Create a mapping from reference data
    reference_mapping = reference_df.dropna().set_index(reference_column)[target_column].to_dict()

    # Fill missing values based on reference mapping
    df_copy[target_column] = df_copy[target_column].fillna(df_copy[reference_column].map(reference_mapping))

    return df_copy

# Apply Cold Deck Imputation to 'Barangay' using historical data
df = cold_deck_imputation(df, 'Barangay', reference_df, 'Municipality')

# FIX: Handle remaining NaNs using mode (most common value)
# Fix: Avoid chained assignment warning by explicitly reassigning df['Barangay']
df['Barangay'] = df['Barangay'].fillna(df['Barangay'].mode()[0])


# Convert 'Barangay' back to integer type
df['Barangay'] = df['Barangay'].astype(int)

# Display missing values count after imputation
missing_data_counts = df.isnull().sum()
print("Missing Values After Cold Deck Imputation:\n", missing_data_counts)

# Display updated DataFrame
print(df.head())
