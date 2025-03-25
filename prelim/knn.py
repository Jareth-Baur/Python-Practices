import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import numpy as np

# Load the Excel file
df = pd.read_excel("C:/Users/Talong PC/OneDrive/Documents/Python Projects/prelim/crime_data_with_issues.xlsx")

# Count duplicate rows
duplicate_count = df.duplicated().sum()

# Display duplicate rows
duplicate_rows = df[df.duplicated()]

# Remove duplicates
df_no_duplicates = df.drop_duplicates()
df = df_no_duplicates

# Count missing values per column before filling
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)

##################################### isa isahon tanan naay column naay number lampas 0
encoder = LabelEncoder()
df['Weather'] = encoder.fit_transform(df['Weather'])
df['Weather'] = df['Weather'].replace(5, np.nan)

# Apply KNN imputation (K=3, using nearest neighbors)  #mao rani irun basta numeric
knn_imputer = KNNImputer(n_neighbors=3, weights="uniform")
df[['Weather']] = knn_imputer.fit_transform(df[['Weather']])

# Convert back to original category labels
df['Weather'] = df['Weather'].round().astype(int)  # Ensure integer values
df['Weather'] = encoder.inverse_transform(df['Weather'])


###############################################
# Count missing values per column after filling
missing_values_after = df.isnull().sum()
print("\nMissing Values per Column (After Filling):")
print(missing_values_after)


#i run ranig na fill in na tanan or zero na tanan
output_path = "C:/finaldata.xlsx"
df.to_excel(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")
