import numpy as np
import pandas as pd
import scipy.integrate as spi
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Function to integrate a mathematical function
def f(x):
    return x**2

x_vals = np.linspace(0, 2, 100)
y_vals = f(x_vals)
cumulative_integral = spi.cumulative_trapezoid(y_vals, x_vals, initial=0)
print("Cumulative Integral Result:", cumulative_integral[-1])

# Load datasets from CSV files
df_loan = pd.read_csv('loan.csv')
df_cancer = pd.read_csv('cancer.csv')

# Selecting relevant columns
df_loan_selected = df_loan[['no_of_dependents', ' loan_status']]
df_cancer_selected = df_cancer[['GENDER', 'LUNGCANCER']].copy()

# Convert categorical columns to numerical
df_cancer_selected['GENDER'] = df_cancer_selected['GENDER'].astype('category').cat.codes
df_cancer_selected['LUNGCANCER'] = df_cancer_selected['LUNGCANCER'].astype('category').cat.codes
df_loan_selected[' loan_status'] = df_loan_selected[' loan_status'].astype('category').cat.codes

# Merging the selected columns into a new dataset
new_dataset = pd.concat([df_loan_selected, df_cancer_selected], axis=1)

# Handle missing values by filling with mean
new_dataset = new_dataset.fillna(round(new_dataset.mean()))

# Display missing values count after imputation
missing_data_counts = new_dataset.isnull().sum()
print("Missing Values After Imputation:\n", missing_data_counts)


# Save the processed dataset
new_dataset.to_csv('integrated_dataset.csv', index=False)


# Data Sampling - Select a subset of the dataset
sampled_dataset = new_dataset.sample(frac=0.5, random_state=42)

# Data Aggregation - Grouping data and calculating mean for numerical features
aggregated_dataset = sampled_dataset.groupby('loan_status').mean().reset_index()

# Dimensionality Reduction using PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sampled_dataset)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Convert PCA results back to DataFrame
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])

# Numerosity Reduction using Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
sampled_dataset['Cluster'] = kmeans.fit_predict(scaled_data)

# Save the processed dataset
pca_df.to_csv('reduced_dataset.csv', index=False)

# Print dataset shapes to verify reduction
print(f"Original dataset shape: {new_dataset.shape}")
print(f"Sampled dataset shape: {sampled_dataset.shape}")
print(f"Aggregated dataset shape: {aggregated_dataset.shape}")
print(f"PCA reduced dataset shape: {pca_df.shape}")

# Plot PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=sampled_dataset['Cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Reduced Dataset')
plt.colorbar(label='Cluster')
plt.show()
