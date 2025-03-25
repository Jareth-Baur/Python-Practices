import pandas as pd

# Load datasets from Excel files
customers = pd.read_excel("C:/Users/Talong PC/OneDrive/Documents/Python Projects/prelim/customers.xlsx")
orders = pd.read_excel("C:/Users/Talong PC/OneDrive/Documents/Python Projects/prelim/orders.xlsx")
payments = pd.read_excel("C:/Users/Talong PC/OneDrive/Documents/Python Projects/prelim/payments.xlsx")

# Step 1: Merge customers and orders on 'Customer_ID'
customer_orders = pd.merge(customers, orders, on='Customer_ID', how='inner')

# Step 2: Merge the result with payments on 'Order_ID'
final_data = pd.merge(customer_orders, payments, on='Order_ID', how='inner')

# Display final integrated dataset
print(final_data.head())

# Save the integrated dataset to a new Excel file
final_data.to_excel("integrated_data.xlsx", index=False)




# Scenario 1: Inconsistent Data Formats
# Convert 'Date' column in Appointments dataset to 'YYYY-MM-DD'
## appointments['Date'] = pd.to_datetime(appointments['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

# Convert 'Date' column in Billing dataset to 'YYYY-MM-DD' (if needed)
## billing['Date'] = pd.to_datetime(billing['Date']).dt.strftime('%Y-%m-%d')


# Scenario 2: Categorical Data Standardization
# Standardize Gender values
## patients['Gender'] = patients['Gender'].replace({
##   'F': 'Female', 'M': 'Male', 'male': 'Male', 'female': 'Female'
## })
