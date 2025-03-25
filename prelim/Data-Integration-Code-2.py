import pandas as pd

# Load data from Excel files
patients = pd.read_excel("patients.xlsx")
appointments = pd.read_excel("appointments.xlsx")
medical_records = pd.read_excel("medical_records.xlsx")
billing = pd.read_excel("billing.xlsx")

# Step 1: Merge patients with appointments on 'Patient_ID'
patients_appointments = pd.merge(patients, appointments, on='Patient_ID', how='inner')

# Step 2: Merge with medical records on 'Patient_ID'
patients_medical = pd.merge(patients_appointments, medical_records, on='Patient_ID', how='inner')

# Step 3: Merge with billing on 'Patient_ID'
final_data = pd.merge(patients_medical, billing, on='Patient_ID', how='inner')

# Display final integrated dataset
print(final_data.head())

# Save the integrated dataset to a new Excel file
final_data.to_excel("integrated_healthcare_data.xlsx", index=False)
