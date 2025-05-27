# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:25:00 2025

@author: Talong PC
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("depression_data.csv")

# --- Streamlit App ---
st.title("Depression Data Visualization")

# 1. Distribution of Age
st.subheader("Distribution of Age")
fig, ax = plt.subplots()
sns.histplot(data["Age"], kde=True, ax=ax)
st.pyplot(fig)

# 2. Gender Distribution
st.subheader("Gender Distribution")
gender_counts = data["Gender"].value_counts()
fig, ax = plt.subplots()
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)


# 3. Depression vs. Suicidal Thoughts
st.subheader("Depression vs. Suicidal Thoughts")
fig, ax = plt.subplots()
sns.countplot(x="Suicidal thoughts ?", hue="Depression", data=data, ax=ax)
st.pyplot(fig)

# 4. CGPA Distribution
st.subheader("CGPA Distribution")
fig, ax = plt.subplots()
sns.histplot(data["CGPA"], kde=True, ax=ax)
st.pyplot(fig)

# 5. Academic Pressure vs. Depression
st.subheader("Academic Pressure vs. Depression")
fig, ax = plt.subplots()
sns.boxplot(x="Academic Pressure", y="Depression", data=data, ax=ax)
st.pyplot(fig)

# 6. Work/Study Hours vs. Depression (Scatter Plot)
st.subheader("Work/Study Hours vs. Depression")
fig, ax = plt.subplots()
sns.scatterplot(x="Work/Study Hours", y="Depression", data=data, ax=ax)
st.pyplot(fig)

# 7.  Sleep Duration vs Depression (Box Plot)
st.subheader("Sleep Duration vs. Depression")
fig, ax = plt.subplots()
sns.boxplot(x='Sleep Duration', y='Depression', data=data, ax=ax)
st.pyplot(fig)

# 8.  Financial Stress vs Depression (Boxplot)
st.subheader("Financial Stress vs. Depression")
fig, ax = plt.subplots()
sns.boxplot(x='Financial Stress', y='Depression', data=data, ax=ax)
st.pyplot(fig)
