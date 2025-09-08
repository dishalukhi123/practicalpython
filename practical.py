import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm

# Load dataset
df = pd.read_csv("household_dataset.csv")

# -----------------------------
# 3. Types of Data
# -----------------------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

# -----------------------------
# 4. Central Tendency
# -----------------------------
mean_income = df['Household_Income'].mean()
median_income = df['Household_Income'].median()
mode_income = df['Household_Income'].mode()[0]

mean_age = df['Age_of_Household_Head'].mean()
median_age = df['Age_of_Household_Head'].median()
mode_age = df['Age_of_Household_Head'].mode()[0]

print("\nCentral Tendency of Income:")
print(f"Mean: {mean_income:.2f}, Median: {median_income}, Mode: {mode_income}")
print("\nCentral Tendency of Age:")
print(f"Mean: {mean_age:.2f}, Median: {median_age}, Mode: {mode_age}")

# # -----------------------------
# # 5. Measures of Dispersion
# # -----------------------------
range_income = df['Household_Income'].max() - df['Household_Income'].min()
variance_income = df['Household_Income'].var()
std_income = df['Household_Income'].std()
iqr_income = df['Household_Income'].quantile(0.75) - df['Household_Income'].quantile(0.25)

print("\nMeasures of Dispersion (Income):")
print(f"Range: {range_income}, Variance: {variance_income:.2f}, Std Dev: {std_income:.2f}, IQR: {iqr_income}")

# # -----------------------------
# # 6. Distribution
# # -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Household_Income'], bins=20, kde=False, color='skyblue', stat="density")

# Fit Gaussian Normal Curve
mu, sigma = df['Household_Income'].mean(), df['Household_Income'].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'r', linewidth=2)

plt.title("Household Income Distribution with Normal Curve")
plt.xlabel("Household Income")
plt.ylabel("Density")
plt.show()

# -----------------------------
# Histogram + KDE for Household Income
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Household_Income'], bins=20, kde=True, color='skyblue')
plt.title("Histogram + KDE of Household Income")
plt.xlabel("Household Income")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# Boxplot: Household Income by Education Level
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x='Education_Level', y='Household_Income', data=df, palette="Set2")
plt.title("Household Income by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Household Income")
plt.show()

# -----------------------------
# Boxplot: Household Income by Urban/Rural
# -----------------------------
plt.figure(figsize=(6,5))
sns.boxplot(x='Urban_Rural', y='Household_Income', data=df, palette="Set3")
plt.title("Household Income by Urban/Rural")
plt.xlabel("Area")
plt.ylabel("Household Income")
plt.show()

# -----------------------------
# Boxplot: Family Size by Education Level
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x='Education_Level', y='Family_Size', data=df, palette="coolwarm")
plt.title("Family Size by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Family Size")
plt.show()

# -----------------------------
# Distribution Curve: Age vs Income
# -----------------------------
plt.figure(figsize=(8,6))
sns.kdeplot(x='Age_of_Household_Head', y='Household_Income', data=df, cmap="Blues", fill=True)
plt.title("Distribution Curve: Age vs Income")
plt.xlabel("Age of Household Head")
plt.ylabel("Household Income")
plt.show()
