# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load Dataset
df = pd.read_csv(r"C:\Users\sajit\OneDrive\Documents\Churn Distributionproject\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Keep original dataset for visualization
df_original = df.copy()


# Show dataset info
print(df.head())
print(df.info())


# Data Cleaning
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df.dropna(inplace=True)


# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)


# Split features and target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train Model
model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()


# Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df_original)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.savefig("churn_distribution.png")
plt.show()


# Churn by Contract
plt.figure(figsize=(7,5))
sns.countplot(x="Contract", hue="Churn", data=df_original)

plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.savefig("churn_by_contract.png")
plt.show()


# Monthly Charges vs Churn
plt.figure(figsize=(7,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df_original)

plt.title("Monthly Charges vs Churn")

plt.savefig("monthly_charges_vs_churn.png")
plt.show()
