# Customer Churn Prediction

This project predicts customer churn using Machine Learning. The goal is to analyze telecom customer data and identify whether a customer is likely to leave the service.

Customer churn prediction helps companies understand customer behavior and take actions to improve customer retention.

---

## Project Overview

Customer churn occurs when a customer stops using a company's service. In this project, we use a telecom dataset to build a machine learning model that predicts churn based on customer features such as contract type, monthly charges, internet service, and tenure.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Machine Learning Model

The project uses the **Logistic Regression** algorithm to classify whether a customer will churn or not.

---

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Machine Learning Model Training
6. Model Evaluation
7. Data Visualization

---

## Dataset

Dataset used: **Telco Customer Churn Dataset**

The dataset contains telecom customer information including:

- Gender
- Contract type
- Internet service
- Monthly charges
- Total charges
- Tenure
- Churn status

---

## Model Performance

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

---

## Visualizations

### Customer Churn Distribution

![Churn Distribution](churn_distribution.png)

---

### Churn by Contract Type

![Churn by Contract](churn_by_contract.png)

---

### Monthly Charges vs Churn

![Monthly Charges vs Churn](monthly_charges_vs_churn.png)

---

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

---

## Key Insights

- Customers with **month-to-month contracts** have a higher churn rate.
- Customers with **higher monthly charges** are more likely to churn.
- Long-term contracts reduce churn probability.

---

## Author

**Sajith**  
B.Tech Computer Science Student  
Aspiring Data Analyst / Data Scientist

---

## Future Improvements

- Try other machine learning models (Random Forest, XGBoost)
- Perform hyperparameter tuning
- Deploy the model as a web application
