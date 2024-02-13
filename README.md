# BANK_CHURN_PREDICTION

## Overview
This project aims to predict customer churn for a bank using different machine learning algorithms. Customer churn, in the context of banking, refers to the phenomenon where customers discontinue their relationship with the bank by closing their accounts or discontinuing the use of their services.
Predicting churn is crucial for banks to identify at-risk customers and take proactive measures to retain them. By leveraging historical data and various machine learning algorithms, this project seeks to build a predictive model that can forecast customer churn accurately.


## Dataset
The dataset used for this project contains historical information about bank customers, including demographics, account information, transaction history, and whether they churned or not.
### Source: https://www.kaggle.com/datasets/santoshd3/bank-customers
### Features
  *RowNumber: Sequential identifier for each row in the dataset.
  *CustomerId: Unique identifier for each customer.
  *Surname: Last name of the customer.
  *CreditScore: Numerical representation of the creditworthiness of the customer.
  *Geography: Country or region where the customer resides.
  *Gender: Gender of the customer.
  *Age: Age of the customer.
  *Tenure: Number of years the customer has been with the bank.
  *Balance: Account balance of the customer.
  *NumOfProducts: Number of bank products the customer is using.
  *HasCrCard: Binary variable indicating if the customer has a credit card (1) or not (0).
  *IsActiveMember: Binary variable indicating if the customer is an active member (1) or not (0).
  *EstimatedSalary: Estimated salary of the customer.

### Target Variable

  *Exited: Binary (0) or (1) to indicate whether customer is a member of the bank or exited.

## Methodology

1.  Exploratory Data Analysis (EDA): EDA is performed to gain insights into the data, identify patterns, correlations, and anomalies. Visualizations such as histograms, pie charts, and box plots are used for this purpose.

2.  Data Preprocessing: This step involves handling missing values, encoding categorical variables, scaling numerical features, and splitting the data into training and testing sets.


3.  Feature Engineering: New features may be created or existing features may be transformed to improve the model's performance. Techniques like one-hot encoding, feature scaling, and dimensionality reduction may be applied.

4.  Model Selection and Training: Various machine learning algorithms such as logistic regression, decision trees, random forests, are considered for building the churn prediction model. The models are trained using the training data and their performance is evaluated using appropriate metrics.

5.  Model Evaluation: The trained models are evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Hyperparameter tuning may be performed to improve the model's performance further.

