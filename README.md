Customer Churn Analysis & Prediction
📌 Project Overview

This project focuses on analyzing customer churn behavior and predicting churn using Machine Learning techniques. It includes end-to-end data analysis starting from data preparation to advanced analytics like cohort analysis, customer lifetime value (CLV), and churn prediction.

🎯 Objectives
Analyze customer churn patterns
Identify key factors influencing churn
Perform cohort-based retention analysis
Estimate customer lifetime value (CLV)
Build a predictive model for churn
📂 Project Workflow
🔹 1. Data Collection
Attempt to load real-world datasets from the working directory
If not available, generate a synthetic subscription dataset
🔹 2. Data Preprocessing
Handle missing values
Convert date columns
Feature engineering (tenure, engagement, etc.)
🔹 3. Exploratory Data Analysis (EDA)
Customer acquisition trends
Monthly churn trends
Active customer base estimation
Tenure distribution
🔹 4. Churn Analysis
Churn rate by:
Billing cycle
Subscription plan
Customer segment
Acquisition channel
Identification of high-risk customer groups
🔹 5. Cohort Analysis
Group customers based on acquisition month
Analyze retention over time
Visualize retention using heatmaps
🔹 6. Customer Lifetime Value (CLV)
Estimate revenue generated per customer
Analyze CLV trends across cohorts
🔹 7. Machine Learning Model
Model Used: Logistic Regression
Goal: Predict probability of churn
Evaluation Metric: ROC-AUC Score
🛠️ Technologies Used
Python 🐍
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
📊 Key Insights
Early engagement significantly reduces churn
Customers with payment failures are more likely to churn
Annual subscription plans show lower churn rates
Higher support tickets indicate potential dissatisfaction
Cohort analysis reveals retention drop in early months
