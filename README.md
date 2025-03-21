#  Unlocking Customer Retention: Churn Analysis and Prediction

Problem Statement: Telco a telecommunication company experienced a surge in short-term customer churn. This issue was addressed by customer behavior analysis to find potential causes and patterns of churn and a predictive model to identify at-risk customers.

Data Collection: IBM Sample dataset from Kaggle is utilized.

Analysis: 

Part 1: Analyzed the customer behavior to answer the following questions-
1)	Within what timeframe does a significant portion of customer churn occur?
2)	What is the tenure and CLTV characteristics of high-value customers?
3)	What are the key characteristics of customers who are likely to churn?
4)	Which specific services or contract types are associated with higher churn rates?
5)	What are the possible reasons of churn by the customers?
6)	What is the retention period of maximum customers?
7)	What are the strategies that can be implemented to reduce churn value?

Part 2: Built predictive models such as Random Forest Classifier, Logistic Regression Model and Support Vector Machine Classifier that identifies customers likely to be churned and customers likely to be continue with the help of features like contract type, services, tenure, charges, CLTV and few more. Evaluated all models on the basis of accuracy, recall, precision and ROC-AUC score. Deployed the random forest classifier for real-time analysis.

Result: The model achieved 95% accuracy with 99% ability to identify churned and non-churned customers. The analysis helped Telco Company to understand potential causes and pattern of churned customers. This enables Telco to proactively identify and target high risk customers for retention efforts.
