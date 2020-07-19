This repository contains Credit card fraud detection algorithm using machine learning techniques in python.

Credit Card Fraud Detection : With a lot of people, banks and online retailer being a victim of credit card fraud, a model detecting whether the transaction is fraud or not can help in saving a huge amount of money.

Dataset : The dataset has been obtained from kaggle. This dataset contains 284807 rows and 30 numeric columns and one class that specifies whether the transaction is fraudulent or not. The values of columns V1-V28 in the dataset may be result of a PCA(Principal Component Analysis) Dimensionality reduction to protect user identities and sensitive information. There are no missing values in the dataset.

Algorithm : The algorithm used in this dataset is Random Forest algorithm. For model improvement normalization and SMOTE techniques (to handle imbalanced data) were used.

Visualisation : The library used for visualizing the data, confusion matrix etc. is seaborn.

You can find the dataset in the link provided : https://www.kaggle.com/mlg-ulb/creditcardfraud

Following accuracy was obtained :

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85296
           1       0.91      0.80      0.86       147
   micro avg       1.00      1.00      1.00     85443
   macro avg       0.96      0.90      0.93     85443
weighted avg       1.00      1.00      1.00     85443
