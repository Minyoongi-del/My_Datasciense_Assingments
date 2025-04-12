#!/usr/bin/env python
# coding: utf-8



# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ##  Data Exploration & Preprocessing


# Load the Titanic train and test datasets
train_data = pd.read_csv('D:\\Logistic Regression\\Titanic_train.csv')
test_data = pd.read_csv('D:\\Logistic Regression\\Titanic_test.csv')




# Visualize the distribution of numerical features
train_data.hist(bins=20, figsize=(10, 8))


# Boxplot to see the relationship between 'Age' and 'Survived'
sns.boxplot(x='Survived', y='Age', data=train_data)



# Select only numeric columns for correlation
numeric_data = train_data.select_dtypes(include=['float64', 'int64'])

# Correlation heatmap to observe feature relationships
corr = numeric_data.corr()

# Plot the heatmap with annotations
sns.heatmap(corr, annot=True, cmap='coolwarm')


# ## Data Preprocessing


# Handle missing values in train data before any encoding
# Impute missing 'Age' values with the median
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

# Since 'Embarked' is already one-hot encoded, no need to fill missing values in 'Embarked'
# Check if 'Embarked' exists before attempting to fill missing values
if 'Embarked' in train_data.columns:
    train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
else:
    print("Column 'Embarked' not found in train_data (likely already one-hot encoded)")

# Encode categorical variables like 'Sex' and 'Embarked' in train data
# This step is redundant now since 'Sex' and 'Embarked' are already encoded
# train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)

# Handle missing values in the test dataset
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# Add missing columns from train_data to test_data (except 'Survived')
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    if col != 'Survived':  # Ignore 'Survived' as it's not in test_data
        test_data[col] = 0

# Align test_data columns with train_data (excluding 'Survived')
test_data = test_data[train_data.columns.drop('Survived')]


# ## Model Building

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Encode categorical variables in train_data and test_data
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Define features (independent variables) and target (dependent variable)
X = train_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = train_data['Survived']

# Split the dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize the logistic regression model with increased max_iter
model = LogisticRegression(max_iter=200)

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)


# ## Model Evaluation

# In[30]:


import warnings
warnings.filterwarnings('ignore')
# Import metrics to evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# ROC-AUC score and ROC curve
y_pred_proba = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)

# Compute false positive rate and true positive rate for the ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)



# Interpret the model coefficients to understand feature importance
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]})

# Higher positive coefficients indicate that the feature increases the likelihood of survival
# Negative coefficients decrease the likelihood of survival

import joblib

# Save the model
joblib.dump(model, 'titanic_logistic_regression_model.pkl')
# Save the scaler
joblib.dump(scaler, 'titanic_scaler.pkl')
