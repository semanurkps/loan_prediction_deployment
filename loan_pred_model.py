# Loading data and preprocessing

import pandas as pd

loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )
#loan_data.head(5)                  # Shows us first 5 rows of the dataframe

#print(loan_data.shape)              # Returns shape of the data (rows,cols)
#print(loan_data.info())             # Returns info about dataframe's columns

#print(loan_data.isnull().sum())     # Returns sum of nulls for all columns

# Handling Missing Values

categoricals=[c for c in loan_data.columns if loan_data[c].dtypes=="object"]
#print(categoricals)                 # Takes categorical columns into another list

numeric=[c for c in loan_data.columns if c not in categoricals]
#print(numeric)                      # Takes numeric columns into another list

# Filling null values with mode for categorical features

for c in categoricals:
    if loan_data[c].isnull().sum()!=0:
        try:
            loan_data[c].fillna(loan_data[c].mode()[0],inplace=True)
        except:
            print("Nulls cannot be filled for {}".format(c))
#print(loan_data.isnull().sum())    # Make sure categoricals are filled!

# Filling null values with mode for numeric features

for n in numeric:
    if loan_data[n].isnull().sum()!=0:
        try:
            loan_data[n].fillna(loan_data[n].mode()[0],inplace=True)
        except:
            print("Nulls cannot be filled for {}".format(n))
#print(loan_data.isnull().sum())    # Make sure numerics are filled!

# Dropping unnecessary columns
loan_data.drop(["Unnamed: 0","Loan_ID"],axis=1,inplace=True)
#print(loan_data.head())

# New categoricals, these will be used for one-hot encoding
cat=[c for c in loan_data.columns if loan_data[c].dtypes=="object"]
# print(cat)

# One-hot encoding for categorical columns
one_hot=pd.get_dummies(data=loan_data, columns=cat,drop_first=True)
# print(one_hot.head())

# Train-test Splitting

from sklearn.model_selection import train_test_split

y=one_hot["Loan_Status"]
X=one_hot.drop("Loan_Status",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#print("Number transactions training datasets: ", X_train.shape)
#print("Number transactions testing datasets: ", X_test.shape)

# Handling class imbalance

#one_hot["Loan_Status"].value_counts()   # Returns the number of elements in each category

from imblearn.over_sampling import SMOTE

#print("Before OverSampling - # of label 1: {}".format(sum(y_train==1)))
#print("Before OverSampling - # of label 0: {} \n".format(sum(y_train==0)))

sm = SMOTE(sampling_strategy=1.0, random_state=25)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train)

#print("==============================================")

#print('After OverSampling - X_train shape: {}'.format(X_train_new.shape))
#print('After OverSampling - t_train shape: {} \n'.format(y_train_new.shape))

#print("After OverSampling - # of label 1: {}".format(sum(y_train_new==1)))
#print("After OverSampling - # of label 0: {}".format(sum(y_train_new==0)))

# Building the model

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

# Logistic Regression
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train_new,y_train_new)
preds=model_lr.predict(X_test)

#print(f1_score(y_test, preds))
#print(accuracy_score(y_test,preds))

# Classification report
#print(classification_report(y_test, preds, digits=4))

# Gridsearch to improve our model

from sklearn.model_selection import GridSearchCV

import numpy as np

log_grid_params={
    "C":np.logspace(-3,3,14),
    "penalty":["l1","l2"]
} # l1 lasso l2 ridge

from sklearn.model_selection import GridSearchCV

grid_log = GridSearchCV(model_lr, log_grid_params, cv=10)
grid_log.fit(X_train, y_train)
pred_grid=grid_log.predict(X_test)

#print("==========================================")
#print("Best parameters for Grid search is:")
#print(grid_log.best_params_)
#print("==========================================")
#print(f1_score(y_test,pred_grid))
#print(accuracy_score(y_test,pred_grid))

final_model=grid_log.best_estimator_
#final_model

# Testing

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')
#test_data.head()

# Preprocessing
test_data.drop("Loan_ID",axis=1,inplace=True)
#test_data.head(3)

#test_data.dtypes
#test_data.isnull().sum()

# Filling nulls for test data
for c in test_data.columns:
    if test_data[c].isnull().sum()!=0:
        try:
            test_data[c].fillna(test_data[c].mode()[0],inplace=True)
        except:
            print("Nulls cannot be filled for {}".format(c))
#test_data_data.isnull().sum()

# onehot encoding
test_onehot=pd.get_dummies(data=test_data,drop_first=True)
#test_onehot.head(2)

# Predicting with the trained model
target = final_model.predict(test_onehot)
print(target)

# Saving model to csv
#res = pd.DataFrame(target) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
#res.index = test_onehot.index # its important for comparison. Here "test_new" is your new test dataset
#res.columns = ["prediction"]
#res.to_csv("prediction_results.csv", index = False)

# Saving model to pickle
import pickle
import os

filename="loan_pred.pkl"
pickle.dump(final_model, open(filename,"wb"))