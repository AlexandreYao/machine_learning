# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 23:00:46 2019

@author: AlexandreYao
"""

#import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pandas.api.types import CategoricalDtype

import seaborn as sns


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Dowload data
data_train = pd.read_csv("adult.data", sep=" *, *", na_values="?")
data_test = pd.read_csv("adult.test", sep=" *, *", na_values="?")
data_train["income"] = data_train["income"].apply(lambda x : 0 if x=="<=50K" else 1)
data_test["income"] = data_test["income"].apply(lambda x : 0 if x=="<=50K." else 1)


# Trying to know about data : are there any missing data ?
data_train.info()
data_test.info()
# => missing data in columns "workclass", "occupation" and "native-country"
att_with_NA = ["workclass", "occupation", "native-country"]


# Handling Numerical Columns
num_attributes_train = data_train.select_dtypes(include=['int64'])
num_attributes_test = data_test.select_dtypes(include=['int64'])

    #1 Data Visualizations
num_attributes_train.hist(figsize=(10,10))
num_attributes_test.hist(figsize=(10,10))

# Handling Categorical Columns

    #1 Data Visualizations
'''plt.figure(3)
sns.countplot(y='workclass', hue='income', data = data_train)
plt.show()
plt.figure(4)
sns.countplot(y='workclass', hue='income', data = data_test)
plt.show()
plt.figure(5)
sns.countplot(y='occupation', hue='income', data = data_train)
plt.show()'''

# Preprocessing

    #1 Drop useless Columns
data_train.drop(['fnlwgt', 'education'], axis=1, inplace=True)
num_attributes_train.drop(['fnlwgt'], axis=1, inplace=True)
num_attributes_test.drop(['fnlwgt'], axis=1, inplace=True)
data_test.drop(['fnlwgt', 'education'], axis=1, inplace=True)

    #2 Missing data
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(data_train[att_with_NA])
data_train[att_with_NA] = imputer.transform(data_train[att_with_NA])
#data_train.info()
imputer.fit(data_test[att_with_NA])
data_test[att_with_NA] = imputer.transform(data_test[att_with_NA])
#data_test.info()

    # Target and predictors
predictors_train = data_train
predictors_test = data_test
target_train = data_train['income']
target_test = data_test['income']
predictors_train.drop(['income'], axis=1, inplace=True)
predictors_test.drop(['income'], axis=1, inplace=True)
att_num_predictors = list(num_attributes_train.columns.values)
att_num_predictors.remove('income')

    #3 Feature Scaling
sc = StandardScaler()
predictors_train[att_num_predictors] = sc.fit_transform(predictors_train[att_num_predictors])
predictors_test[att_num_predictors] = sc.fit_transform(predictors_test[att_num_predictors])

        # visualising data distribution after scaling
'''plt.figure(9)
plt.title('After Scaling predictor train: StandardScaler')
sns.kdeplot(predictors_train['age'])
sns.kdeplot(predictors_train['education-num'])
sns.kdeplot(predictors_train['capital-gain'])
sns.kdeplot(predictors_train['capital-loss'])
sns.kdeplot(predictors_train['hours-per-week'])
plt.show()

plt.figure(10)
plt.title('After Scaling predictor test : StandardScaler')
sns.kdeplot(predictors_test['age'])
sns.kdeplot(predictors_test['education-num'])
sns.kdeplot(predictors_test['capital-gain'])
sns.kdeplot(predictors_test['capital-loss'])
sns.kdeplot(predictors_test['hours-per-week'])
plt.show()'''

    #4 Encoding categorical data
cat_columns = ['workclass',
               'marital-status',
               'occupation',
               'relationship',
               'race',
               'sex',
               'native-country']
join_df = pd.concat([predictors_train, predictors_test])
join_df = join_df.select_dtypes(include=['object'])
categories = dict()
for column in join_df.columns:
    categories[column] = join_df[column].value_counts().index.tolist()

cat_predictors_train = predictors_train.select_dtypes(include=['object'])
for column in cat_predictors_train.columns:
    cat_predictors_train[column] = cat_predictors_train[column].astype({column:
        CategoricalDtype(categories[column])})
dummies_train = pd.get_dummies(cat_predictors_train, drop_first=True)
cat_predictors_test = predictors_test.select_dtypes(include=['object'])
for column in cat_predictors_test.columns:
    cat_predictors_test[column] = cat_predictors_test[column].astype({column:
        CategoricalDtype(categories[column])})
dummies_test = pd.get_dummies(cat_predictors_test, drop_first=True)

predictors_train.drop(cat_columns, axis=1, inplace=True)
predictors_test.drop(cat_columns, axis=1, inplace=True)

predictors_train = pd.concat([predictors_train, dummies_train], axis = 1)
predictors_test = pd.concat([predictors_test, dummies_test], axis = 1)

# Training the model
model = LogisticRegression(random_state=0)
model.fit(predictors_train, target_train)

# Testing the model
predicted_target = model.predict(predictors_test)

# Model Evaluation
accuracy_score(predicted_target, target_test.values)

cfm = confusion_matrix(predicted_target, target_test.values)
sns.heatmap(cfm, annot=True)
plt.xlabel('Predicted Target')
plt.ylabel('Actual classes')
