import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import mutual_info_regression

import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def read_data_from_csv(path):
    """Load datasets from CSV files.
    Args:
    path (str): Path to the CSV file.
    Returns:
    X (np.ndarray): Features of samples.
    y (np.ndarray): Labels of samples, only provided in the public
   datasets.
    """
    assert os.path.exists(path), f'File not found: {path}!'
    assert os.path.splitext(path)[
               -1] == '.csv', f'Unsupported file type {os.path.splitext(path)[-1]}!'
    data = pd.read_csv(path)
    column_list = data.columns.values.tolist()
    if 'Label' in column_list:
        # for the public dataset, label column is provided.
        column_list.remove('Label')
        X = data[column_list].values
        y = data['Label'].astype('int').values
        return X, y
    else:
        # for the private dataset, label column is not provided.
        X = data[column_list].values
        return X


X_public, y_public = read_data_from_csv('assignment_3_public.csv')
print('Shape of X_public:', X_public.shape)  # n_sample, m_feature (590, 14)
print('Shape of y_public:', y_public.shape)  # n_sample (590,)

# build a decision tree model
model = DecisionTreeClassifier()

# feature engineering
# 1. scalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_public = scaler.fit_transform(X_public)

# 2. feature select
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=5)
X_public = selector.fit_transform(X_public, y_public)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'ccp_alpha': [0.0, 0.01, 0.1, 0.2, 0.5]  # pruning, avoid overfitting
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_public, y_public)

print("The best params:", grid_search.best_params_)

print("Best score during training on X_train:", grid_search.best_score_)

best_model = DecisionTreeClassifier(**grid_search.best_params_)
best_model.fit(X_public, y_public)
y_pred = best_model.predict(X_public)
print("Report result on public dataset with the best model:")
print(classification_report(y_public, y_pred))


X_private = read_data_from_csv('assignment_3_private.csv')
print('Shape of X_private:', X_private.shape)

# remove and make your own predictions.
X_private = scaler.transform(X_private)
X_private = selector.transform(X_private)
preds = best_model.predict(X_private)
submission = pd.DataFrame({'Label': preds})
submission.to_csv('assignment_3.csv', index=True, index_label='Id')
