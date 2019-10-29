# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = 10, 5
import seaborn as sns
sns.set_style("darkgrid")

#%%
# Load training set
train_raw = pd.read_csv('./input/train.csv', index_col = 'PassengerId')
target = 'Survived'
features = [c for c in list(train_raw.columns.values) if c != target]

# list of features
train_raw.head()

#%%
# =============================================================================
# # DATA EXPLORATION & VISUALIZATION
# =============================================================================

#target values with their frequency
pd.value_counts(train_raw[target].values)/train_raw.shape[0]

#%%
# visualization of interactions btw survival and sex/class 
g = sns.factorplot(x='Pclass', y=target, hue='Sex', data=train_raw,
                   size=6, kind='bar', palette='muted')
g.despine(left=True)
g.set_ylabels("survival probability")

#%%
# visualization of interactions btw survival and sex/age 
sns.stripplot(x='Sex', y='Age', hue=target,
                       data=train_raw, jitter=True, palette='muted')

#%%
# =============================================================================
# # DATA PREPROCESSING
# =============================================================================
   
def data_preprocessing(data):
    # we do not touch the original dataframe
    df = data.copy()  

    # default age as median of training set
    default_age = train_raw['Age'].median()
    df['Age'].fillna(value=default_age, inplace=True)
      
    # extract title from Name
    df['Title'] = df['Name'].apply(lambda x: 
        x[x.find(',')+2:x.find('.')])
    # and regroup values to get a proxy for male/female + child/adult
    df['Title'].replace(['Mr','Don','Rev','Dr','Major','Sir','Col',
      'Capt','Jonkheer'], 'Mr', inplace=True) 
    df['Title'].replace(['Mrs','Mme','Lady','the Countess','Dona'], 
      'Mrs', inplace=True)     
    df['Title'].replace(['Miss','Ms','Mlle'], 'Miss', inplace=True) 
        
    # convert binary variables to 0/1 int values
    mapping = {'Sex': {'female': 0, 'male': 1}}
    df.replace(mapping, inplace=True)
    
    # encode categorical variables: Pclass, Embarked, Title
    return pd.get_dummies(df, columns=['Pclass', 'Title'], 
                          drop_first=True)

train_pp = data_preprocessing(train_raw)      

#%% PREPARE DATA FOR MACHINE LEARNING

train_set = train_pp.copy()
# drop features not used for the analysis
train_set.drop(columns=['Name','Ticket','Embarked','Cabin'], inplace=True)
X = train_set[[c for c in list(train_set.columns.values) if c != target]]
y = train_set[target]

# scale data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#%% 
# =============================================================================
# # LOGISTIC REGRESSION
# =============================================================================
clf1 = LogisticRegression()

# tune hyper-parameter using grid search
C = np.power(10, np.linspace(-2,2,41))
param_grid = {'penalty': ['l1','l2'],
              'C': C}
grid_search = GridSearchCV(clf1, param_grid=param_grid,
                           return_train_score=True,
                           cv=10) #10-fold validation 
grid_search.fit(X, y)

m = grid_search.best_score_
s = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
print('Best score = {:.4f}+/-{:.4f} (parameters: {})'.format(
        m, s, grid_search.best_params_))

#%% plot results of the grid search for hyper-parameters tuning

# load GridSearch results in a dataframe
df = pd.DataFrame.from_dict(grid_search.cv_results_)
# keep only rows for best penalty 
penalty = grid_search.best_params_['penalty']
df = df[df.param_penalty==penalty]

x_plot = -np.log10(df['param_C'].astype(np.float64))   # to be able to apply log10 function
# plot mean train value for best penalty
y_plot = df['mean_train_score']
y_dev = df['std_train_score']/3
plt.plot(x_plot, y_plot, label='Train',color='blue')
plt.fill_between(x_plot, y_plot-y_dev, y_plot+y_dev, color='blue', 
                 alpha=0.2)
# plot mean test value for best penalty
y_plot = df['mean_test_score']
y_dev = df['std_test_score']/3
plt.plot(x_plot, y_plot, label='Test', color='green')
plt.fill_between(x_plot, y_plot-y_dev, y_plot+y_dev, color='green', 
                 alpha=0.2)

# title, axis labels and legend
plt.title('Impact of regularization strength on fit (penalty=' 
                                                     + penalty + ')')
plt.xlabel('Regularization strengh (-log(C))')
plt.ylabel('Accuracy (mean +/- std error)')
plt.legend()
plt.show()

#%% 
# =============================================================================
# # RANDOM FOREST
# =============================================================================

clf2 = RandomForestClassifier()

# tune hyper-parameter using grid search
param_grid = {'n_estimators': [i for i in range(10,31,5)],
              'max_features': [i for i in range(2,10)],
              'min_samples_leaf': [i for i in range(1,20)]}
grid_search = GridSearchCV(clf2, param_grid=param_grid,
                           return_train_score=True, refit=True,
                           cv=3) #3-fold validation
grid_search.fit(X, y)

m = grid_search.best_score_
s = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
print('Best score = {:.4f}+/-{:.4f} (parameters: {})'.format(
        m, s, grid_search.best_params_))

#%% plot results of the grid search for hyper-parameters tuning

# load GridSearch results in a dataframe
df = pd.DataFrame.from_dict(grid_search.cv_results_)
# keep only rows for best n_estimator et max_features parameters 
n_trees = grid_search.best_params_['n_estimators']
n_feat = grid_search.best_params_['max_features']
df = df[(df.param_n_estimators==n_trees) & 
        (df.param_max_features==n_feat)]

#need to specify type for fill_between
x_plot = df['param_min_samples_leaf'].astype(np.int64) 
# plot mean train value for best number of estimators
y_plot = df['mean_train_score'].values
y_dev = df['std_train_score'].values/(2**0.5)
plt.plot(x_plot, y_plot, label='Train',color='blue')
plt.fill_between(x_plot, y_plot-y_dev, y_plot+y_dev, color='blue', 
               alpha=0.2)
# plot mean test value for best penalty
y_plot = df['mean_test_score']
y_dev = df['std_test_score']/(2**0.5)
plt.plot(x_plot, y_plot, label='Test', color='green')
plt.fill_between(x_plot, y_plot-y_dev, y_plot+y_dev, color='green', 
                 alpha=0.2)

# title, axis labels and legend
plt.title('Impact of minimum leaf size on fit \
          ({} trees, {} features max per tree)'.format(n_trees, n_feat))
plt.xlabel('Minimum leaf size')
plt.ylabel('Accuracy (mean +/- std error)')
plt.legend()
plt.show()

#%%
# =============================================================================
# # SVM
# =============================================================================

clf3 = SVC()

# tune hyper-parameter using grid search
C = np.power(10, np.linspace(-1,1,9))
gamma = np.power(10, np.linspace(-2,0,11))
param_grid = [{'kernel': ['poly'],
               'degree': [2,3],
              'C': C,
              'gamma': gamma},
              {'kernel': ['rbf','sigmoid'],
              'C': C,
              'gamma': gamma}] 

grid_search = GridSearchCV(clf3, param_grid=param_grid,
                           return_train_score=True, refit=True,
                           cv=3) #3-fold validation 
grid_search.fit(X, y)

m = grid_search.best_score_
s = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
print('Best score = {:.4f}+/-{:.4f} (parameters: {})'.format(
        m, s, grid_search.best_params_))

#%%

# load GridSearch results in a dataframe
df = pd.DataFrame.from_dict(grid_search.cv_results_)
# keep only rows for best n_estimator et max_features parameters 
k = grid_search.best_params_['kernel']
g = grid_search.best_params_['gamma']
df = df[(df.param_kernel==k) & 
        (df.param_gamma==g)]

#need to specify type for fill_between
x_plot = np.log10(C) 
# plot mean train value for best number of estimators
y_plot = df['mean_train_score'].values
y_dev = df['std_train_score'].values/(2**0.5)
plt.plot(x_plot, y_plot, label='Train',color='blue')
plt.fill_between(x_plot, y_plot-y_dev, y_plot+y_dev, color='blue', 
               alpha=0.2)
# plot mean test value for best penalty
y_plot = df['mean_test_score']
y_dev = df['std_test_score']/(2**0.5)
plt.plot(x_plot, y_plot, label='Test', color='green')
plt.fill_between(x_plot, y_plot-y_dev, y_plot+y_dev, color='green', 
                 alpha=0.2)

# title, axis labels and legend
plt.title('Impact of penalty coefficient of the error term on fit \
          (kernel: {}, gamma={})'.format(k, g))
plt.xlabel('Penalty coefficient')
plt.ylabel('Accuracy (mean +/- std error)')
plt.legend()
plt.show()

#%%
# =============================================================================
# # Make prediction on the test set and submit results
# =============================================================================

# using best SVC and refit on the total training set
clf = grid_search.best_estimator_
clf.fit(X,y)

# read and preprocess test data
test_raw = pd.read_csv('./input/test.csv', index_col = 'PassengerId')
test_raw['Fare'].fillna(value=test_raw['Fare'].mean(), inplace=True) 

test_pp = data_preprocessing(test_raw)    
test_pp.drop(columns=['Name','Ticket','Embarked','Cabin'], inplace=True)
X_test = scaler.transform(test_pp)

# make prediction and save the result
y_pred = clf.predict(X_test)
test_raw['Survived'] = y_pred

test_raw.to_csv('./output/result.csv', 
                columns=['Survived'],
                header=True,
                index=True)
