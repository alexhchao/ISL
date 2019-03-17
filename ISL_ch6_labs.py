# Thursday 3-14-2019
# 7 PM Argo tea
# Ch 6 - Linear Model Selection and Regularization (Lasso)
##########################

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # Do this before importing pyplot!
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/49918998/plt-show-not-working-in-pycharm

#import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
pd.options.display.float_format = '{:,.4f}'.format
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

pd.options.display.max_rows = 10
pd.options.display.max_columns = 20
#plt.interactive(True)

#import glmnet as gln
#from glmnet import ElasticNet
# image not found
#import glmnet_python
#from glmnet import glmnet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm
# cannot import name 'GeneralizedPoisson'

import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant

#from ols_functions import replace_with_dummies
from ols_functions import replace_with_dummies
# ah I need to add __init__.py to be able to import libraries

#%matplotlib inline
#plt.style.use('seaborn-white')

# In R, I exported the dataset from package 'ISLR' to a csv file.
df = pd.read_csv('Data/Hitters.csv', index_col=0).dropna()
df.index.name = 'Player'
df.info()

df.Hits.hist()

replace_with_dummies(df, categorical_cols = ['NewLeague','League','Division'])

#df.query(" Salary.notnull() ", engine='python')
#df.query(" Salary.isnull() ", engine='python')
################
# use sklearn version
categorical_cols = ['NewLeague','League','Division']

#_df = replace_with_dummies(df, categorical_cols)


X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
# Define the feature set X.
dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])

# you needto exclude one dummy from each category!!!
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)


#########
_df

y = _df['Salary']
X = _df.drop('Salary', axis=1)

smf.OLS(y, add_constant(X)).fit(data = _df).summary()
# works! with pandas 0.20.2

smf.ols('Salary ~ Hits + Runs', data=_df).fit().summary()
# doesnt fucking work, wtf is widepanel????? srsly!


####################
alphas = 10**np.linspace(10,-2,100)*0.5
alphas

ridge = Ridge()
coefs = []

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(scale(X), y)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization');
###########
lamb = 11498

ridge.set_params(alpha=lamb)
ridge.fit(scale(X), y)
ridge.coef_

pd.DataFrame(scale(X)).describe()

#####################
# LASSO
######################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.50, random_state=42)


lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas*2:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization');

#########################

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

# fit vs transform
# fit fits calcs the params, in this case, we calc the means and std of each factor
# transform then returns the transformed X
######################



# scaler.mean_ shows mean of eery columns
scaler.mean_
scaler.std_

X_train.describe()

zscores = StandardScaler().fit(X_train)



alphas = 10**np.linspace(10,-2,100)*0.5

ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error')
ridge_cv_model = ridgecv.fit(zscores.transform(X_train), y_train)

ridge_cv_model.alpha_
# 115

ridge2 = Ridge()

ridge2.set_params(alpha=ridgecv.alpha_)
ridge2.fit(zscores.transform(X_train), y_train) # no need to scale y

mean_squared_error(y_test, ridge2.predict(StandardScaler().fit_transform(X_test)))
#144,886

#mean_squared_error(y_test, ridge2.predict(scale(X_test)))

pd.Series(ridge2.coef_, index=X.columns)


###############
# Lasso
lassocv = LassoCV(alphas=None, cv=10, max_iter=10000)
#lassocv.fit(scale(X_train), y_train.values.ravel())

lassocv.fit(scale(X_train), y_train.values)
lassocv.alpha_

###

lasso = Lasso()

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(scale(X_train), y_train)
mean_squared_error(y_test, lasso.predict(scale(X_test)))

lasso.intercept_
pd.Series(lasso.coef_, index=X.columns).plot(kind='bar')

lassocv_2 = LassoCV(eps=0.0001,
        n_alphas=400,
        max_iter=200000, cv=10,
        normalize=False, random_state=9)

lassocv_2.fit(scale(X_train), y_train.values)
lassocv_2.alpha_


lasso.set_params(alpha=lassocv_2.alpha_)
lasso.fit(scale(X_train), y_train)
mean_squared_error(y_test, lasso.predict(scale(X_test)))



####################
# try another data set

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

pd.DataFrame(diabetes_X_train).describe()


regr = Lasso(alpha=.3)
regr.fit(scale(diabetes_X_train), diabetes_y_train)
regr.coef_

regr.score(scale(diabetes_X_test), diabetes_y_test)


#k_fold = cross_validation.KFold(n=400, k=10, indices=True)

lasso = LassoCV(cv=10)
X_diabetes = diabetes.data
y_diabetes = diabetes.target

lasso.fit(scale(X_diabetes), y_diabetes)
lasso.alpha_


#########

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

######################
#plt.show(block=True)

df.Hits.hist()
######################
# https://www.kaggle.com/floser/aw6-the-lasso-cross-validated
# using Lasso on the baseball data

alphas = 10**np.linspace(6,-2,50)*0.5
alphas

lasso = Lasso(max_iter=10000, normalize=True)
coefs = []

# for each alpha calc the coefs
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X, y)
    lasso.score(scale(X),y)
    coefs.append(lasso.coef_)

np.shape(coefs)

pd.DataFrame(coefs)

# print coefs

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


###
X_train, X_test , y_train, y_test = train_test_split(X, y,
                                                      test_size=0.5,
                                                  random_state=1)

lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
lassocv.fit(X_train, y_train)
lasso.set_params(alpha=lassocv.alpha_)
print("Alpha=", lassocv.alpha_)
lasso.fit(X_train, y_train)
print("mse = ",mean_squared_error(y_test, lasso.predict(X_test)))
print("best model coefficients:")

pd.Series(lasso.coef_, index=X.columns).plot(kind='bar')

