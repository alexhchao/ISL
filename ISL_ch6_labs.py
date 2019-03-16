# Thursday 3-14-2019
# 7 PM Argo tea
##########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:,.4f}'.format
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

pd.options.display.max_rows = 10
pd.options.display.max_columns = 20

import glmnet as gln
from glmnet import ElasticNet
# image not found
import glmnet_python
from glmnet import glmnet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
# cannot import name 'GeneralizedPoisson'

import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant

#%matplotlib inline
#plt.style.use('seaborn-white')

# In R, I exported the dataset from package 'ISLR' to a csv file.
df = pd.read_csv('Data/Hitters.csv', index_col=0).dropna()
df.index.name = 'Player'
df.info()

#df.query(" Salary.notnull() ", engine='python')
#df.query(" Salary.isnull() ", engine='python')




################
# use sklearn version
categorical_cols = ['NewLeague','League','Division']

_df = replace_with_dummies(df, categorical_cols)

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




######################



def replace_with_dummies(df, categorical_cols):
    """
    
    Parameters
    ----------
    df
    categorical_cols

    Returns
    -------

    """
    _dummies = pd.get_dummies(df.loc[:,categorical_cols ])

    return pd.concat([df.drop(categorical_cols, axis=1), _dummies],axis=1)

