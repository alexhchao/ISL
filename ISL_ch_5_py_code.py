#%matplotlib inline
import pandas as pd
import numpy as np

pd.options.display.max_rows = 10

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

import statsmodels.api as sm
import statsmodels.formula.api as smf

credit = pd.read_csv('./data/Credit.csv')

credit = credit.iloc[:,1:]

auto = pd.read_csv('./Data/Auto.csv', na_values='?').dropna()
auto.info()

###############
# learning curves
##################3
X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(2)

poly.fit_transform(X)

X = auto.loc[:,['horsepower']]

X_2 = poly.fit_transform(X)

y = auto['mpg']

#X = sm.add_constant(X)

model = sm.OLS(y, PolynomialFeatures(6).fit_transform(X)).fit()

model.fittedvalues

mse = ((y - model.fittedvalues)**2).mean()

mse

print(res.summary())

def calc_mse(model, y_true):
    return ((y_true - model.fittedvalues)**2).mean()

calc_mse(model, y)

model.fittedvalues

mse = ((y - model.fittedvalues)**2).mean()

mse
#poly = PolynomialFeatures(interaction_only=True)
#poly.fit_transform(X)
#PolynomialFeatures(1).fit_transform(X)

# training mse

################3
# test mse ?

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

mse_train = {}
mse_test = {}
for order in np.arange(1,11):
    model_train = sm.OLS(y_train, PolynomialFeatures(order).fit_transform(X_train)).fit()
    mse_train[order] = calc_mse(model_train, y_train)

    model_test = sm.OLS(y_test, PolynomialFeatures(order).fit_transform(X_test)).fit()
    mse_test[order] = calc_mse(model_test, y_test)

mse_train_test = pd.concat([pd.Series(mse_train),
           pd.Series(mse_test)],axis=1)

mse_train_test.iloc[:8].plot()



############# using sklearn cross validation

regr = skl_lm.LinearRegression()

_mses = {}
for x,y in [(X_train, y_train),(X_test, y_test)]:
    model_test = sm.OLS(y_test, PolynomialFeatures(order).fit_transform(X_test)).fit()
    _mses[order] = calc_mse(model_test, y_test)



##############################
# k fold cross validation
#################################

mse_train = {}
mse_test = {}
for order in np.arange(1,11):
    mse_train[order] = (-1)*cross_val_score(regr, PolynomialFeatures(order).fit_transform(X_train),
                                            y_train, cv=10,
                                       scoring = 'neg_mean_squared_error').mean()

    mse_test[order] = (-1)*cross_val_score(regr, PolynomialFeatures(order).fit_transform(X_test),
                                       y_test, cv=10,
                                       scoring='neg_mean_squared_error').mean()


mse_train_test = pd.concat([pd.Series(mse_train),
           pd.Series(mse_test)],axis=1)

mse_train_test.plot()
