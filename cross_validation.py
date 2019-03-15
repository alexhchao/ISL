import pandas as pd
import numpy as np

pd.options.display.max_rows = 10

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split

credit = pd.read_csv('./data/Credit.csv')

credit = credit.iloc[:,1:]

auto = pd.read_csv('./Data/Auto.csv')
auto.replace("?",np.NaN, inplace=True)
auto['horsepower'] = pd.to_numeric(auto.horsepower)

########## plot learning curves for MSE

auto

#######
#split into train and test first
x_name = 'horsepower'
y_name = 'mpg'

auto_no_nulls = auto.dropna()
auto_no_nulls

X_train, X_test, y_train, y_test = train_test_split(auto_no_nulls.loc[:,x_name],
                                                    auto_no_nulls.loc[:,y_name],
                                                    test_size=0.50)

X_train
y_train



# order = 1
train = pd.concat([y_train, X_train],axis=1)
eqn = '{} ~ {}'.format(y_name, x_name)
model_1 = smf.ols(eqn, data = train).fit()
model_1.summary()

#######  train MSE / test MSE

model_1.mse_resid

train_mse = {}
test_mse = {}

train_mse[1]  = mean_squared_error(y_true = y_train, y_pred = model_1.fittedvalues)
test_mse[1] = mean_squared_error(y_true = y_test, y_pred = model_1.predict(X_test))

# order = 2


# order = 1
train = pd.concat([y_train, X_train],axis=1)
#eqn = '{} ~ {}'.format(y_name, x_name)
order = 2

eqn_2 = eqn + ' + I({}**{})'.format(x_name, order)

model_2 = smf.ols(eqn_2, data = train).fit()
model_2.summary()

mse_order_2_train = mean_squared_error(y_true = y_train, y_pred = model_2.fittedvalues)
mse_order_2_test = mean_squared_error(y_true = y_test, y_pred = model_2.predict(X_test))


# for i in 1,2 ...10:
n=10

_eqn = eqn

#for order in np.arange(2,n+1):
#    #print(order)
#    _eqn += ' + I({}**{})'.format(x_name, order)
#    print(_eqn)


for order in np.arange(2,n+1):
    #print(order)
    # cumulative terms
    #_eqn += ' + I({}**{})'.format(x_name, order)
    _eqn = '{} ~ I({}**{})'.format(y_name, x_name, order)
    print(order, _eqn)

    _model = smf.ols(_eqn, data = train).fit()
    _model.summary()

    train_mse[order] = mean_squared_error(y_true = y_train, y_pred = _model.fittedvalues)
    test_mse[order]= mean_squared_error(y_true = y_test, y_pred = _model.predict(X_test))

train_test_mse = pd.concat([
    pd.Series(train_mse),
    pd.Series(test_mse)], axis=1)

train_test_mse



_model.mse_resid