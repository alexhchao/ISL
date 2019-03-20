# Tues - March 19, 2019
#########################

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

credit = pd.read_csv('./data/Credit.csv')

credit = credit.iloc[:,1:]

credit

sns.pairplot(credit[['Balance','Age','Cards','Education','Income','Limit','Rating']])

est = smf.ols('Balance ~ Gender', credit).fit()
est.summary()

credit_2 = pd.concat([credit,
           pd.get_dummies(credit['Gender'])],axis=1)

X = credit_2.loc[:,['Female']]
y = credit['Balance']

X = sm.add_constant(X)

res_1 = sm.OLS(y, X).fit()
print(res_1 .summary())

####################
# okay, now, what happens if we include BOTH male and female dumies? and no intercept?
credit_2.describe()

X_2 = credit_2.iloc[:,-2:]
y_2 = credit['Balance']

#X = sm.add_constant(X)

res_2 = sm.OLS(y_2, X_2).fit()
print(res_2.summary())

variance_inflation_factors(X)

#################
# same model!

est = smf.ols('Balance ~ Ethnicity', credit).fit()
est.summary()

X = credit_2.iloc[:,-2:]
y = credit['Balance']

X = sm.add_constant(X)

res_2 = sm.OLS(y, X).fit()
print(res_2.summary())

np.sum(res_1.predict(X)-res_2.predict(X_2))
# same model

##############
# table 3.8
res_3 = smf.ols('Balance ~ Ethnicity', credit).fit()
print(res_3.summary())

X_3 = pd.get_dummies(credit.Ethnicity)

res_4 = sm.OLS(y, X_3).fit()
print(res_4.summary())

#############
# Weighted least squares

#np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
nsample = 50
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, (x - 5)**2))
X = sm.add_constant(X)
beta = [5., 0.5, -0.01]
sig = 0.5
w = np.ones(nsample)
w[nsample * 6//10:] = 3
y_true = np.dot(X, beta)
e = np.random.normal(size=nsample)
y = y_true + sig * w * e
X = X[:,[0,1]]


mod_wls = sm.WLS(y, X, weights=1./(w ** 2))
res_wls = mod_wls.fit()
print(res_wls.summary())

ols_get_coefs(X,y, w = np.diag(1./(w ** 2)) )
# cool weighted least sq is working

plt.scatter(X[:,1], y)


