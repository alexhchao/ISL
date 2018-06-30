
# sat - 6-23-2018
# sitting in argo tea

# ch 4 linear regresion

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

#####################
# p 84,
# table 3.7
# male / female indicator variable

credit = pd.concat([credit,
           pd.get_dummies(credit['Gender'])],axis=1)


credit.groupby('Gender').mean()

X = credit.loc[:,['Female']]
y = credit['Balance']

X = sm.add_constant(X)

res = sm.OLS(y, X).fit()
print(res.summary())

### or another way to incorporate categorical variables

import statsmodels.formula.api as smf

df = credit.copy()
df

model = smf.ols("Balance ~ C(Gender)", data=df).fit()

model.summary()

model = smf.ols("Income ~ C(Gender)", data=df).fit().summary()

###############################################################

# multiple categories

smf.ols("Balance ~ C(Ethnicity)", data=df).fit().summary()

###############################################################
# 3.3.2 extenstions of the linear model
###############################################################

advertising = pd.read_csv('./data/Advertising.csv')
advertising = advertising.iloc[:,1:]

df = advertising.copy()

smf.ols('Sales ~ TV + Radio', data=df).fit().summary()


# adding interactions

smf.ols('Sales ~ Radio * TV', data=df).fit().summary()

# adding interactions among categorical / non-categorical variables

df_credit = credit.copy()

model = smf.ols('Balance ~ Income + C(Student)', data=df_credit).fit()


model = smf.ols('Balance ~ Income + C(Student) + Income * C(Student)', data=df_credit).fit()

model.summary()
#model.coef_

model.params

income = np.arange(0,151)

line_student = model.params[0] + model.params[1] + (model.params[2]+model.params[3])*income

line_non_student = model.params[0] + (model.params[2])*income

student_vs_non_student = pd.DataFrame({'student':line_student,
              'non_student':line_non_student})

plt.plot(line_student)

########## p 90, figure 3.7

student_vs_non_student.plot()


#### Non linear relationships

auto = pd.read_csv('./data/Auto.csv')

auto.horsepower.hist()

auto[auto=='?'] = np.NaN

# need to get scatter plot to work!


auto.horsepower[auto.horsepower=='?'] = np.NaN

auto.loc[:,['horsepower','mpg']].plot(kind='scatter')

ax = sns.regplot(x=auto['horsepower'], y=auto['mpg'], fit_reg=False)


plt.scatter(auto.horsepower, auto.mpg, facecolors='None', edgecolors='k', alpha=.5)


plt.scatter(auto.horsepower.sort_values(), auto.mpg, facecolors='None', edgecolors='k', alpha=.5)















