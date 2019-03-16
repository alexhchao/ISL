# June 30, 2018
# saturday
# argo

###################
# function to use with statsmodels api

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
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def replace_with_dummies(df, categorical_cols):
    """

    Parameters
    ----------
    df
    categorical_cols

    Returns
    -------

    """
    _dummies = pd.get_dummies(df.loc[:, categorical_cols])

    return pd.concat([df.drop(categorical_cols, axis=1), _dummies], axis=1)



def plot_fitted_vs_resids(model):
    """
    plots fitted vs resids plot
    
    Parameters
    ----------
    ols_model - model from statsmodels

    Returns
    -------
    df
    """
    if not isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        raise ValueError("model is not a statsmodels model!")

    ax = sns.regplot(x = model.fittedvalues, y=model.resid, lowess = True)
    ax.set(xlabel='Fitted values', ylabel='Residuals')
    plt.show()


def variance_inflation_factors(exog_df):
    '''
    https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
    
    Parameters
    ----------
    exog_df : dataframe, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression.

    Returns
    -------
    vif : Series
        variance inflation factors
    '''
    exog_df = add_constant(exog_df)
    vifs = pd.Series(
        [1 / (1. - OLS(exog_df[col].values,
                       exog_df.loc[:, exog_df.columns != col].values).fit().rsquared)
         for col in exog_df],
        index=exog_df.columns,
        name='VIF'
    )
    return vifs


def mse(model):
    """
    
    Parameters
    ----------
    model

    Returns
    -------
    MSE - float
    """
    if not isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        raise ValueError("model is not a statsmodels model!")

    return (model.resid**2).mean()
