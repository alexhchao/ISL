# Sun 3-17-2019
# Argo

##########################
# using diamonds data

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
from scipy.stats import zscore

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

from ols_functions import replace_with_dummies


from ggplot import diamonds

diamonds_df = diamonds

categorical_cols = ['cut','color','clarity']

diamonds_df_clean = replace_with_dummies(diamonds_df, categorical_cols)

y = diamonds_df_clean['price']

diamonds_z = zscore_but_ignore_binary_cols(diamonds_df_clean)

#diamonds_df_clean
diamonds_z.describe()


###########
import statsmodels.api as sm
import statsmodels.formula.api as smf
X = diamonds_z.drop('price',axis=1)

##############################
X_train, X_test , y_train, y_test = train_test_split(X, y,
                                                      test_size=0.5,
                                                  random_state=1)
###############################
# run OLS
from statsmodels.tools.eval_measures import mse, rmse

X = sm.add_constant(X)
res = sm.OLS(y, X).fit()
print(res.summary())

#np.sqrt(mse(res.predict(X_test), y_test))

mse(res.predict(X_test), y_test)

plot_pred_vs_actual(res.predict(X_test), y_test)

res.params.plot(kind='bar')

pd.Series(lasso.coef_, index=X.columns).plot(kind='bar')

################################
# run LASSO
lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
lassocv.fit(X_train, y_train)
print("Alpha=", lassocv.alpha_)

lasso = Lasso()
lasso.set_params(alpha=lassocv.alpha_)

lasso.fit(X_train, y_train)
print("mse = ",mean_squared_error(y_test, lasso.predict(X_test)))
print("best model coefficients:")

pd.Series(lasso.coef_, index=X.columns).plot(kind='bar')

plot_pred_vs_actual(lasso.predict(X_test), y_test)

#################################
# run random forest

regr1 = RandomForestRegressor(max_features='sqrt',
                              n_estimators=300,
                              random_state=1)
regr1.fit(X_train, y_train)
#regr1.score()

pred = regr1.predict(X_test)


mean_squared_error(y_test, pred)



Importance = pd.DataFrame({'Importance':regr1.feature_importances_*100},
                          index=X.columns)
Importance.sort_values('Importance', axis=0,
                       ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')




def plot_pred_vs_actual(pred, actual):
    """

    Parameters
    ----------
    pred
    actual

    Returns
    -------

    """
    plt.scatter(pred, actual, label='medv')
    plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
    plt.xlabel('pred')
    plt.ylabel('actual')
