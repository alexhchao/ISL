# ch6 Tree methods
# Argo - Sunday - 3-17-2018
##################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pydot
from IPython.display import Image

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
from scipy.stats import zscore

from ols_functions import replace_with_dummies

# This function creates images of tree models using pydot
def print_tree(estimator, features, class_names=None, filled=True):
    tree = estimator
    names = features
    color = filled
    classn = class_names

    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, feature_names=features, class_names=classn, filled=filled)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return (graph)

#######################
# Paremters to tune for Random Forest

#n_estimators = number of trees in the foreset
#max_features = max number of features considered for splitting a node (use sqrt)
#max_depth = max number of levels in each decision tree
#min_samples_split = min number of data points placed in a node before the node is split
#min_samples_leaf = min number of data points allowed in a leaf node
#bootstrap = method for sampling data points (with or without replacement)



######################
df = pd.read_csv('Data/Hitters.csv', index_col=0).dropna()
df.index.name = 'Player'
df.info()

df.Hits.hist()

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

X_train, X_test , y_train, y_test = train_test_split(X, y,
                                                      test_size=0.5,
                                                  random_state=1)


#regr = DecisionTreeRegressor(max_leaf_nodes=3)
#regr.fit(X, y)

#graph, = print_tree(regr, features=['Years', 'Hits'])
#Image(graph.create_png())

###################
# Bagging: using all features

regr1 = RandomForestRegressor(max_features='sqrt',
                              n_estimators=300,
                              random_state=1)
regr1.fit(X_train, y_train)
#regr1.score()

pred = regr1.predict(X_test)

plt.scatter(pred, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

mean_squared_error(y_test, pred)

Importance = pd.DataFrame({'Importance':regr1.feature_importances_*100},
                          index=X.columns)
Importance.sort_values('Importance', axis=0,
                       ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')

all_mse = []
for n_trees in np.arange(1,301):
    regr1 = RandomForestRegressor(max_features='sqrt',
                                  n_estimators=n_trees,
                                  random_state=1)
    regr1.fit(X_train, y_train)

    pred = regr1.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    print("{}: {}".format(n_trees, MSE))
    all_mse.append(MSE)

mse_and_n_trees = pd.Series(all_mse, index = np.arange(1,301))
mse_and_n_trees.plot()




#series = diamonds_df_clean['color_H']
#pd.Series(_series).values


#list_bin_cols = [c for c in diamonds_df_clean.columns if is_binary(diamonds_df_clean[c])]

#_df = diamonds_df_clean.copy()

#def zscore(x):
#    return (x-x.mean())/x.std()






#pd.DataFrame(scale(diamonds_df_clean))


##for col in categorical_cols:
#    pd.get_dummies(diamonds_df['cut']).iloc[:, 1:]

#categorical_cols = ['cut','color','clarity']
#diamonds_df['color'].value_counts()

#pd.concat([pd.get_dummies(diamonds_df[c]).iloc[:, 1:] for c in categorical_cols],axis=1)



def replace_with_dummies(df, categorical_cols,
                         leave_one_out = True):
    """

    Parameters
    ----------
    df
    categorical_cols
    leave_one_out = for each categorical col, leave one out

    Returns
    -------

    """
    if leave_one_out:
        _dummies = pd.concat([pd.get_dummies(
            diamonds_df[c]).iloc[:, 1:] for c in categorical_cols], axis=1)
    else:
        _dummies = pd.get_dummies(df.loc[:, categorical_cols])

    return pd.concat([df.drop(categorical_cols, axis=1), _dummies], axis=1)


