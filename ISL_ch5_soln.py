import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

df1 = pd.read_csv('Data/Auto.csv', na_values='?').dropna()
df1.info()

t_prop = 0.5
p_order = np.arange(1, 11)
r_state = np.arange(0, 10)

X, Y = np.meshgrid(p_order, r_state, indexing='ij')
Z = np.zeros((p_order.size, r_state.size))

regr = skl_lm.LinearRegression()

# Generate 10 random splits of the dataset
for (i, j), v in np.ndenumerate(Z):
    poly = PolynomialFeatures(int(X[i, j]))
    X_poly = poly.fit_transform(df1.horsepower.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_poly, df1.mpg.ravel(),
                                                        test_size=t_prop, random_state=Y[i, j])

    regr.fit(X_train, y_train)
    pred = regr.predict(X_test)
    Z[i, j] = mean_squared_error(y_test, pred)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left plot (first split)
ax1.plot(X.T[0], Z.T[0], '-o')
ax1.set_title('Random split of the data set')

# Right plot (all splits)
ax2.plot(X, Z)
ax2.set_title('10 random splits of the data set')

for ax in fig.axes:
    ax.set_ylabel('Mean Squared Error')
    ax.set_ylim(15, 30)
    ax.set_xlabel('Degree of Polynomial')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(2, 11, 2))

