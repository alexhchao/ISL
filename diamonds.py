# diamonds

import numpy as np
import pandas as pd
import seaborn as sns
from ggplot import diamonds
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = diamonds


df.head()

df.cut.unique()


list_cat_cols = ['cut','color','clarity']

[df[c].unique() for c in df.columns if c in list_cat_cols]



smf.ols("price ~ carat + C(cut) + C(color) + C(clarity)", data=df).fit().summary()
# baseline model
# Rsqr = .916
#

smf.ols("price ~ carat + C(cut) + C(color) + C(clarity) + carat * C(cut) ", data=df).fit().summary()


smf.ols("price ~ carat + C(cut) + C(color) + C(clarity) + "
        "carat * C(cut) + carat * C(color) + carat * C(clarity)", data=df).fit().summary()

smf.ols("price ~ carat + I(carat**2)", data=df).fit().summary()

#################################
# what measure is best? out of sample R squared ?
####################################

# ideas
# 1) include all interactions and also quadratic terms and use lasso?
############
# random forest ?

